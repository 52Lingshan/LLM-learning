# vLLM ZMQ 端口竞态条件问题分析

## 问题现象

```
zmq.error.ZMQError: Address already in use (addr='tcp://10.233.190.93:45137')
ERR99999 UNKNOWN application exception
```

vLLM 启动时，ZMQ 套接字绑定失败，端口已被占用。最后的 `ERR99999` 是昇腾 NPU（CANN）的级联报错，非独立问题。

## 根因

多个 worker/进程在启动时并发调用 `get_open_port()`，存在 TOCTOU（Time-Of-Check-Time-Of-Use）竞态条件：

```
进程A: get_open_port() → 发现端口45137空闲
进程B: get_open_port() → 发现端口45137空闲（A还没绑定）
进程A: zmq.bind("tcp://host:45137") → 成功
进程B: zmq.bind("tcp://host:45137") → EADDRINUSE ✗
```

---

## 三个修复 PR 对比分析

### PR #28636 — Socket Holding（已关闭未合并）

**方案**：先用 TCP socket 占住端口，等 ZMQ 绑定后再释放。

```python
# 新增 get_and_hold_open_port()
held_socket, port = get_and_hold_open_port()
zmq_socket.bind(f"tcp://host:{port}")
held_socket.close()  # ZMQ绑定后释放
```

**问题**：
- ZMQ bind 和 TCP socket close 之间仍有极短时间窗口
- 调用方需要管理 socket 生命周期，实现较复杂
- **已关闭未合并**，无法通过升级获得

---

### PR #30520 — Late Binding 完整版（Open，有 Bug）

**方案**：ZMQ 直接 bind 到端口 0，OS 原子分配端口，通过 `last_endpoint` 获取实际地址。

```python
# 绑定到 wildcard 地址
zmq_socket.bind("tcp://host:0")
# 获取 OS 分配的实际端口
actual_endpoint = zmq_socket.last_endpoint.decode("utf-8")
```

**在 `--api-server-count 1` 下存在 Bug**：

单 server 路径中，拿到实际地址后**没有回写给 addresses**，导致 engine core 收到的仍是 `tcp://host:0`，无法连接，30秒后超时：

```
TimeoutError: Timed out waiting for engines to send initial message on input socket.
```

具体缺失代码：
```python
self.input_socket, actual_input_address = make_zmq_socket(...)
# ❌ 漏了这两行
# addresses.inputs[0] = actual_input_address
# addresses.outputs[0] = actual_output_address

with launch_core_engines(vllm_config, addresses, ...):  # 仍用 tcp://host:0
    ...
```

**时序对比**：

```
PR #30520（有Bug）：
  bind("tcp://host:0") → actual="tcp://host:54321"
  ❌ addresses.inputs[0] 仍是 "tcp://host:0"
  engine connect("tcp://host:0") → 超时

PR #35977（正确）：
  bind("tcp://host:0") → actual="tcp://host:54321"
  ✅ addresses.inputs[0] = "tcp://host:54321"
  engine connect("tcp://host:54321") → 正常
```

---

### PR #35977 — Late Binding 精简版（Open，推荐）

**方案**：与 #30520 相同的 Late Binding，但更完整。

**额外修复了 #30520 遗漏的问题**：

1. **地址回写**（单 server 路径）：
```python
self.input_socket, actual_input = make_zmq_socket(..., return_address=True)
addresses.inputs[0] = actual_input  # ✅ 回写实际地址

self.resources.output_socket, actual_output = make_zmq_socket(..., return_address=True)
addresses.outputs[0] = actual_output  # ✅ 回写实际地址
```

2. **`split_zmq_path` 的 port 0 隐藏 Bug**：
```python
# 旧代码：port=0 时 0 是 falsy，返回空字符串
port = str(parsed.port or "")  # ❌ 0 → ""

# PR #35977 修正：
port = str(parsed.port) if parsed.port is not None else ""  # ✅
```

3. **明确区分单/多 server 场景**：
```python
# vllm/v1/engine/utils.py
use_late_binding = num_api_servers == 1  # 单server用late binding，多server用Pipe回传
```

---

## 三个 PR 总览对比

| 维度 | PR #28636 | PR #30520 | PR #35977 |
|------|-----------|-----------|-----------|
| 状态 | 已关闭未合并 | Open | Open |
| 方案 | Socket Holding | Late Binding | Late Binding |
| 根治竞态 | 不彻底 | 是 | 是 |
| `--api-server-count 1` | 可用 | ❌ 超时 Bug | ✅ 正常 |
| `split_zmq_path` port 0 bug | 未修 | 未修 | ✅ 已修 |
| 代码复杂度 | 中 | 较高 | 低 |
| **推荐** | 否 | 否 | **是** |

---

## 结论与建议

**推荐 cherry-pick PR #35977**，两个 PR 均未合并，需手动应用。

### 最小改动集（针对 `--api-server-count 1`）

| 文件 | 改动内容 |
|------|---------|
| `vllm/utils/network_utils.py` | 加 `is_wildcard_addr`、`_resolve_bound_address`、`make_zmq_socket(return_address=True)` overload；修 `split_zmq_path` port 0 bug |
| `vllm/v1/utils.py` | `get_engine_client_zmq_addr` 加 `late_binding` 参数 |
| `vllm/v1/engine/utils.py` | `use_late_binding = num_api_servers == 1` |
| `vllm/v1/engine/core_client.py` | 改用 `return_address=True`，**回写实际地址** |
| `vllm/distributed/device_communicators/shm_broadcast.py` | 改用 port 0 + `last_endpoint` |

`coordinator.py` 的 Pipe 部分是多 server 场景，单 server 可暂不改。

### 应用 PR 前的临时规避

若暂时无法 cherry-pick，可错开 worker 启动时间：

```bash
worker0 &
sleep 2
worker1 &
sleep 2
worker2 &
```

---

## 参考

- [PR #28636](https://github.com/vllm-project/vllm/pull/28636) — fix(network): prevent port allocation race condition
- [PR #30520](https://github.com/vllm-project/vllm/pull/30520) — [BugFix] Use late binding to avoid zmq port conflict race conditions
- [PR #35977](https://github.com/vllm-project/vllm/pull/35977) — [Bugfix] Fix DP port conflict race condition with late binding
- Issue [#28498](https://github.com/vllm-project/vllm/issues/28498) — 原始 bug report
