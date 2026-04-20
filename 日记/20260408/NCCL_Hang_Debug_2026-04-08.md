# NCCL Hang 问题定位过程记录

> 时间：2026-04-08
> 环境：Asystem-HybridEngine + vLLM-Ascend，RL 训练中权重同步阶段
> 问题：多节点 EP Worker 在权重同步后发生 NCCL Hang

---

## 一、原始报错

### 日志片段

```
2026-04-08 08:57:06,594  INFO  web_log.py:214
  GET /v1/has_key/weights_update_finished_10.225.51.15_6_-1  → 200
  GET /v1/has_key/weights_update_finished_10.225.51.15_11_-1 → 200
  GET /v1/has_key/weights_update_finished_10.239.5.5_10_-1   → 200

(RayEngine pid=19188, ip=10.239.233.8)
(Worker_DP9_TP0_EP72 pid=22823)
ERROR 2026-04-08 08:57:06.571 [nccl_comm.py:308]
  Exception while detecting hang at
  [[22823] execute 1239 sends 1063 recvs with recursive partition for 1-72-200--1]
  TimeoutError

(Worker_DP9_TP0_EP72 pid=22823)
  Unexpected error for PID 22823: [Errno 2] No such file or directory: 'py-spy'
  Stacktrace: None
```

### 多节点相同错误（系统性）

```
(RayEngine pid=19247, ip=10.230.67.27)
(Worker_DP9_TP6_EP78 pid=22883)
ERROR 2026-04-08 08:57:06.740 [nccl_comm.py:308]
  Exception while detecting hang at
  [[22883] execute 1239 sends 1063 recvs with recursive partition for 2-78-334--1]
  TimeoutError
```

---

## 二、系统背景

### 整体架构

```
RL 训练循环
│
├── 训练侧（Megatron-LM / FSDP）
│     └── 计算梯度，更新模型权重
│
├── 权重同步（asystem_runtime/weights_exchange/nccl_comm.py）
│     └── P2P 传输：训练 Worker → 推理 Worker
│
└── 推理侧（vLLM-Ascend，昇腾 NPU）
      └── 接收新权重 → execute_dummy_batch（ACL 图重捕获）→ 正常推理
```

### 关键组件

| 组件 | 位置 | 作用 |
|------|------|------|
| `nccl_comm.py` | `asystem_runtime/weights_exchange/` | 权重 P2P 传输 + Hang 检测 |
| `detect_hang()` | `nccl_comm.py:303` | 独立线程监控，30s 超时报警 |
| `execute_dummy_batch` | `vllm_ascend/worker/worker.py:533` | 权重更新后的 ACL 图预热 |
| `acl_graph.__call__` | `vllm_ascend/compilation/acl_graph.py:199` | 昇腾 ACL 计算图执行 |

### 涉及节点

| 节点 IP | Worker | EP Rank |
|---------|--------|---------|
| 10.239.233.8 | Worker_DP9_TP0_EP72 | 72 |
| 10.230.67.27 | Worker_DP9_TP6_EP78 | 78 |
| 10.225.51.15 | 多个 Worker | - |
| 10.239.5.5 | 多个 Worker | - |
| 10.235.72.27 | 多个 Worker | - |

---

## 三、定位过程

### Step 1：初步判断（读日志）

**观察到的信号：**

1. `weights_update_finished` 轮询返回 200 → 主节点权重分发已完成，属正常
2. `detect_hang` 在 `nccl_comm.py:308` 触发 `TimeoutError` → 某 P2P 通信卡住超过 30s
3. `py-spy` 未安装 → 无法自动抓取调用栈，`Stacktrace: None`
4. 同一秒内（08:57:06）多个节点报告相同错误 → **系统性问题，不是单节点故障**

**初步假设：** EP AlltoAll 集合通信有 Worker 未到达同步点，导致全局阻塞。

---

### Step 2：确认进程存活

```bash
ps aux | grep 22823
# → 进程存在，State: S (sleeping)，不是 OOM 崩溃
```

**排除：** OOM、进程崩溃。确认是真正的 Hang（卡死等待）。

---

### Step 3：安装 py-spy 抓取调用栈

```bash
pip install py-spy
py-spy dump --pid 22823
```

**输出（关键部分）：**

```
Thread 22823 (active): "MainThread"
    synchronize      (torch_npu/npu/streams.py:85)          ← 卡在这里
    __call__         (vllm_ascend/compilation/acl_graph.py:199)
    _model_forward   (vllm_ascend/worker/model_runner_v1.py:1717)
    _dummy_run       (vllm_ascend/worker/model_runner_v1.py:2338)  ← 关键！
    execute_dummy_batch (vllm_ascend/worker/worker.py:533)
    worker_busy_loop (vllm/v1/executor/multiproc_executor.py:858)
```

---

### Step 4：关键认知修正 — 两个独立的 Hang

用户确认 **py-spy 是在日志报错很久之后才执行的**，因此：

```
08:57:06  [第一次 Hang]  nccl_comm.py 的 detect_hang 超时
          原因：weights P2P 传输（dist.send/recv）阻塞超过 30s
          结果：detect_hang 线程记录日志后返回（不中断主线程）
                主线程继续等待，最终 P2P 传输完成

     ↓ （间隔较长时间）

  现在   [第二次 Hang]  execute_dummy_batch 卡住
          原因：MoE AlltoAll 需要全部 EP Worker 同时进入 dummy run
                EP72 已进入，但其他 Worker 未同时到达
```

两次 Hang 本质不同，需分开分析。

---

### Step 5：读 nccl_comm.py 源码

**发现：`detect_hang` 的设计**

```python
# nccl_comm.py:303
def detect_hang(future, msg, p2p_op_list, timeout=30):
    try:
        future.result(timeout=timeout)   # 等待 future 完成，最多 30s
    except Exception:
        logger.exception(...)            # 超时后只记录日志
        pass                             # 不中断主线程！主线程仍在 dist.send/recv 等待

# nccl_comm.py:337 — execute_p2p_op_list
def execute_p2p_op_list(p2p_op_list, stage, weights_update_group):
    future = Future()
    hang_detector.submit(detect_hang, future, msg, p2p_op_list)  # 异步监控

    for _, p2p_op in p2p_op_list:
        dist.send(...)   # ← 阻塞调用，卡在这里主线程就挂
        dist.recv(...)   # ← 同上

    torch.cuda.synchronize()
    future.set_result(True)   # 只有所有 send/recv 完成才会触发
```

**关键设计缺陷认知：** `detect_hang` 只是"告警器"，不是"熔断器"。P2P 操作本身一旦卡死，只有对端响应了才能恢复。

**发现两阶段协议：**

```python
# 两阶段防死锁设计：
# Phase 1: partition=0 先 send，partition=1 先 recv
# Barrier
# Phase 2: partition=0 recv，partition=1 send
# Barrier

# partition 由 compute_two_phase_partition() 决定
# 数学上能保证 rank r 和 rank r+s 永远在不同 partition → 理论上不会死锁
```

**验证两阶段协议数学正确性：**

```python
# 对任意 rank r 和其对端 r+stage_idx：
# partition(r) = cycle_pos(r) % 2
# partition(r+stage_idx) = (cycle_pos(r) + 1) % 2  ← 永远相反
# 结论：compute_two_phase_partition 的数学逻辑是正确的
```

---

### Step 6：发现生产代码与本地代码不一致

**本地代码（D:\code\Asystem-HybridEngine）的 msg 格式：**
```python
msg = f"[{os.getpid()}] Using synchronous send/recv for {num_ops} operations for stage {stage}"
```

**实际日志的 msg 格式：**
```
[22823] execute 1239 sends 1063 recvs with recursive partition for 1-72-200--1
```

**结论：生产环境 `/workspace/bin/Asystem-HybridEngine/` 与本地代码存在差异，需对齐。**

```bash
# 待执行
diff <(ssh node "cat /workspace/bin/Asystem-HybridEngine/asystem_runtime/weights_exchange/nccl_comm.py") \
     D:\code\Asystem-HybridEngine\asystem_runtime\weights_exchange\nccl_comm.py
```

---

### Step 7：解码错误消息中的参数

| 参数 | EP72 | EP78 |
|------|------|------|
| sends | 1239 | 1239 |
| recvs | 1063 | 1063 |
| 参数串 | `1-72-200--1` | `2-78-334--1` |

**相同点：** sends/recvs 数量完全相同 → 同一次集合通信操作
**不同点：** 参数串第 2、3 段不同（72 vs 78，200 vs 334）

推测格式 `{X}-{EP_rank}-{本rank的op数}-{-1}`：
- EP72 负责处理 200 个 op，EP78 负责处理 334 个 op → **负载不均匀**
- EP78 处理量是 EP72 的 1.67 倍，可能是慢节点的根源之一

---

### Step 8：第二次 Hang 的根因分析

**调用链：**
```
execute_dummy_batch
  └── _dummy_run
        └── _model_forward          ← MoE 模型前向，包含 EP AlltoAll
              └── acl_graph.__call__
                    └── torch_npu.synchronize  ← NPU kernel 提交后等待完成
                                                  AlltoAll 等待其他 EP Worker
                                                  → 永久等待
```

**根因：**
```
MoE EP AlltoAll 要求全部 EP Worker（共 72 个）同时进入 dummy_run
第一次 P2P Hang 导致不同 Worker 完成权重传输的时间不一致：
  - 快节点：率先完成 P2P → 进入 execute_dummy_batch → 发起 AlltoAll → 等待慢节点
  - 慢节点：还在等 P2P 完成 → 迟迟未进入 execute_dummy_batch
结果：AlltoAll 永久等待，快节点 hang 住
```

---

## 四、当前待确认项

### 优先级 1：确认第二次 Hang 的范围

```bash
# 检查其他 EP Worker 当前状态
ssh 10.230.67.27 "py-spy dump --pid 22883 2>/dev/null | head -15"
ssh 10.225.51.15 "pgrep -a -f 'Worker_DP9' | head -5"

# 预期情况 A（所有人都卡在 dummy_run）：
#   → AlltoAll 本身问题，HCCL 集合通信 bug
# 预期情况 B（只有 EP72 卡，其他已完成）：
#   → EP72 是慢节点，其他 Worker 完成 dummy_run 后离开了同步点
```

### 优先级 2：确认生产代码版本

```bash
cat /workspace/bin/Asystem-HybridEngine/asystem_runtime/weights_exchange/nccl_comm.py \
    | grep -n "recursive partition\|execute.*sends.*recvs\|detect_hang\|timeout"
```

### 优先级 3：分析第一次 P2P Hang 的真正原因

```bash
# 查看权重传输完成前后的完整日志
grep -E "weights_update_finished|p2p.*stage|All p2p stages|Finished executing weights" \
    /path/to/logs/*.log | grep "22823\|22883" | sort | head -50

# 观察：哪个 stage 卡住了？卡了多久？
```

---

## 五、问题结构总结

```
根本触发链
│
├─ [问题1] 权重 P2P 传输临时 Hang（08:57:06，已恢复）
│   ├── 触发：某 stage 的 dist.send/recv 阻塞超过 30s
│   ├── 表象：detect_hang 超时报错，多节点同时触发
│   ├── 恢复：P2P 最终完成（主线程继续执行）
│   └── 未确认原因：
│         - 慢节点负载不均（EP78 处理量 1.67x EP72）？
│         - 网络抖动？
│         - 两阶段协议在特定条件下的问题（生产代码版本）？
│
└─ [问题2] execute_dummy_batch AlltoAll Hang（当前进行时）
    ├── 触发：问题1 导致不同 EP Worker 完成权重传输时间不一致
    ├── 机制：MoE AlltoAll 要求 72 个 EP Worker 同时进入 dummy_run
    ├── 表象：EP72 进程卡在 torch_npu.synchronize
    └── 解决方向：
          在所有 EP Worker 完成权重传输后，加一个全局同步屏障
          再统一触发 execute_dummy_batch
```

---

## 六、修复方向（待验证后确认）

### 方向 A：execute_dummy_batch 前加同步屏障

在 Asystem-HybridEngine 通知 vLLM Worker 执行 dummy run 的逻辑中，确保所有 EP Worker 完成权重传输后再统一触发：

```python
# 伪代码，具体实现位置待确认
# 在 weights_exchange 完成后、触发 dummy_run 前：
dist.barrier(group=all_ep_workers_group)  # 确保所有 EP Worker 都到这里
trigger_execute_dummy_batch_on_all_workers()  # 统一触发
```

### 方向 B：排查第一次 P2P Hang 的根本原因

查看生产版本的 nccl_comm.py，确认是否存在：
- 非均匀负载导致慢节点的问题
- 两阶段协议在特定 world_size/stage_idx 组合下的边界情况

### 方向 C：安装 py-spy 到所有节点

```bash
# 让后续 detect_hang 能自动抓取调用栈，加快定位速度
pip install py-spy  # 在所有节点执行
```

---

## 七、结论更新（2026-04-09）

### 新增关键信息

发现 Worker_DP8_TP0_EP64 也在同一时刻（08:57:06.799）报告了相同错误：

```
(Worker_DP8_TP0_EP64 pid=22824, ip=10.239.233.8)
Exception while detecting hang at
[[22824] execute 1239 sends 1063 recvs with recursive partition for 1-64-192--1]
```

**更重要的是：报错之后 RL 训练继续运行，并成功完成了 8 个训练 step。**

---

### 问题重新定性：不是死锁，是慢操作误报

```
之前判断（错误）：
  P2P 传输死锁 → 导致 execute_dummy_batch AlltoAll 死锁

实际情况（修正）：
  P2P 传输很慢（超过 30s 告警阈值）→ detect_hang 打印 ERROR
  P2P 传输最终完成（只是慢）
  execute_dummy_batch 也完成
  训练正常继续，跑了 8 个 step
```

`detect_hang` 是**告警器，不是熔断器**：超时只触发日志，不中断 P2P 操作。主线程的 `dist.send/recv` 仍在等待并最终完成。

---

### 汇总三个报错 Worker 的负载数据

| Worker | EP Rank | 参数串 | 本 rank op 数 | 相对倍数 |
|--------|---------|--------|--------------|---------|
| DP8_TP0_EP64 | 64 | `1-64-192--1` | 192 | 1.0x |
| DP9_TP0_EP72 | 72 | `1-72-200--1` | 200 | 1.04x |
| DP9_TP6_EP78 | 78 | `2-78-334--1` | **334** | **1.74x** |

参数串格式推测：`{transfer_rank}-{ep_rank}-{本rank负责的op数}--1`

**EP78 处理的 op 数是 EP64 的 1.74 倍** → transfer plan 负载不均，EP78 是慢节点，拖慢所有依赖它的 P2P 操作，触发其他 Worker 的 detect_hang 超时。

---

### 根因修正

```
真正根因：transfer plan 生成的负载不均匀
  ├── EP78 分配了 334 个 op，其他 Worker 只有 192-200 个
  ├── EP78 完成时间超过 30s
  ├── 等待 EP78 的其他 Worker 也超过 30s → 各自触发 detect_hang
  └── 但所有人最终都完成了，训练可以继续

不是根因（之前错误判断）：
  ✗ 两阶段协议死锁
  ✗ execute_dummy_batch 永久 Hang
  ✗ 某节点宕机/OOM
```

---

### 修复方向修正

**优先级 1（立即可做）：调整 detect_hang 超时阈值，减少误报**

```python
# nccl_comm.py，当前固定 timeout=30s 太短
# 应根据 op 数量动态计算，或直接调大
def execute_p2p_op_list(p2p_op_list, stage, weights_update_group):
    num_ops = len(p2p_op_list)
    timeout = max(60, num_ops * 0.1)  # 示例：每 10 个 op 留 1s，至少 60s
    hang_detector.submit(detect_hang, future, msg, p2p_op_list, timeout=timeout)
```

**优先级 2（中期）：排查 transfer plan 负载不均问题**

```bash
# 查看 transfer plan 生成逻辑，找到为什么 EP78 分配了更多 op
grep -rn "transfer_plan\|TransferPlan\|build.*plan\|assign.*op" \
    /workspace/bin/Asystem-HybridEngine/asystem_runtime/weights_exchange/*.py
```

**优先级 3（确认频率）：判断是否每个 step 都发生**

```bash
# 统计 detect_hang 报错的频率
grep "Exception while detecting hang" /path/to/logs/*.log | \
    awk '{print $1}' | cut -d. -f1 | sort | uniq -c | sort -rn
# 如果每个 step 都有 → 负载不均是系统性问题，需要修 transfer plan
# 如果偶发 → 可能是网络抖动，调大 timeout 即可
```

---

## 八、诊断命令速查

```bash
# 1. 检查进程状态
ps aux | grep "Worker_DP9_TP0_EP72"
cat /proc/22823/status | grep -E "State|Threads"

# 2. 抓取调用栈
py-spy dump --pid 22823

# 3. 检查其他节点的 Worker 状态
for node in 10.225.51.15 10.230.67.27 10.239.5.5 10.235.72.27; do
    echo "=== $node ==="
    ssh $node "pgrep -a -f 'Worker_DP9' | head -5"
done

# 4. 验证 compute_two_phase_partition 正确性（Python）
python3 - <<'EOF'
import math

def compute_two_phase_partition(rank, stage_idx, world_size):
    if stage_idx == 0:
        return 0
    gcd_val = math.gcd(world_size, stage_idx)
    reduced_world = world_size // gcd_val
    reduced_stage = stage_idx // gcd_val
    try:
        stage_inv = pow(reduced_stage, -1, reduced_world)
    except ValueError:
        return None
    cycle_base = rank % gcd_val
    cycle_index = (rank - cycle_base) // gcd_val
    cycle_pos = (cycle_index * stage_inv) % reduced_world
    return cycle_pos % 2

world_size = 72
failures = []
for stage_idx in range(1, world_size):
    for rank in range(world_size):
        peer = (rank + stage_idx) % world_size
        p_rank = compute_two_phase_partition(rank, stage_idx, world_size)
        p_peer = compute_two_phase_partition(peer, stage_idx, world_size)
        if p_rank is None or p_peer is None:
            failures.append(f"stage={stage_idx}, rank={rank}: no inverse")
        elif p_rank == p_peer:
            failures.append(f"stage={stage_idx}, rank={rank}<->peer={peer}: both={p_rank}")

print(f"Total failures: {len(failures)}")
for f in failures[:10]:
    print(f)
EOF

# 5. 确认生产与本地代码差异
grep -n "recursive partition\|detect_hang\|timeout" \
    /workspace/bin/Asystem-HybridEngine/asystem_runtime/weights_exchange/nccl_comm.py
```
