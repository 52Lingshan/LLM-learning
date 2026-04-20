# RayEngine Save 操作 OOM 问题排查

## 问题概述

**日期**: 2026-04-14

**现象**: AReaL RL 训练过程中，save checkpoint 时 RayEngine 进程崩溃，导致保存失败。

## 错误日志摘要

### 客户端错误
```
[ATinker Server] (RayWorker pid=114090) 20260413-15:07:21.776 SyncRPCServer ERROR: Engine method 'save' failed: 
error_type: TrainEngineError, sub_error: SaveError, 
reason: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

### 关键错误链
```
1. RemoteDisconnected: Remote end closed connection without response
2. Connection refused: [Errno 111] Connection refused (重试时)
3. Worker unexpectedly exits with a connection error code 2
```

## 排查过程

### 1. 初步分析

**错误链分析**:
```
RayEngine (port 30714) ← HTTP server for save operations
        ↓
save() called → POST /save to RayEngine
        ↓
RayEngine crashes (OOM or SIGSEGV)
        ↓
Connection dropped → RemoteDisconnected
        ↓
Worker dies → port 30714 no longer listening
        ↓
Retry → Connection refused
```

### 2. 排除显存 OOM

用户反馈：
- 显存总量: 64G
- 使用峰值: 44G

**结论**: 显存充足，排除 GPU/NPU OOM。

### 3. 查找 RayEngine 日志

```bash
# 找到 RayEngine 日志文件
grep -r "30714" /tmp/ray/session_*/logs/

# 日志文件路径
/tmp/ray/session_latest/logs/worker-db5bac4efe26da6f3985d13c994caed917c266385b596a216b3a433b-04000000-114083.out
```

### 4. 分析日志时间线

| 时间 | 事件 |
|------|------|
| 14:54:37 | RayEngine 启动，端口 30714 |
| 14:54:42 | HTTP 服务器就绪 |
| 14:55:48 | 初始化成功 |
| 15:05:43 | 训练 step 0 完成 |
| 15:05:55 | 收到 save 请求 |
| 15:06:07 | 开始 save_hf，内存 25.24 GB |
| **15:07:21** | **连接断开（进程崩溃）** |

日志在 15:06:07 后突然中断，没有错误信息，说明进程被"突然杀死"。

### 5. 检查系统日志

```bash
dmesg | grep -i "killed\|oom\|sigkill\|sigsegv" | tail -30
```

**关键发现**:
```
[8240192.131108] Memory cgroup out of memory: Killed process 4019963 (ray::RayEngine) 
total-vm:8848364644kB, anon-rss:33547980kB, file-rss:598648kB
```

## 根本原因

### 确认：系统内存 OOM（不是显存）

| 指标 | 值 | 说明 |
|------|-----|------|
| `anon-rss` | **32 GB** | 进程使用的匿名内存 |
| `constraint` | `CONSTRAINT_MEMCG` | 容器/Pod 内存限制 |
| 被杀原因 | **cgroup 内存超限** | 不是显存，是系统 RAM |

### 为什么会 OOM？

**HuggingFace 格式转换**会在 **CPU 内存**上进行：
1. 加载模型权重到 CPU
2. 转换格式（需要额外内存）
3. 保存到磁盘

Pod/容器有内存限制，转换过程中超出了这个限制。

## 解决方案

### 方案 1：增加 Pod 内存限制（推荐）

```yaml
# 在 Kubernetes Pod 配置中
resources:
  limits:
    memory: "64Gi"  # 或更高
  requests:
    memory: "48Gi"
```

### 方案 2：使用原生格式保存

```python
# 修改 save 参数，跳过 HuggingFace 转换
save_type: 'megatron'  # 或 'torch_dist'，不使用 'huggingface'
```

### 方案 3：分片保存

如果支持，将模型分片保存，减少单次内存占用。

## 检查命令参考

```bash
# 查看 cgroup 内存限制
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# 查看当前内存使用
cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes

# 查找 RayEngine 日志
grep -r "30714" /tmp/ray/session_*/logs/

# 检查 OOM 日志
dmesg | grep -i "killed\|oom" | tail -30
```

## 总结

| 问题 | 状态 |
|------|------|
| 直接原因 | RayEngine HTTP 服务器崩溃 |
| 根本原因 | **系统内存 OOM（cgroup 限制）** |
| 触发点 | HuggingFace 格式转换消耗大量 CPU 内存 |
| 次要问题 | `rotary_emb.inv_freq` 未正确保存（警告，非崩溃原因） |

## 经验教训

1. **显存 ≠ 系统内存**: GPU 显存充足不代表 CPU 内存充足
2. **容器环境需关注 cgroup 限制**: 即使物理机内存大，容器可能有限制
3. **HuggingFace 转换内存消耗大**: 大模型转换需要额外内存，需预留足够空间
4. **日志中断 = 进程被杀**: 没有错误信息突然中断，通常是 OOM 或 SIGKILL