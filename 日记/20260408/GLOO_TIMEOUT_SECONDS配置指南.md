# GLOO_TIMEOUT_SECONDS 配置指南

## 概述

`GLOO_TIMEOUT_SECONDS` 是 PyTorch Gloo 后端的超时环境变量，用于控制分布式通信操作的超时时间（单位：秒）。默认值为 1800 秒（30 分钟）。

## RL 训练架构中的超时控制

在 RL 训练系统中，存在两种角色，各自使用不同的超时控制机制：

| 角色 | 引擎类型 | 后端 | 超时控制参数 | 配置方式 |
|------|---------|------|-------------|---------|
| **rollout** | inference | vLLM/sglang | `GLOO_TIMEOUT_SECONDS` | 环境变量 |
| **actor** | training | Megatron | `distributed_timeout_minutes` | YAML 配置 |

## 环境变量传递机制

### AsystemScheduler 流程

```
builder.py envs
    ↓
_rl_job_builder_args['envs']
    ↓
container["env"] = combined_envs
    ↓
传递给容器环境变量
    ↓
launch-worker.sh 可能覆盖
```

**注意**：`launch-worker.sh` 中设置的 `export GLOO_TIMEOUT_SECONDS=1800` 会覆盖 builder.py 传入的值。

### RayScheduler 流程

```
base_trainer.py
    ↓
extra_envs = {} (只包含特定变量)
    ↓
RayScheduler(config={"extra_envs": extra_envs, ...})
    ↓
engine_envs = default_envs.copy() + extra_envs
    ↓
runtime_env={"env_vars": engine_envs}
    ↓
Ray actor 环境变量
```

**关键点**：Ray actor 的环境变量通过 `runtime_env={"env_vars": engine_envs}` 指定，**不是继承宿主机环境**。

## 关键文件位置

### 1. default_envs 定义

```python
# areal/extension/asystem/ascheduler/__init__.py:27-50
default_envs = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "GLOO_SOCKET_IFNAME": "eth0",
    # ... 其他环境变量
}
```

### 2. engine_envs 构建

```python
# areal/extension/asystem/ascheduler/ray/__init__.py:344-354
engine_envs = default_envs.copy()       # 复制默认环境变量
engine_envs.update(self.extra_envs)      # 更新配置传入的环境变量
engine_envs.update(engine_spec.env_vars) # 更新 spec 传入的环境变量
custom_envs = {
    "TYPE": engine_spec.task_type,
    "ROLE": worker_group_key,
    "WORK_MODE": "GENERATION" if worker_group_key == "rollout" else "TRAINING",
    "IMAGE": engine_spec.image,
    "AISTUDIO_JCS_JOB_ID": os.environ["AISTUDIO_JCS_JOB_ID"]
}
engine_envs.update(custom_envs)
```

### 3. runtime_env 设置

```python
# areal/extension/asystem/ascheduler/ray/__init__.py:358-367
engine = RayEngine.options(
    scheduling_strategy=NodeAffinitySchedulingStrategy(...),
    runtime_env=ray.runtime_env.RuntimeEnv(container=container_spec, env_vars=engine_envs),
    name=engine_actor_id,
    max_restarts=0
).remote(...)
```

## Gloo 初始化调用栈

### Inference 模式 (vLLM/sglang)

```
RayEngine.initialize()
    ↓
run_in_thread(host, port, engine_type, rollout_backend)
    ↓
asyncio.run(run_engine_server(...))
    ↓
engine_server.py: EngineServer.initialize()
    ↓
online_vllm_backend.py: SPMDInferenceEngine.initialize()
    ↓
init_process_group(rank, world_size, master_addr, master_port)  (gpu_utils.py:18)
    ↓
dist.init_process_group(backend="gloo", ...)  (gpu_utils.py:30)
```

### Training 模式 (Megatron)

```
MegatronTrainEngine.initialize()
    ↓
MegatronBackend.initialize()
    ↓
init_megatron()  (megatron_helper.py:566)
    ↓
finish_mpu_init()
    ↓
torch.distributed.init_process_group(
    backend=args.distributed_backend,
    timeout=timedelta(minutes=args.distributed_timeout_minutes),  # 使用配置参数
    ...
)
```

## 解决方案

### 方案 1：在 default_envs 中添加（推荐）

```python
# areal/extension/asystem/ascheduler/__init__.py
default_envs = {
    ...
    "GLOO_SOCKET_IFNAME": "eth0",
    "GLOO_TIMEOUT_SECONDS": "3600",  # 添加这行，设置 1 小时超时
    ...
}
```

### 方案 2：在 extra_envs 中添加

```python
# areal/examples/base_trainer.py
extra_envs = {
    "GLOO_TIMEOUT_SECONDS": os.environ.get("GLOO_TIMEOUT_SECONDS", "3600"),
    ...
}
```

### 方案 3：Training 模式配置

对于 Training 模式，需要在 YAML 配置中添加：

```yaml
actor:
  distributed_timeout_minutes: 60  # 60分钟 = 3600秒
```

## 常见问题

### Q1: 为什么 builder.py 设置的 GLOO_TIMEOUT_SECONDS 不生效？

**原因**：RayScheduler 不使用 builder.py 中的环境变量，而是通过 `runtime_env={"env_vars": engine_envs}` 单独指定 Ray actor 的环境变量。

### Q2: 为什么在 Ray actor 中打印看到 3600，但实际超时是 1800？

**可能原因**：
1. `dist.is_initialized()` 已返回 True，Gloo 在其他地方初始化
2. 走的是 Training 模式，使用 `distributed_timeout_minutes` 参数
3. 环境变量在 subprocess 启动时被过滤掉

### Q3: 如何验证环境变量是否正确传递？

在以下位置添加打印验证：

```python
# 1. RayScheduler 创建 engine_envs 后
print(f"[RayScheduler] engine_envs GLOO_TIMEOUT_SECONDS = {engine_envs.get('GLOO_TIMEOUT_SECONDS', 'NOT SET')}")

# 2. RayEngine.initialize() 中
print(f"[RayEngine.initialize] GLOO_TIMEOUT_SECONDS = {os.environ.get('GLOO_TIMEOUT_SECONDS', 'NOT SET')}")

# 3. init_process_group 中
print(f"[init_process_group] GLOO_TIMEOUT_SECONDS = {os.environ.get('GLOO_TIMEOUT_SECONDS', 'NOT SET')}")
```

## 相关文件

| 文件 | 用途 |
|------|------|
| `asys_cli/builder.py` | AsystemScheduler 环境变量构建 |
| `areal/extension/asystem/ascheduler/__init__.py` | default_envs 定义 |
| `areal/extension/asystem/ascheduler/ray/__init__.py` | RayScheduler 环境变量传递 |
| `areal/extension/asystem/ascheduler/ray/actor.py` | Ray actor 定义 |
| `asystem_runtime/utils/gpu_utils.py` | init_process_group 实现 |
| `asystem_runtime/backend/online_vllm_backend.py` | vLLM 后端初始化 |
| `asystem_runtime/third_party/megatron/megatron_0_11_0/megatron_helper.py` | Megatron 初始化 |