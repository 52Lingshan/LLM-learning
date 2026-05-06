# 16B 模型 Step 0 vLLM Sleep 显存分析报告

- **Profiling 数据**: `D:\work\KimiK2.5\profiling\step0_vllm\16B_step0_vllm_rank0_profile`（初始采集）+ `D:\work\KimiK2.5\profiling\step0_vllm_ascend_sleep\16B_step0_vllm_server_rank0_profile`（camem.py sleep 专项采集）
- **模型**: 16B MoE 模型
- **并行配置**: d2t2p4c2e2 (DP=2, TP=2, PP=4, CP=2, EP=2)
- **vLLM 引擎**: world_size=8 (TP=2, PP=4), 每节点 2 个 engine 实例
- **分析日期**: 2026-04-25

---

## 一、Step 0 的本质

Step 0 不是推理，而是 **`release_memory_occupation`** —— 训练前将 vLLM 推理显存释放给 Megatron 训练使用。

### 1.1 调用链

```
AReaL/asystem_runtime/backend/online_vllm_backend.py
  └─ release_memory_occupation()          # HTTP POST to /sleep
      └─ vLLM server /sleep endpoint
          └─ vllm_ascend/worker/worker.py:197  sleep()
              └─ CaMemAllocator.sleep()    # camem.py
                  ├─ memcpy D2H (offload 权重到 CPU)
                  ├─ unmap_and_release()    # CANN 底层释放
                  ├─ gc.collect()
                  └─ torch.npu.empty_cache()
```

### 1.2 CaMemAllocator 机制

CaMemAllocator 是基于 CANN-mem 的 PyTorch pluggable allocator（`vllm_ascend_C`），**绕过 PyTorch 标准分配器**。这意味着：
- PyTorch APP 层 profiling 只能看到 ~108 MB（PyTorch 自身的少量分配）
- 14+ GB 的模型权重和 KV cache 通过 `python_create_and_map` / `python_unmap_and_release` 直接操作 CANN 内存，不经过 PyTorch caching allocator

---

## 二、初始采集分析（step0_vllm 目录）

### 2.1 数据来源

两个采集快照：
- `*_023411707_ascend_pt` — 主采集，覆盖 sleep 全过程（4.33s）
- `*_023422692_ascend_pt` — 确认快照，sleep 完成后稳态（0.07s）

### 2.2 内存时间线

| 层级 | sleep 前 | sleep 后 | 释放量 |
|------|---------|---------|--------|
| APP (PyTorch 分配器) | ~108 MB | ~108 MB | 0（PyTorch 层无变化） |
| Device (NPU 总占用) | 18,600 MB | 4,611 MB | **-13,989 MB** |

### 2.3 Device 层下降模式

97 次内存释放事件，按大小聚类：

| 释放大小 | 次数 | 对应内容（推测） |
|----------|------|----------------|
| ~326 MB | 16 次 | MoE expert 权重（16 experts） |
| ~368 MB | 11 次 | Attention/FFN 权重块 |
| ~92 MB | 13 次 | 中等层权重 |
| ~42 MB | 17 次 | 小型参数/buffer |

### 2.4 PyTorch API 调用

- PYTORCH_API 层记录 15,201 次调用，**全部是 Python builtins**（无 aten 算子）
- 这进一步证实 sleep 过程中没有 PyTorch 计算，只有 CANN 底层内存操作

### 2.5 已跟踪 vs 未跟踪

| 来源 | 大小 | profiling 跟踪 |
|------|------|---------------|
| PyTorch allocator (APP) | 107.9 MB | ✓ |
| CANN RUNTIME module | 26.1 MB | ✓ |
| 未跟踪 | 4,480 MB | ✗ |
| **合计** | **4,614 MB** | |

NPU_MODULE_MEM 列举了 77 个 CANN 模块（HCCL, FMK, CCE, DRV...），全部显示 0 MB，只有 RUNTIME 26.1 MB。4.48 GB 的显存不是通过 CANN 标准 module 接口分配的。

---

## 三、camem.py Sleep 专项采集分析（step0_vllm_ascend_sleep 目录）

### 3.1 采集方式

在 `camem.py` 的 `sleep()` 方法中硬编码 `torch_npu.profiler`，仅 rank 0 采集：

```python
def sleep(self, offload_tags=None):
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if local_rank == 0:
        prof.start()
    # ... sleep 逻辑 ...
    gc.collect()
    torch.npu.empty_cache()
    if local_rank == 0:
        prof.step()
        prof.stop()
```

Profiler 配置：Level1, profile_memory=True, with_stack=True, 输出格式 Text。

### 3.2 数据格式

采集输出为**原始二进制格式**（未经 `msprof --export`），每条记录：
- `npu_mem.app.0.slice_0` — 32 字节/记录，APP 层内存时间线
- `npu_mem.data.0.slice_0` — 32 字节/记录，Device 层内存时间线
- `npu_module_mem.data.0.slice_0` — 24 字节/记录，按 CANN Module 分解

### 3.3 sleep() 期间完整内存变化

**采集窗口**: 2026-04-25 07:38:59 ~ 07:39:03（~3.85 秒）

| 指标 | sleep 前 | sleep 后 | 释放量 |
|------|---------|---------|--------|
| APP (PyTorch 分配器) | 16,601 MB | 2,767 MB | **-13,834 MB** |
| Device (驱动层总占用) | 18,711 MB | 4,881 MB | **-13,830 MB** |

### 3.4 按 CANN Module 分解

| Module ID | 身份（推断） | sleep 前 | sleep 后 | 变化 |
|-----------|------------|---------|---------|------|
| 33 | App 内存池（权重+KV cache） | 14,030 MB | 192 MB | **-13,838 MB** |
| 3 | **HCCL 通信 buffer** | **2,081 MB** | **2,081 MB** | **不变** |
| 45 | Profiler buffer | 150 MB | 150 MB | 不变 |
| 7 | Framework 开销 | 65 MB | 65 MB | 不变 |
| 0 | Runtime/Driver | 32 MB | 32 MB | 不变 |
| 其余 | 未使用 | 0 MB | 0 MB | — |

### 3.5 关键发现

1. **CaMemAllocator 工作正常** — Module 33 从 14 GB 平滑释放到 192 MB，每次采样间隔释放 80-90 MB
2. **HCCL 完全静态** — sleep 前后 2,081 MB 一字不差，sleep() 没有任何销毁通信组的逻辑
3. **Profiler 自身占 150 MB** — 本次采集引入的额外开销，正式运行时不存在

---

## 四、4.88 GB 残留显存构成

sleep 完成后 Device 层仍有 **4,881 MB (4.77 GB)**，分解如下：

| 组件                                       | 大小       | 占比    | 可优化       |
| ---------------------------------------- | -------- | ----- | --------- |
| **HCCL 通信 buffer** (Module 3)            | 2,081 MB | 42.6% | ✓ 可销毁通信组  |
| **Driver/Runtime gap** (Device - APP 差值) | 2,114 MB | 43.3% | ✗ 硬件栈固有   |
| Module 33 残留（allocator 基础设施常驻开销）         | 192 MB   | 3.9%  | ✗ 不可优化    |
| Profiler buffer (Module 45)              | 150 MB   | 3.1%  | ✓ 正式运行无此项 |
| Caching allocator 内部开销                   | 247 MB   | 5.1%  | ✗         |
| Runtime + Framework (Module 0+7)         | 97 MB    | 2.0%  | ✗         |

### 4.1 Module 33 残留 192 MB 根因

**本质**：pluggable allocator 基础设施的常驻开销，非应用层可释放内存。

**验证过程**：

初始假设认为 192 MB 是 MemPool free list 缓存的已释放临时 tensor。尝试了两种修复方案：

1. **`pool.snapshot()` 清理 free list** — 失败，`torch.npu.memory.MemPool` 未实现 `snapshot()` API（上游 CUDA 版 `cumem.py` 有此接口，但 torch_npu 未移植）
2. **`self.allocator_and_pools.clear()`** — 释放 Python 层 pool/allocator 对象引用

第二次采集验证结果（`step0_vllm_ascend_sleep_optimizer` 目录）：

| 指标            | 优化前          | `clear()` 后  | 变化          |
| ------------- | ------------ | ------------ | ----------- |
| Module 33 残留  | 192 MB       | **192 MB**   | 无变化         |
| APP 总量        | 2,767 MB     | 2,767 MB     | 无变化         |
| **Device 总量** | **4,881 MB** | **4,650 MB** | **-231 MB** |

**结论**：192 MB 是 NPU pluggable allocator（THPAllocator）基础设施的常驻内存——创建内存池时由 CANN 驱动分配的页表、元数据等，不受 Python 层 pool 对象生命周期控制，**不可通过应用层手段释放**。

`allocator_and_pools.clear()` 虽未清理 Module 33，但释放了 **231 MB 设备端元数据**（Device 层减少），属于有效优化。

**持久性**：192 MB 在整个进程生命周期中永远不会被释放：
- sleep()/wake_up() 均只操作 `pointer_to_data`，无法触及 allocator 基础设施
- 训练进程（Megatron）无法触及 vLLM 进程的 CANN 上下文
- 每个 sleep→wake→sleep 循环中此残留持续存在，成为训练阶段的固定显存税

### 4.2 `allocator_and_pools.clear()` 优化效果

在 `sleep()` 的 unmap 循环之前增加 `self.allocator_and_pools.clear()`，释放 pool/allocator 对象关联的设备端元数据。

**16B 模型实测**：单卡节省 231 MB Device 显存。

**K25 128K PP10 集群估算**（40 机 × 16 NPU = 640 卡）：

| 场景 | 单卡节省 | 640 卡总节省 |
|------|---------|------------|
| 保守（同 16B） | 231 MB | **144 GB** |
| 等比放大（Module 33 总量 42.6 GiB vs 14 GiB，×3） | ~693 MB | **433 GB** |

对 PP mid stage（free 低至 35 MB）而言，单卡省 231~693 MB 可能是 OOM 与否的关键差距。

> 精确值需要对 K25 vLLM server 做同样的 camem.py profiling 采集。

### 4.3 正式运行（无 profiler + `clear()` 优化）的预期残留

去掉 profiler 150 MB，加上 `clear()` 节省 231 MB：**~4,350 MB**，其中：
- HCCL: 2,081 MB（可优化，见第五节）
- Driver/Runtime: ~1,883 MB（不可优化）
- Module 33 残留: 192 MB（不可优化，allocator 基础设施）
- 其他: ~194 MB（不可优化）

---

## 五、HCCL 2,081 MB 深度分析

### 5.1 Rank 0 参与的通信组

vLLM 引擎 world_size=8（TP=2, PP=4），rank 0 参与：

| 通信组 | Rank 成员 | 大小 | 用途 |
|--------|----------|------|------|
| Default PG | [0,1,2,3,4,5,6,7] | 8 | 全局通信 |
| WORLD | [0,1,2,3,4,5,6,7] | 8 | 全局协调 |
| TP | [0,1] | 2 | Attention 层 all-reduce |
| EP | [0,1] | 2 | MoE all-to-all |
| PP | [0,2,4,6] | 4 | Pipeline send/recv |
| DP/DCP/PCP | [0] | 1 | 单卡，buffer 极小 |

### 5.2 HCCL Buffer 计算

`HCCL_BUFFSIZE=600` (MB/组)，设置于 `online_vllm_backend.py:1054`。

HCCL 对**相同 rank 集合**的通信组共享 buffer：

| 有效 Buffer | 共享的组 | Buffer 大小 |
|------------|---------|------------|
| Buffer 1 | Default PG + WORLD（都是 [0-7]） | 600 MB |
| Buffer 2 | TP + EP（都是 [0,1]） | 600 MB |
| Buffer 3 | PP（[0,2,4,6]，独立） | 600 MB |

**3 × 600 = 1,800 MB** + **~281 MB HCCL runtime 元数据** = **~2,081 MB**

### 5.3 weights_exchange 通信组

不在 2,081 MB 中。该组在 NPU 上默认 `init_every_step=True`，每次权重交换后 `destroy_process_group`，sleep 时已不存在。

- 设置: `asystem_runtime/weights_exchange/weights_reader.py:895`
- Buffer: `GROUP_HCCL_BUFFER_SIZE` 环境变量控制，默认 600 MB
- 生命周期: `weights_reader.py:918` 创建 → `weights_reader.py:955` 销毁

### 5.4 关键代码路径

| 用途 | 文件 | 行号 |
|------|------|------|
| HCCL_BUFFSIZE 设置 | `online_vllm_backend.py` | 1054 |
| vLLM 通信组创建 | `vllm/distributed/parallel_state.py` | 1271-1422 |
| GroupCoordinator new_group | `vllm/distributed/parallel_state.py` | 278-345 |
| weights_exchange 组 | `asystem_runtime/weights_exchange/util.py` | 14-57 |
| weights_exchange 生命周期 | `asystem_runtime/weights_exchange/weights_reader.py` | 910-979 |

---

## 六、优化建议

### 6.1 ~~修复 Module 33 残留~~ → `allocator_and_pools.clear()` 优化（已验证，单卡节省 231 MB）

在 `sleep()` 的 unmap 循环之前增加 `self.allocator_and_pools.clear()`，释放 pool/allocator 对象关联的设备端元数据。

- **16B 实测**：Device 层从 4,881 MB 降至 4,650 MB，单卡节省 231 MB
- **Module 33 的 192 MB 不受影响**（allocator 基础设施常驻，不可优化）
- **K25 640 卡集群预估**：保守节省 144 GB，等比放大可达 433 GB
- **注意**：`torch.npu.memory.MemPool.snapshot()` 在 torch_npu 中未实现，无法使用上游 cumem.py 的 free list 清理方案

### 6.2 回收 HCCL Buffer（预期节省 ~2 GB）

在 `CaMemAllocator.sleep()` 末尾销毁 vLLM 通信组：

```python
import torch.distributed as dist
from vllm.distributed.parallel_state import get_world_group, get_tp_group, get_pp_group

for group in [get_tp_group(), get_pp_group(), ...]:
    dist.destroy_process_group(group.device_group)
```

**代价**: `wake_up()` 需要重建所有通信组，增加唤醒延迟（预计数秒）。

### 6.3 降低 HCCL_BUFFSIZE（低成本方案）

将 `HCCL_BUFFSIZE` 从 600 降到 200：
- HCCL 占用: 3×200 + 281 = ~881 MB（节省 ~1.2 GB）
- 需评估对 TP all-reduce / PP send-recv 的通信性能影响

### 6.4 合并重复通信组

如果 Default PG 和 WORLD 可以合并为一个，EP 和 TP 在 rank 集合相同时复用，可以减少通信组数量。但这需要修改 vLLM 上游代码。

---

## 七、与其他 Step 的关系

| Step | 阶段 | Device 显存 | 说明 |
|------|------|------------|------|
| Step 0 (sleep 前) | vLLM 推理态 | ~18.7 GB | 权重 + KV cache + HCCL |
| Step 0 (sleep 后，优化前) | 释放完成 | ~4.9 GB | HCCL + Driver 残留 |
| **Step 0 (sleep 后，`clear()` 优化)** | **释放完成** | **~4.65 GB** | **节省 231 MB** |
| Step 1 (训练) | Megatron 训练 | 峰值 19.7 GB | 在残留基础上训练分配 |
| Step 3 (稳态训练) | Megatron 训练 | 峰值 19.0 GB | 训练内存收敛 |

sleep 后的残留成为训练阶段的**显存基线**，直接影响训练可用显存上限。`clear()` 优化将基线从 4.9 GB 降至 4.65 GB。
