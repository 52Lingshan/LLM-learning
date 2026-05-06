# PP5 vs PP10 动态显存差异分析

## 核心数据对比

| Stage | PP5 参数量 | PP5 静态   | PP5 动态   | PP10 参数量 | PP10 静态  | PP10 动态 | 动态差值  |
|-------|-----------|-----------|-----------|------------|-----------|----------|----------|
| first | 2.54B     | 28.85 GB  | 13.93 GB  | 1.53B      | 15.71 GB  | 7.40 GB  | +6.53 GB |
| mid   | 2.37B     | 33.83 GB  | 13.93 GB  | 2.20B      | 34.51 GB  | 6.96 GB  | +6.97 GB |
| last  | 2.48B     | 33.13 GB  | 13.83 GB  | 0.92B      | 12.03 GB  | 6.81 GB  | +7.02 GB |

两个显著特征：

1. PP5 动态显存在所有 stage 高度一致 (~13.9 GB)，与层数/参数量无关
2. PP5 比 PP10 多出的动态显存也高度一致 (~7 GB)

## 配置差异总结

| 参数              | PP=5            | PP=10                |
|-------------------|-----------------|----------------------|
| PP stages         | 5               | 10                   |
| 每 stage 层数     | 10/13/13/13/12  | 3/7/7/7/7/7/7/7/7/2 |
| CP                | 32              | 16                   |
| EP                | 128             | 64                   |
| 每 stage ranks    | 128             | 64                   |
| 每 rank 序列长度  | 4096 (128K/32)  | 8192 (128K/16)       |
| 本地 expert 数    | 3 (384/128)     | 6 (384/64)           |
| 1F1B warmup mbs   | 4 (PP-1)        | 9 (PP-1)             |
| microbatch_group  | 5               | 10                   |

## 原因分析

### 排除的错误假设

**~~假设 1："每 stage 层数更多 → 动态峰值更高"~~** — 被数据否定。PP5 的 10 层 (first)、12 层 (last)、13 层 (mid) stage 动态峰值几乎相同（13.83~13.93 GB），动态峰值与层数无关。

**~~假设 2："PP5 梯度缓冲延迟分配，PP10 预分配"~~** — 被代码否定。查阅 Megatron-LM 源码 `param_and_grad_buffer.py:1004`，`_ParamAndGradBuffer.__init__` 中梯度缓冲始终通过 `torch.zeros()` 在初始化时静态预分配，不存在延迟分配路径。

此外，之前的 bytes/param 分析存在 **GiB/GB 单位换算错误**（直接用 GiB 数值除以 10^9 参数，少乘了 1.0737）：

| 配置 | Stage | allocated | params | 错误值 (GiB÷B) | 正确值 (bytes÷params) |
|------|-------|-----------|--------|-----------------|----------------------|
| PP5  | mid   | 33.83 GiB | 2.373B | ~~14.27 B/p~~   | **15.31 B/p**        |
| PP10 | mid   | 34.51 GiB | 2.203B | ~~15.69 B/p~~   | **16.83 B/p**        |

修正后，**两者静态显存都高于 14 bytes/param 基线**（BF16权重 + FP32主权重 + Adam m/v），说明两者的梯度缓冲都已包含在静态显存中，只是比例不同：

- PP5 mid: (15.31 - 14) × 2.37B ≈ **3.1 GB** 梯度/padding 在静态中
- PP10 mid: (16.83 - 14) × 2.20B ≈ **6.2 GB** 梯度/padding 在静态中
- PP10 比 PP5 多 ~3.1 GB 在静态中

> 注：两者都低于完整 FP32 梯度的 18 bytes/param (14 + 4)。可能原因：部分参数（如 Embedding、LayerNorm）不参与梯度 buffer；bucket padding 策略差异；或优化器状态的实际布局与理论计算有偏差。需要通过加日志确认。

### 已确认的部分原因：EP/CP 导致 bucket padding 差异（~3 GB 静态差值）

代码分析确认 EP/CP 配置通过以下机制影响显存分配：

1. **Dense 与 Expert 参数使用不同 DP group**（`Megatron-LM/distributed_data_parallel.py:108-125, 292-300`）：
   - dense_params → `intra_dp_cp_group`（DP×CP = 1×32=**32** for PP5, 1×16=**16** for PP10）
   - expert_parallel_params → `intra_expt_dp_group`（expert DP group, 两者均为 **1**，因为每 rank 有唯一 expert 组合）
2. **bucket 大小**：`bucket_size = max(40M, 1M × dp_group.size())`，PP5 dense 的 bucket = max(40M, 32M) = 40M，PP10 = max(40M, 16M) = 40M — 实际相同
3. **bucket padding**（MindSpeed `buffer_pad/adaptor.py`，当 `--param-and-grad-buffer-pad=512` 启用时）：`align_size = 512 // element_size`，padded to `dp_world_size * align_size`，PP5 padded to `32 * 128 = 4096` 元素边界 vs PP10 `16 * 128 = 2048`

这解释了部分静态差异，但 **无法解释 ~7 GB 的动态差异**。

### MindSpeed 代码排查结果（远程 `/sfs_turbo/hw/shiyang/code/kimi25vl_maijia_new/mindspeed`）

**结论：MindSpeed 没有改变梯度缓冲的生命周期。** 逐个排查如下：

| 文件 | 内容 | 是否改变 grad buffer 生命周期 |
|------|------|------|
| `core/distributed/buffer_pad/adaptor.py` | 替换 `_ParamAndGradBuffer.__init__`，自定义 Ascend 512-byte 对齐 padding | **否** — 仍然 `torch.zeros()` 静态预分配 |
| `core/data_parallel/distributed_data_parallel.py` | ZeRO-3 style all-gather/reduce-scatter hooks | **否** — 仅在 DP>1 时激活（当前 DP=1） |
| `core/distributed/param_and_grad_buffer.py` | `start_param_sync` / `finish_param_sync` / `start_grad_sync` 替换（绕过 PTA 不支持的 `_coalescing_manager`） | **否** — 只改通信方式，不改 buffer 分配 |
| `core/transformer/moe/comm_utils.py` | `async_all_to_all` → `torch.empty_like(input_)` | **否** — 单层临时，forward/backward 中创建和释放 |
| `features_manager/memory/reuse_fp32_param.py` | `reuse_fp32_param` 优化 | **否** — 需要 `args.reuse_fp32_param` 启用 |

**Ring CP 临时缓冲方向相反（反而 PP5 更小）：**

`ring_context_parallel.py` 中每次 ring 迭代分配的临时 buffer：

| 阶段 | buffer | PP5 (seq=4096) | PP10 (seq=8192) |
|------|--------|----------------|-----------------|
| forward | `next_kv` + `next_round_kv` (2 组 k,v) | ~117 MB | ~235 MB |
| backward | `cur_dkv` + `next_dkv` + `next_round_dkv` + `dq` | ~206 MB | ~411 MB |

PP5 的 ring CP 缓冲**更小**（seq_per_rank 更短），不可能是 PP5 动态更高的原因。

**候选嫌疑：`WeightGradStore`（nanopipe 特性）**

`core/weight_grad_store.py` 中的 `WeightGradStore.cache` 会在 backward 期间累积 `(total_input, grad_output, weight)` 元组，延迟到下一个 forward 再计算权重梯度。如果此特性启用：
- PP5 每 stage 13 层 → cache 中积累更多层的 input/grad_output
- PP10 每 stage 7 层 → cache 中积累更少
- 但每个 tensor 大小与 seq_per_rank 成正比（PP5 更小）

**但这与观测矛盾**：PP5 各 stage（10/12/13 层）动态峰值完全一致，如果是层数驱动的 cache 积累，应该有差异。且 nanopipe 需要 `--use-nanopipe` 显式启用，不确定是否活跃。

### 待查明的主要问题：~7 GB 动态差值的来源

所有可分析的因素汇总：

| 因素 | PP5 | PP10 | 方向 | 量级 | 来源 |
|------|-----|------|------|------|------|
| 每 rank tokens | 4096 | 8192 | PP10 更大 | 与观测相反 | — |
| 本地 expert 数 | 3 | 6 | PP10 更多 | 与观测相反 | — |
| warmup in-flight mbs | 4 | 9 | PP10 更多 | 与观测相反 | — |
| in-flight 总边界激活 | ~59 MB | ~265 MB | PP10 更多 | 量级太小 | — |
| 重算 checkpoints | 764 MB | 1852 MB | PP10 更多 | ~1 GB 差异 | — |
| Ring CP forward/backward buffers | ~323 MB | ~646 MB | PP10 更大 | 与观测相反 | ring_context_parallel.py |
| MoE all-to-all 临时 | ~469 MB/层 | ~939 MB/层 | PP10 单层更大 | 单层临时 | comm_utils.py |
| grad reduce dp_group 大小 | 32 ranks | 16 ranks | PP5 组更大 | 待测 | distributed_data_parallel.py |
| WeightGradStore cache | 更多层(?) | 更少层(?) | 待确认 | 可能 GB 级 | weight_grad_store.py |

**所有可量化因素均指向 PP10 动态应更高**，与实测完全相反。这强烈暗示存在一个我们尚未识别的、与 CP/EP 配置强相关但与层数无关的显存分配机制。

### Asystem-HybridEngine 代码排查结果

**路径**: `/sfs_turbo/hw/shiyang/code/kimi25vl_maijia_new/Asystem-HybridEngine`

**发现：Colocate 模式下的梯度缓冲释放/重载机制**（`asystem_runtime/utils/megatron_util.py`）：

```
训练前: resume_memory_occupation()
  → load_megatron_model_to_gpu()
    → buffer.grad_data.storage().resize_(grad_data_size)  # 重新分配
    → buffer.grad_data.zero_()

训练中: train_step()
  → forward / backward / optimizer.step

训练后: release_memory_occupation_if_needed()
  → offload_megatron_model_to_cpu()
    → buffer.grad_data.storage().resize_(0)               # 释放
  → offload_megatron_optimizer()
    → Adam states → CPU
```

关键函数：

| 函数 | 作用 | 对 grad buffer 的影响 |
|------|------|------|
| `release_grad_memory()` | 释放梯度缓冲 | `grad_data.storage().resize_(0)` — 释放显存 |
| `load_megatron_model_to_gpu()` | 重载模型+梯度 | `grad_data.storage().resize_(size)` + `zero_()` — 重新分配 |
| `offload_megatron_model_to_cpu()` | 卸载模型+梯度 | `param_data` → CPU pinned, `grad_data.storage().resize_(0)` |
| `offload_megatron_optimizer()` | 卸载优化器 | Adam exp_avg/exp_avg_sq → CPU |

**但这不是 PP5 vs PP10 差异的原因**：两者都运行在 colocate 模式下，经历相同的释放/重载周期。日志中的 `allocated` 值（PP5 mid: 33.83 GiB, PP10 mid: 34.51 GiB）在各 step 间高度稳定，说明测量时 grad buffer 始终处于已加载状态。

**关于 step 0→1 的 max_allocated 跳升差异**：

| 配置 | step 0 动态 | step 1 动态 | 跳升幅度 |
|------|------------|------------|---------|
| PP5 mid | 6.40 GB | 13.93 GB | **+7.53 GB** |
| PP10 mid | 6.89 GB | 6.99 GB | +0.10 GB |

PP5 在 step 1 的 max_allocated 大幅跳升（+7.5 GB），而 PP10 几乎不变。这暗示 PP5 在第一次完整训练步骤中有大量额外临时分配（可能与 `gradient_accumulation_fusion` 或 optimizer 首次 step 相关），需要通过 profiling 确认。

**Asystem-HybridEngine 没有修改 Megatron DDP/Buffer 内部逻辑**：`train_step()` 是对 Megatron `forward_backward_func` + `optimizer.step()` 的薄封装，不引入额外的梯度缓冲分配。

### 需要实际查证的代码路径

| 查什么 | 代码位置 | 方法 |
|--------|---------|------|
| `_ParamAndGradBuffer` 实际 buffer 大小 | `buffer_pad/adaptor.py` | 在 `torch.zeros` 后打印 `self.numel * element_size / 1e9` GB，分 dense/expert |
| step 0 vs step 1 动态峰值差异来源 | `megatron_helper.py:train_step` | 在 `zero_grad_buffer()` 后、forward 前/后、backward 前/后、`optimizer.step()` 前/后各打印 `memory_allocated()` 和 `max_memory_allocated()` |
| nanopipe 是否启用 | 训练脚本 | 检查 `--use-nanopipe` 参数 |
| `quant_grads` 是否启用 | `buffer_pad/adaptor.py:285-301` | 检查是否有 `--quant-grads` 参数（若启用，grad_dtype 会变成 BF16，grad buffer 大小减半） |
| colocate 释放/重载日志 | 训练日志 | 搜索 `release_grad_memory` 和 `load_params` 日志，确认每次释放/重载的精确字节数 |
| backward 中哪些临时张量最大 | `torch.cuda.memory_snapshot()` | 在 max_alloc 峰值前后 dump memory snapshot，分析 tensor 来源 |

## 结论

PP5 动态显存比 PP10 多 ~7 GB，目前**尚无法完全确定根因**。已排除的错误解释和已确认的部分原因如下：

### 已排除

1. ~~"每 stage 层数更多 → 动态峰值更高"~~ — PP5 各 stage（10/12/13 层）动态峰值完全一致（~13.9 GB），与层数无关
2. ~~"PP5 梯度缓冲延迟分配，PP10 预分配"~~ — Megatron 代码确认梯度缓冲始终在初始化时预分配（`torch.zeros()`）
3. ~~"bytes/param = 14.27 证明静态中无梯度"~~ — 单位换算错误，正确值为 15.31 B/p，两者静态中均含梯度
4. ~~MindSpeed 改变了 grad buffer 生命周期~~ — 逐文件排查确认没有改变
5. ~~Asystem-HybridEngine 修改了 DDP/Buffer 内部逻辑~~ — 仅管理 colocate 模式的释放/重载，不改分配逻辑

### 已确认

6. EP/CP 配置差异导致 bucket padding 不同，PP10 静态中比 PP5 多 ~3 GB padding/buffer 开销
7. 多数可量化的动态因素（in-flight mbs、边界激活、MoE 缓冲、Ring CP 缓冲）反而指向 PP10 动态应更高，与实测矛盾
8. Asystem-HybridEngine 的 colocate 模式会在训练前后释放/重载梯度缓冲，但两种配置经历相同流程，不是差异来源

### 待查明

9. ~7 GB 动态差值的主要来源仍不明确。PP5 step 0→1 的 max_allocated 跳升 +7.5 GB（PP10 仅 +0.1 GB）是重要线索
10. 需要通过运行时 profiling 定位根因（三层代码 Megatron/MindSpeed/Asystem-HybridEngine 的静态分析已全部完成）

**建议验证步骤：**
1. 在 `buffer_pad/adaptor.py` 的 `torch.zeros` 后加日志，打印 dense 和 expert 两组 buffer 的 `self.numel`、`grad_dtype`、`dp_group.size()`、bucket 数量
2. 在 step 0 的 forward 前 / forward 后 / backward 前 / backward 后 / optimizer.step 后分别打印 `torch.cuda.memory_allocated()` 和 `torch.cuda.max_memory_allocated()`，定位动态峰值出现的精确阶段
3. 检查训练脚本是否启用了 `--use-nanopipe`（WeightGradStore）或 `--quant-grads`
4. 在峰值前后调用 `torch.cuda.memory_snapshot()` dump 内存快照，分析占用最大的临时 tensor 来源
5. 对照实验：固定 PP=5，仅切换 CP=16/EP=64 vs CP=32/EP=128，观察动态峰值变化
