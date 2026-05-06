# 16B 模型 NPU Profiling 动态显存分析报告 (Step 3)

- **Profiling 文件**: `D:\work\KimiK2.5\profiling\ASCEND_PROFILER_OUTPUT\ascend_pytorch_profiler_0.db`
- **文件大小**: 4.4GB SQLite 数据库
- **分析日期**: 2026-04-24
- **模型**: 16B MoE 模型
- **并行配置**: d2t2p4c2e2 (DP=2, TP=2, PP=4, CP=2, EP=2)
- **采集配置**: 单 rank (rank 0), torch_npu.profiler, with_stack=True

---

## 一、Profiling 数据概览

### 1.1 数据库关键表

| 表名 | 行数 | 说明 |
|------|------|------|
| MEMORY_RECORD | 2,588,382 | 显存时间序列 (timestamp, totalAllocated, totalReserved) |
| OP_MEMORY | 431,397 | 单次显存分配记录 (size, allocationTime, releaseTime, op name) |
| PYTORCH_API | — | PyTorch 算子调用栈 (startNs, endNs, name, callchainId) |
| PYTORCH_CALLCHAINS | 79,365,008 | 调用栈详情 |
| NPU_MEM | 121,124 | NPU 硬件层面显存 |

### 1.2 全局时间线 (总采集 1210 秒)

```
时间(s)    alloc(GB)   阶段
────────────────────────────────────────────────────
0~470       14.0       推理 serving (模型权重在 GPU, 稳态)
470~478     14.0       推理 → 训练切换 (empty_cache)
478~590     14.0~17.6  ★ 训练阶段 (train_batch, 动态峰值)
492~610     14.0       checkpoint 保存
611~614     14.0→0.09  offload to CPU (colocate 释放)
614~726     0.09       过渡
726~1163    0.09       空闲 (推理引擎占用)
1163~1210   0.09→14.0  reload (resume_memory_occupation)
```

### 1.3 全局显存曲线

```
alloc(GB)
  │
18 ┤                                              ╭─╮ peak 17.63 GB
   │                                            ╭─╯ ╰─╮
16 ┤                                           ╭╯      ╰╮
   │                                          ╭╯        ╰╮
14 ┼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯ ←train→  ╰━━━━━━━━━━━  ← serving
   │                                                       ╲ offload
12 ┤                                                        ╲
   │                                                         ╲
 0 ┼───────────────────────────────────────────────────────── ╲━━━━━━━  ← reload
   ├──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────→
   0     100    200    300    400    500    600    700    800   1200  time(s)
```

---

## 二、静态显存组成 (基线 ~14.0GB)

### 2.1 四个 ParamAndGradBuffer

静态显存由 4 个通过 `param_and_grad_buffer.py:979 reload_from_cpu` 的 `storage().resize_()` 分配的大 buffer 构成：

| Buffer | 大小       | 累计      | 存储对象 (推测)                | 分配调用栈                                     |
| ------ | -------- | ------- | ------------------------ | ----------------------------------------- |
| #1     | 1,412 MB | 1.41 GB | dense 参数 FP32 梯度 buffer  | `reload_from_cpu` → `storage().resize_()` |
| #2     | 2,825 MB | 4.24 GB | expert 参数 FP32 梯度 buffer | 同上                                        |
| #3     | 1,661 MB | 5.86 GB | dense 参数 BF16 权重 buffer  | 同上                                        |
| #4     | 3,324 MB | 9.18 GB | expert 参数 BF16 权重 buffer | 同上                                        |

**合计**: ~9.2 GB (加上 Embedding、LayerNorm 等小参数 → ~14.0 GB 基线)

### 2.2 Colocate 生命周期

这 4 个 buffer 经历 3 次 resize 周期，对应 colocate 模式的释放/重载：

| 时间 | 事件 | 调用来源 |
|------|------|---------|
| 616.3s | 首次 reload (save 后) | `save_mcore_checkpoint` → `resume_memory_occupation` → `load_megatron_model_to_gpu` |
| 737.1s | 第二次 reload | `write_weights` → `_release_memory_for_weights_exchange` → `load_megatron_model_to_gpu` |
| 1163.5s | 第三次 reload | `handle_events` → `resume_memory_occupation` → `load_megatron_model_to_gpu` |

每次释放通过 `grad_data.storage().resize_(0)` 将显存归零，reload 通过 `grad_data.storage().resize_(size)` 重新分配。

---

## 三、动态显存分析 (训练阶段 478~492s)

### 3.1 动态显存时间线

```
alloc(GB)
         │
  17.63  ┤ · · · · · · · · · · · · · · · · · · · · · · ╭─╮ · ·  ← 绝对峰值
         │                                             │ │
  15.83  ┤ · · · · · · · · · · · · · · · · · · · · ╭──╯ ╰──╮ · ·
         │                                        ╭─╯        ╰─╮
  14.85  ┤ · · · · · · · · · · · · · · · · · · · ·│             │ ╭─╮ ·  ← Optimizer
         │         ╭╮    ╭╮    ╭╮      ╭╮    ╭╮   │             ╰─╯ ╰╮
  14.50  ┤ · · · ╭─╯╰────╯╰────╯╰─╮ ╭─╯╰────╯╰─╮  │                  │
         │       │                 ╰─╯            ╰─╯                  │
  14.00  ┼━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━  ← 静态基线
         ├──────┬──────────────────┬───┬──────────┬───┬────────────────→
               478               480  481        483  484             time(s)
         │      │                  │   │           │
         │      ├─ Forward  (6层)  │   │           └─ Optimizer step 结束
         │      └─ Backward (6层)  │   └─ Optimizer step 开始
         │                         │
         │                └─ finalize_model_grads  ★ 峰值在这里!
```

### 3.2 精确显存快照 (峰值前后)

| 时间 | 事件 | alloc | reserved | 动态增量 |
|------|------|-------|----------|---------|
| 480.9s | finalize 前 | 14.09 GB | 15.41 GB | +0.09 GB |
| 481.015s | flat_buffer #1 分配 | 15.83 GB | 17.20 GB | +1.83 GB |
| **481.018s** | **flat_buffer #2 分配 (峰值)** | **17.63 GB** | **18.99 GB** | **+3.63 GB** |
| 481.019s | flat_buffer #2 释放 | 15.83 GB | 18.99 GB | +1.83 GB |
| 481.081s | flat_buffer #1 释放 | 14.02 GB | 18.99 GB | +0.02 GB |
| 481.1s | finalize 结束 | 14.02 GB | 18.99 GB | +0.02 GB |

**关键观察**: 峰值持续时间极短 (~1ms)，但 reserved 从 15.41GB 涨到 18.99GB **没有降回来** (PyTorch 缓存碎片)。

---

## 四、峰值调用栈分析

### 4.1 峰值完整调用栈

```
train_step                                                  (481s, 持续481秒)
  └→ forward_backward_pipelining_without_interleaving       (481092ms)
       └→ finalize_model_grads                              (128.6ms)
            └→ _allreduce_non_tensor_model_parallel_grads   (97.5ms)
                 └→ _flatten_dense_tensors                  (20.8ms)
                      └→ aten::flatten_dense_tensors
                           └→ aten::cat ← 分配 1791MB × 2
                                │
                                ▼
                           HcclAllreduce                    (0.6ms)
                           跨 non-TP ranks 聚合梯度
```

### 4.2 图解: _flatten_dense_tensors 做了什么

```
模型中 "非张量并行" 的参数 (Embedding, LayerNorm 等):

  embed.grad    ln1.grad    ln2.grad    ···    lnN.grad
  ┌──────────┐  ┌────────┐  ┌────────┐        ┌────────┐
  │ 散布于内存 │  │        │  │        │  ···   │        │
  └────┬─────┘  └───┬────┘  └───┬────┘        └───┬────┘
       │             │           │                  │
       └─────────────┴───────────┴──────────────────┘
                               │
                               ▼
              _flatten_dense_tensors
              (torch._utils._flatten_dense_tensors)
                               │
                    ╔══════════╧═══════════════════════╗
                    ║  flat_buffer #1                  ║
                    ║  [embed │ ln1 │ ln2 │ ··· │ lnN] ║  1791 MB
                    ║  连续内存，一次性 allreduce 效率高 ║
                    ╚══════════╤═══════════════════════╝
                               │
                        allreduce(flat_buffer)
                        需要独立输出 buffer
                               │
                    ╔══════════╧═══════════════════════╗
                    ║  flat_buffer #2                  ║
                    ║  allreduce output buffer         ║  1791 MB
                    ╚══════════════════════════════════╝

  两个 buffer 短暂共存 → 峰值 14.0 + 1.79 + 1.79 = 17.63 GB ✓

  时间线:
  ───┬──────────────────────────┬──────────────────────────────┬───→
  481.015s                  481.018s                        481.019s
     │← flat_buffer #1 分配    │← flat_buffer #2 分配(峰值)   │← #2 释放
     │                         │  17.63 GB                    │
     │                         └──────────────────────────────┘ 存活 ~1ms
     │
     └─────────────────────────────────────────────────────────── #1 释放 at 481.081s
```

### 4.3 两个 flat_buffer 的生命周期

| Buffer | 分配时间 | 释放时间 | 存活时长 | 大小 |
|--------|---------|---------|---------|------|
| #1 (flatten 输出 / allreduce 输入) | 481.015s | 481.081s | **66.1ms** | 1,791 MB |
| #2 (allreduce 输出) | 481.018s | 481.019s | **0.6ms** | 1,791 MB |

**结论**: 两个 buffer 都是临时的，allreduce 完成后立即释放。峰值仅持续 ~1ms。

---

## 五、各阶段动态显存分类汇总

### 5.1 Forward / Backward 阶段 (478~481s)

所有分配均为临时，用完即释放，不会累积 (因 recompute_num_layers=1)。

| 算子 | 单次最大 | 次数 | 说明 |
|------|---------|------|------|
| `npu::npu_grouped_matmul` | 369 MB | ×6 | MoE Expert 矩阵乘 (forward + backward) |
| `npu::npu_grouped_matmul` | 185 MB | ×6 | MoE Expert 矩阵乘 (较小维度) |
| `aten::embedding_dense_backward` | 336 MB | ×2 | Embedding 层反向梯度 |
| `npu::npu_fusion_attention` | 111 MB | ×16 | FlashAttention 前向 |
| `npu::npu_fusion_attention_grad` | 100 MB | ×16 | FlashAttention 反向 |
| `aten::matmul` | 46 MB | ×38 | 各种矩阵乘 |
| `npu::npu_swiglu_backward` | 40 MB | ×10 | SwiGLU 激活反向 |
| `aten::scatter_add_` | 34 MB | ×12 | MoE token dispatch |
| `aten::mul` | 28 MB | ×84 | 逐元素乘法 |
| `npu::_npu_dtype_cast` | 28 MB | ×18 | 精度转换 |
| `npu::npu_rms_norm` / `_backward` | 18 MB | ×48 | RMSNorm 前向+反向 |
| `aten::silu` | 14 MB | ×36 | SiLU 激活 |
| `aten::sigmoid` | 14 MB | ×24 | Sigmoid (gating) |

**同一时刻最大共存**: ~0.5 GB (通常只有 1~2 个算子的临时 buffer 同时存在)

### 5.2 finalize_model_grads 阶段 (481.0~481.1s) ★ 绝对峰值

| 分配 | 大小 | 存活 | 算子 |
|------|------|------|------|
| flat_buffer #1 (flatten 输出) | 1,791 MB | 66ms | `aten::cat` via `_flatten_dense_tensors` |
| flat_buffer #2 (allreduce 输出) | 1,791 MB | 0.6ms | `aten::cat` via `_flatten_dense_tensors` |

**峰值贡献**: +3.58 GB

### 5.3 Optimizer step 阶段 (481.2~484s)

| 算子 | 单次最大 | 次数 | 说明 |
|------|---------|------|------|
| `aten::_foreach_sqrt` | 145 MB | ×56 (并发) | Adam: sqrt(v), 合计 **0.831 GB** |
| `aten::_foreach_*` 系列 | 19~38 MB | ×多 | Adam: m/v 更新、权重更新 |

**峰值贡献**: 0.831 GB (481.4s 时 alloc=14.85GB)

**`_foreach_sqrt` 调用栈**: `optimizer.step()` → `_multi_tensor_adam` → `aten::_foreach_sqrt`。PyTorch 的融合 Adam 一次性对所有参数的 v 求 `sqrt(v) + eps`，56 个按 dtype/device 分组的 fused 调用，合计临时分配 831MB，~23ms 后释放。

### 5.4 Checkpoint 保存阶段 (492~610s)

调用栈: `save` → `save_hf_checkpoint` → `save_hf_pretrained` → `save_hf_weights` → `_save_generator_distributed` → `stream_weights_megatron_to_hf`

| 分配 | 大小 | 算子 | 说明 |
|------|------|------|------|
| HF 权重转换 buffer | 671 MB × 4 | `aten::cat` / `aten::empty_strided` | Megatron → HF 格式转换 |
| TP gather buffer | 336 MB × 5 | `aten::empty_like` / `aten::empty` | TP rank 间 gather 权重 |

**峰值**: 16.35 GB (动态 +2.35 GB)

### 5.5 汇总: 动态峰值占比

| 阶段 | 峰值贡献 | 占绝对峰值(3.63GB) | 同时存在? |
|------|---------|-------------------|----------|
| **finalize_model_grads** | **+3.58 GB** | **98.6%** | 独立时段 |
| Optimizer step (_foreach_sqrt) | +0.831 GB | 22.9% | 独立时段 |
| Forward/Backward | +0.5 GB | 13.8% | 独立时段 |
| Checkpoint save | +2.35 GB | — | 训练后 |

> 各阶段不同时存在, 绝对峰值 = 基线 14.0 + finalize 的 3.58 = **17.63 GB**

---

## 六、PyTorch 缓存碎片分析

### 6.1 reserved vs allocated

| 时间点 | allocated | reserved | 碎片 (reserved - allocated) |
|--------|-----------|----------|-----------------------------|
| 训练前 (470s) | 14.05 GB | 15.41 GB | 1.36 GB |
| 训练峰值 (481s) | 17.63 GB | 18.99 GB | 1.36 GB |
| 训练后 (481.1s) | 14.02 GB | **18.99 GB** | **4.97 GB** |
| offload 前 (610s) | 14.00 GB | 18.99 GB | 4.97 GB |

**关键问题**: finalize_model_grads 的两个 1791MB buffer 释放后, allocated 回到 14GB, 但 reserved 保持在 18.99GB 不降。这意味着 PyTorch 缓存池多占了 ~5GB 显存, 直到 offload 才释放。

### 6.2 碎片成因

`_flatten_dense_tensors` 分配的 1791MB 大块释放后进入 PyTorch block cache, 但后续训练中不会再有这么大的连续分配需求, 导致这块缓存无法被复用, 形成碎片。

---

## 七、结论

1. **动态显存峰值 (17.63GB) 几乎 100% 由 `_flatten_dense_tensors` 造成** — 它在 `finalize_model_grads` 阶段把所有非 TP 梯度 cat 成连续 buffer 做 allreduce, 瞬间分配 1791MB × 2 = 3.58GB

2. **Forward/Backward 阶段动态贡献很小 (~0.5GB)** — 得益于 `recompute_num_layers=1` 的激活重算, 激活几乎不累积

3. **所有动态分配都会释放**, 但 PyTorch 缓存池会保留已释放的大块 (reserved 从 15.4GB 涨到 19.0GB 不降), 导致后续可用显存减少约 5GB

4. **`_flatten_dense_tensors` 的 buffer 大小由非 TP 参数的梯度总量决定**, 与并行配置 (CP/EP/PP) 强相关。不同并行配置下 flat buffer 大小会显著不同
