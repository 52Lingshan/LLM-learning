# 16B 模型 Step 1 vs Step 3 动态显存对比分析

- **Profiling 文件**: `D:\work\KimiK2.5\profiling\ascend_pytorch_profiler_0.db` (5.5GB, Step 1)
- **对比文件**: `D:\work\KimiK2.5\profiling\ASCEND_PROFILER_OUTPUT\ascend_pytorch_profiler_0.db` (4.4GB, Step 3)
- **模型**: 16B MoE 模型
- **并行配置**: d2t2p4c2e2 (DP=2, TP=2, PP=4, CP=2, EP=2)
- **分析日期**: 2026-04-24

---

## 一、全局对比总览

| 指标              | Step 1 (本次)  | Step 3 (上次) | 差异                   |
| --------------- | ------------ | ----------- | -------------------- |
| 采集时长            | 1194s        | 1210s       | 相近                   |
| **训练前静态基线**     | **10.91 GB** | **14.0 GB** | **-3.09 GB**         |
| **训练后静态基线**     | **13.98 GB** | **14.0 GB** | 相同                   |
| 绝对峰值 (alloc)    | 16.28 GB     | 17.63 GB    | -1.35 GB             |
| 绝对峰值 (reserved) | 16.97 GB     | 18.99 GB    | -2.02 GB             |
| 动态增量 (基于训练前基线)  | +5.37 GB     | +3.63 GB    | **Step 1 多 1.74 GB** |
| 动态增量 (基于训练后基线)  | +2.30 GB     | +3.63 GB    | Step 1 少 1.33 GB     |

**核心发现**: Step 1 的训练前基线比 Step 3 低 ~3GB，因为 **Adam 优化器状态 (exp_avg, exp_avg_sq) 在 Step 1 首次初始化**，此前不存在。

---

## 二、全局时间线

```
Step 1 ─ alloc (GB)
         │
  16.28 ─┤ · · · · · · · · · · · · · · · · · · · · · · · · ·╭─╮ ◀ 绝对峰值 (ckpt save)
         │                                                   │  │
  14.78 ─┤ · · · · · · · · · · · · · · · · · · ★ · · · · · ·╯  ╰─╮ ◀ ★ optimizer 训练峰值
         │                              ╭─╮      ╰──────────╯     │   (Adam init + sqrt)
  14.48 ─┤ · · · · · · · · · · · ·╭─╮· │  │ ·                     │
         │                        │  │  │  │                       │
  13.95 ─┤ · · · · · · · · · · · ·╯  ╰━━╯  ╰━━━━━━━━━━━━━━━━━━━━━━╯ ◀ 训练后基线
         │                       ╱ optimizer init (zeros_like ×370)
  12.60 ─┤ · · · · · · · · · · ╱ · · ·
         │                    ╱  Adam states 斜坡 (+3.07GB)
  10.91 ━┿━━━━━━━━━━━━━━━━━━━╱━━━━━━━                              ◀ 训练前基线
         │                   ↑
         ╰──────┬────────────┬────┬──────┬────────┬──────────→ t(s)
                0           400  488   491       510    620
               推理         │fwd/bwd│  Adam    ckpt  offload
                           finalize  init+sqrt

Step 3 ─ alloc (GB)
         │
  17.63 ─┤ · · · · · · · · · · · · · ·╭─╮ · · · · · ◀ 绝对峰值 (finalize)
         │                            │  │
  15.80 ─┤ · · · · · · · · · · · · ·╭╯  ╰╮ · · · · ·
         │                          │      │
  14.00 ━┿━━━━━━━━━━━━━━━━━━━━━━━━━━╯      ╰━━━━━━━━━ ◀ 稳态基线 (Adam 已存在)
         ╰──────┬──────────────────┬─────┬──────────→ t(s)
                0                 400   481  482
               推理               推理  finalize
```

---

## 三、Step 1 独有现象: Adam 优化器状态首次初始化

### 3.1 现象

训练前基线 10.91GB → 训练后基线 13.98GB，永久增加 **+3.07 GB**。

### 3.2 时间与调用栈

```
t=490.533s 开始, 持续约 0.4s 完成全部分配

调用栈:
  train_step
    → optimizer.step()
      → megatron/core/optimizer/distrib_optimizer.py(2597): step_with_ready_grads
        → torch/optim/adam.py(213): step
          → torch/optim/adam.py(138): _init_group
            → torch.zeros_like(param)  ← 为每个参数创建 exp_avg 和 exp_avg_sq
```

### 3.3 分配明细

| 指标 | 值 |
|------|-----|
| 总分配次数 | **370 次** `aten::zeros_like` |
| 总大小 | **3.08 GB** |
| 最大单次 | 145 MB × 2 (最大的两组参数) |
| 典型单次 | 23 MB × 2 (成对出现: exp_avg + exp_avg_sq) |
| 存活时长 | **126 秒** (持续到 offload 才释放) |

### 3.4 分配过程

```
alloc (GB)
          │
  13.95 ━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮━━━━━━━━━━ ◀ 训练后新基线
          │                             ╱
  13.00 ──┤ · · · · · · · · · · · · ·╱ · ·              Adam states 斜坡
          │                        ╱       ← 370 次 zeros_like
  12.00 ──┤ · · · · · · · · · · ╱           逐个为每个参数创建
          │                   ╱               exp_avg & exp_avg_sq
  11.00 ──┤ · · · · · · · · ╱ · ·
          │               ╱
  10.91 ━━┿━━━━━━━━━━━━━━╱━━━━━━━━━━━━━━━━━━━━━━━━━━━ ◀ 训练前基线
          │              ↑ 斜坡开始
          ╰──────────────┬──────┬──→ t(s)
                      490.5  491.0
                       ◀─ 仅 0.4s ─▶
```

### 3.5 为什么 Step 3 没有这个跳升?

Step 3 的基线已经是 14.0GB —— Adam 状态在 Step 1 初始化后就一直存在。Colocate offload 时会把 Adam states 搬到 CPU (`offload_megatron_optimizer`)，reload 时搬回 GPU，但 **不需要重新 zeros_like**。

---

## 四、动态显存分阶段对比

### 4.1 Forward / Backward 阶段

| 算子 | Step 1 峰值 | Step 3 峰值 | 变化 |
|------|------------|------------|------|
| `npu_grouped_matmul` (MoE) | 369 MB | 369 MB | 相同 |
| `npu_fusion_attention` | 111 MB | 111 MB | 相同 |
| `npu_fusion_attention_grad` | 93 MB | 100 MB | 略小 |
| `embedding_dense_backward` | 336 MB | 336 MB | 相同 |
| 同时刻最大共存 | ~0.5 GB | ~0.5 GB | **相同** |

**结论**: Forward/Backward 的动态显存在 Step 1 和 Step 3 之间完全一致。

### 4.2 finalize_model_grads 阶段

| 指标 | Step 1 | Step 3 | 变化 |
|------|--------|--------|------|
| flat_buffer #1 (aten::cat) | 1,791 MB | 1,791 MB | **完全相同** |
| flat_buffer #2 (aten::cat) | 1,791 MB | 1,791 MB | **完全相同** |
| 合计动态 | +3.58 GB | +3.58 GB | 相同 |
| #1 存活时长 | 68.1ms | 66.1ms | 相近 |
| #2 存活时长 | 2.5ms | 0.6ms | 略长 |
| 峰值 alloc | 14.48 GB | 17.63 GB | 基线不同导致 |

**关键差异**: Step 1 的 finalize 发生在 Adam 初始化**之前** (基线还是 10.91GB)，所以峰值 = 10.88 + 1.79 + 1.79 = **14.48 GB**。而 Step 3 已包含 Adam 状态 (基线 14.0GB)，所以峰值 = 14.0 + 1.79 + 1.79 = **17.63 GB**。

```
  Step 1 finalize                      Step 3 finalize
  alloc (GB)                           alloc (GB)
         │                                    │
  14.48 ─┤ · · ╭─╮ · ◀ peak               17.63 ─┤ · · ╭─╮ · ◀ peak
         │    │  │                                │    │  │
  12.69 ─┤ ·╭╯  ╰╮· ·                     15.83 ─┤ ·╭╯  ╰╮· ·
         │  │      │                              │  │      │
  10.88 ━┿━━╯      ╰━ ◀ 基线低            14.00 ━┿━━╯      ╰━ ◀ 基线高
         ╰──┬──────┬→ t(s)                        ╰──┬──────┬→ t(s)
         488.8   489.0                           481.0    481.1
```

### 4.3 Optimizer step 阶段

| 指标 | Step 1 | Step 3 | 变化 |
|------|--------|--------|------|
| **Adam 首次 init** | **+3.070 GB (永久)** | 无 | **Step 1 独有** |
| `_foreach_sqrt` 临时 buffer | 0.831 GB (56 个并发 sqrt) | 0.831 GB | 相同 |
| 单次最大 `_foreach_sqrt` | 145 MB | 145 MB | 相同 |
| sqrt 存活时长 | ~23ms | ~23ms | 相同 |
| **Step 1 训练期间峰值** | **14.784 GB** | — | **★ Step 1 训练真实峰值** |

**`_foreach_sqrt` 调用栈**:

```
optimizer.step()
  → torch/optim/adam.py: step
    → torch/optim/adam.py: _multi_tensor_adam   ← 多张量融合 Adam
      → aten::_foreach_sqrt   ← 对所有 v (exp_avg_sq) 一次性求 sqrt
```

PyTorch 的 `_multi_tensor_adam` 一次性对所有参数的 v 求 `sqrt(v) + eps`，需要与所有 v 同样大小的临时输出 buffer。56 个 `_foreach_sqrt` 调用（按 dtype/device 分组的 fused 调用），合计临时分配 831MB，算完后立即用于 `param -= lr * m / (sqrt(v) + eps)`，~23ms 后释放。

**Step 1 训练期间精确对账**:

```
Step 1 训练期间 max_alloc: 14.784 GB
─────────────────────────────────────
基线 (训练前):                10.880 GB
+ Adam states (永久):         + 3.070 GB  (370 次 zeros_like)
+ _foreach_sqrt (临时):       + 0.831 GB  (56 个并发 sqrt)
+ 残差:                       + 0.004 GB  (测量误差)
─────────────────────────────────────
合计:                         14.785 GB ✓
```

### 4.4 Checkpoint 保存阶段

| 指标 | Step 1 | Step 3 | 变化 |
|------|--------|--------|------|
| HF 转换 buffer | 671 MB × 4 | 671 MB × 4 | 相同 |
| 峰值 alloc | **16.28 GB** | 16.35 GB | 相近 |
| 位置 | t=510.5s | t=492.6s | 时间偏移 |

**重要**: Step 1 的**绝对峰值 16.28GB 出现在 checkpoint save 阶段**（而非 finalize），因为此时 Adam 状态已初始化，基线已经是 13.93GB，671MB × 2 buffer 共存时达到 16.28GB。

---

## 五、完整显存构成对比

### Step 1 训练过程中的显存变化

```
阶段              alloc     构成
──────────────────────────────────────────────────────────
推理稳态           10.91GB   权重(BF16+FP32) + grad_buffer (已prealloc)
                            ┌───────────────────────────┐
                            │ 4个 ParamAndGradBuffer:    │
                            │ 3324+2825+1661+1412=9.2GB │
                            │ + 小参数 ~1.7GB            │
                            └───────────────────────────┘

Forward/Backward   ~11.4GB   基线 + 临时激活/梯度 (~0.5GB)
                  (peak)

finalize_model     14.48GB   基线 + flatten buffer ×2 (3.58GB)
_grads (peak)               释放后回到 10.88GB

Optimizer step     13.95GB   基线 + Adam states 首次 init (+3.07GB)
                  (永久)     ┌───────────────────────────┐
                            │ 370次 zeros_like:          │
                            │ exp_avg + exp_avg_sq       │
                            │ 合计 3.07GB, 永久存在       │
                            └───────────────────────────┘
                            基线从此变为 13.95GB

Optimizer 临时     14.78GB   新基线 + _foreach_sqrt (0.831GB)
(训练期间峰值)     ★          ┌───────────────────────────┐
                            │ 56个并发 _foreach_sqrt:     │
                            │ 最大单次 145MB              │
                            │ 合计 0.831GB, ~23ms 后释放   │
                            └───────────────────────────┘
                            ★ Step 1 训练期间的真实峰值

Checkpoint save    16.28GB   新基线 + HF转换 buffer (671MB×2+335MB×N)
(绝对峰值)                   ★ 整个 Step 1 的最高点 (含 ckpt)

offload            0.07GB    全部搬到 CPU
```

### Step 3 训练过程中的显存变化

```
阶段              alloc     构成
──────────────────────────────────────────────────────────
推理稳态           14.0GB    权重 + grad_buffer + Adam states (已存在)

Forward/Backward   ~14.5GB   基线 + 临时激活/梯度 (~0.5GB)

finalize_model     17.63GB   基线 + flatten buffer ×2 (3.58GB)
_grads (峰值)               ★ 整个 Step 3 的最高点
                            释放后回到 14.0GB

Optimizer 临时      ~14.85GB  基线 + _foreach_sqrt 等 (~0.85GB)
                            (Adam 状态已存在, 无 init)

Checkpoint save    16.35GB   基线 + HF转换 buffer

offload            0.07GB    全部搬到 CPU
```

---

## 六、关键结论

### 6.1 Step 1 vs Step 3 的根本差异

| 问题                          | 答案                                                                |
| --------------------------- | ----------------------------------------------------------------- |
| 为什么 Step 1 基线从 10.9→14.0GB? | Adam 优化器 exp_avg/exp_avg_sq **首次初始化** (+3.08GB, 370 次 zeros_like) |
| 这些新增显存会释放吗?                 | 训练期间**不释放** (dur=126s)，直到 colocate offload 搬到 CPU                 |
| 为什么 Step 3 没有这个跳升?          | Step 3 的 Adam 状态已存在，reload 时从 CPU 搬回 GPU，无需 zeros_like            |
| Step 1 训练期间峰值在哪?            | optimizer step 阶段 (14.78GB)，而非 finalize (14.48GB)                 |
| Step 1 绝对峰值在哪?              | checkpoint save 阶段 (16.28GB)，含 ckpt buffer                        |
| Step 3 绝对峰值在哪?              | finalize_model_grads 阶段 (17.63GB)                                 |

### 6.2 两个 Step 的不变量

| 指标 | Step 1 | Step 3 | 结论 |
|------|--------|--------|------|
| `_flatten_dense_tensors` buffer | 1791MB × 2 | 1791MB × 2 | 与 step 无关，由模型结构决定 |
| Forward/Backward 临时峰值 | ~0.5GB | ~0.5GB | recompute 策略不变 |
| Optimizer 临时 (_foreach_sqrt) | 0.831GB | 0.831GB | 56 个并发 sqrt, ~23ms 释放 |
| Checkpoint save buffer | 671MB × 4 | 671MB × 4 | HF 转换逻辑一致 |

### 6.3 Step 1 训练期间峰值定位

Step 1 训练期间的真实峰值不是 `finalize_model_grads` (14.48GB)，而是 `optimizer step` (14.78GB)：

```
峰值时刻图解                              alloc (GB)
                                                   │
  15.00 ─ · · · · · · · · · · · · · · · · · · · ·╭╮ · ★ 14.78GB
         │                              ╱         ││     (Adam init + sqrt)
  14.78 ─┤ · · · · · · · · · · · · · ╱· · · · · ·╯│ ◀──────────────────
         │                           ╱              │
  14.48 ─┤ · · · · · ╭╮ · · · · · ╱ · · · · · · · │ ◀ 14.48GB (finalize)
         │          ╱  ╲          ╱                 │
  13.00 ─┤ · · · ·╱· · ·╲ · · ·╱ · · · · · · · ·  │
         │       │        ╲   ╱                     │
  10.91 ━┿━━━━━━━╯         ╲╱━━━━━━━━━━━━━━━━━━━━━━╯  ◀ 基线
         ╰────┬──────────┬───────────────┬──────────→ t(s)
            fwd/bwd   finalize          optimizer
                     (flatten)          (Adam init
                                         + sqrt)
  注意：这两个峰值不会叠加，因为 finalize 的
        flatten buffer 在 Adam init 前已释放
```

在 Adam 状态已存在的稳态（Step 3+），`finalize_model_grads` 才是绝对峰值 (17.63GB)；但在 Step 1 这种特殊情况下，optimizer step 才是训练期间的绝对峰值 (14.78GB)。

### 6.4 Step 0→1 max_allocated 跳升机制

```
Step 0→1 跳升分解:
├── Adam exp_avg + exp_avg_sq (永久):   +3.07GB (zeros_like ×370)
├── finalize flatten buffer (临时):     +3.58GB (cat ×2, step 0 可能未触发完整 finalize)
├── _foreach_sqrt 临时 (Adam step):     +0.83GB (56 个并发 sqrt)
└── 其他小残差:                          +0.05GB
```

Step 0 通常只执行推理或部分前向，不触发完整的 optimizer.step()，因此 Adam 状态在 Step 1 首次初始化时产生一次性跳升。后续 step 不再有此跳升。

---

## 七、显存占比饼图

### Step 1 绝对峰值 (16.28GB) 构成

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│   ParamAndGradBuffer (权重+梯度)     9.22GB  56.6%   │
│   ████████████████████████████████████                │
│                                                      │
│   Adam States (exp_avg+exp_avg_sq)   3.08GB  18.9%   │
│   ████████████                                       │
│                                                      │
│   Checkpoint Save Buffer             2.35GB  14.4%   │
│   ██████████                                         │
│                                                      │
│   小参数 + 杂项                       1.63GB  10.0%   │
│   ██████                                             │
│                                                      │
└──────────────────────────────────────────────────────┘

### Step 3 绝对峰值 (17.63GB) 构成

┌──────────────────────────────────────────────────────┐
│                                                      │
│   ParamAndGradBuffer (权重+梯度)     9.22GB  52.3%   │
│   ████████████████████████████████                    │
│                                                      │
│   flatten buffer ×2 (finalize)      3.58GB  20.3%   │
│   ██████████████                                     │
│                                                      │
│   Adam States (exp_avg+exp_avg_sq)  3.08GB  17.5%   │
│   ████████████                                       │
│                                                      │
│   小参数 + 杂项                      1.75GB   9.9%   │
│   ██████                                             │
│                                                      │
└──────────────────────────────────────────────────────┘
```
