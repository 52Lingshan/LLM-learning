# Kimi-K2.5 RL 训练显存分析报告 (PP=5 配置)

- **日志文件**: `log_aistudio-7z10ju6c_2026-04-20_03-23-44.txt`
- **分析日期**: 2026-04-20

---

## 一、硬件与训练概况

### 1.1 集群配置

| 项目 | 值 |
|------|-----|
| 硬件 | 华为昇腾 NPU, 每卡 ~61.27 GB HBM |
| 集群规模 | 40 节点 × 16 NPU = 640 NPU |
| 模型 | Kimi-K2.5 MoE (类 DeepSeek-V2 架构, 61 层) |
| 精度 | BF16, preserve_fp32_weights=True |
| Colocate 模式 | 训练和推理共享同一组 NPU |
| 分配模式 | vllm[rollout]:d5t8p1e128 + megatron[actor]:d1t4p5c32e128 |

### 1.2 并行策略

| 并行维度                   | 值     | 说明                             |
| ---------------------- | ----- | ------------------------------ |
| TP (Tensor Parallel)   | 4     | 模型权重按 attention head 切分        |
| PP (Pipeline Parallel) | **5** | First:10层, Mid:13层×3, Last:12层 |
| EP (Expert Parallel)   | 128   | MoE expert 分布在 128 个 rank      |
| CP (Context Parallel)  | 32    | 128K 序列切分到 ~4K/rank            |
| DP (Data Parallel)     | 1     | 无数据并行, 优化器状态未分片                |

**PP 层数分配**: 61 层分为 5 个 PP stage: Stage 0 (first) = 10 层, Stage 1-3 (mid) = 13 层/stage × 3 = 39 层, Stage 4 (last) = 12 层。每个 PP stage 占 8 个节点 (TP=4 × CP=32 = 128 ranks = 8 节点 × 16 NPU)。

### 1.3 模型参数分布

| PP Stage | 层数 | 每 TP rank 参数量 | 说明 |
|----------|------|-------------------|------|
| Stage 0 (first) | 10 层 | 2.54B (2,543,293,936) | Embedding + 10 Transformer 层 |
| Stage 1-3 (mid) | 13 层/stage | 2.37B (2,372,943,872) | 13 MoE Transformer 层 |
| Stage 4 (last) | 12 层 | 2.48B (2,484,018,176) | 12 Transformer 层 + Output Head |

### 1.4 显存优化配置

| 配置项 | 值 | 效果 |
|--------|-----|------|
| recompute_granularity | full | 全层激活重计算 |
| recompute_num_layers | 1 | 每 1 层重计算, 激活显存极低 |
| sequence_parallel | True | 激活按 TP 维度切分 |
| use_distributed_optimizer | True | 分布式优化器 (但 DP=1 无分片效果) |
| data_parallel_sharding_strategy | no_shard | 优化器状态不分片 |
| overlap_grad_reduce | True | 梯度归约与计算重叠 |
| overlap_p2p_comm | True | PP 通信与计算重叠 |

---

## 二、三类 PP Stage 静态显存对比

日志中发现三类不同 torch allocated 的节点, 分别对应 PP=5 中的 first / mid / last stage:

| PP Stage | 代表节点 | 层数 | 节点数 | 参数量/TP | allocated | max_alloc | free(step1) |
|----------|----------|------|--------|-----------|-----------|-----------|-------------|
| Stage 0 (first) | 10.233.123.60 | 10 层 | 8 节点 | 2.54B | **28.85 GB** | 42.78 GB | 4.53 GB |
| Stage 4 (last) | 10.237.63.63 | 12 层 | 8 节点 | 2.48B | **33.13 GB** | 46.96 GB | 3.22 GB |
| Stage 1-3 (mid) | 10.234.9.85 | 13 层 | 24 节点 | 2.37B | **33.83 GB** | 47.76 GB | 9.12 GB |

**关键发现**:
- PP=5 配置下三类 stage 的 allocated 差异较小 (28.85~33.83 GB, 最大差 17%)，比 PP=10 配置均衡得多
- PP mid (24 节点 / 60% NPU) 的 allocated 最高但 free 最多，反而 **PP first 和 PP last 的 free 更少**
- PP first (28.85 GB allocated) 虽然参数最多 (2.54B) 但 allocated 最低，可能因为 Embedding 层无 MoE expert 参数

---

## 三、各 Step 显存增长趋势 (共 4 步, Step 0-3)

### 3.1 PP first - Stage 0 (Node 10.233.123.60, device 0)

| Step  | allocated (GB) | reserved (GB) | max_alloc (GB) | max_resv (GB) | free (GB) | occupy (GB) |
| ----- | -------------- | ------------- | -------------- | ------------- | --------- | ----------- |
| 0     | 28.85          | 36.04         | 33.28          | 36.04         | 17.36     | 43.91       |
| **1** | **28.85**      | **48.62**     | **42.78**      | **50.01**     | **4.53**  | **56.74**   |
| 2     | 28.85          | 41.12         | 42.78          | 50.01         | 12.09     | 49.18       |
| **3** | **28.84**      | **48.62**     | **42.78**      | **50.01**     | **4.54**  | **56.73**   |

### 3.2 PP last - Stage 4 (Node 10.237.63.63, device 0)

| Step | allocated (GB) | reserved (GB) | max_alloc (GB) | max_resv (GB) | free (GB) | occupy (GB) |
|------|----------------|---------------|----------------|---------------|-----------|-------------|
| 0 | 33.13 | 42.56 | 39.04 | 42.56 | 10.85 | 50.42 |
| **1** | **33.14** | **50.15** | **46.96** | **53.36** | **3.22** | **58.05** |
| 2 | 33.14 | 44.62 | 46.96 | 53.36 | 8.75 | 52.52 |
| 3 | 33.14 | 48.37 | 46.96 | 53.36 | 5.00 | 56.27 |

### 3.3 PP mid - Stage 1-3 (Node 10.234.9.85, device 0)

| Step | allocated (GB) | reserved (GB) | max_alloc (GB) | max_resv (GB) | free (GB) | occupy (GB) |
|------|----------------|---------------|----------------|---------------|-----------|-------------|
| 0 | 33.83 | 44.88 | 40.23 | 44.88 | 8.65 | 52.62 |
| 1 | 33.83 | 44.25 | 47.76 | 52.01 | 9.12 | 52.15 |
| 2 | 33.83 | 41.57 | 47.76 | 53.24 | 11.21 | 50.06 |
| **3** | **33.83** | **50.80** | **47.76** | **53.24** | **2.44** | **58.83** |

### 3.4 趋势总结

| 指标 | 变化规律 | 结论 |
|------|---------|------|
| torch allocated (静态显存) | 全程恒定: first 28.85, last 33.13~33.14, mid 33.83 | 模型权重+优化器状态完全固定 |
| torch max_allocated (动态峰值) | Step 0→1 跳升后稳定: first +9.5 GB, last +7.9 GB, mid +7.5 GB | 首次训练建立梯度缓冲后收敛 |
| torch reserved (缓存池) | 大幅波动 (锯齿形): first 36→48 GB, last 42→50 GB, mid 44→50 GB | PyTorch 缓存碎片化, 每步数据长度不同 |
| max_reserved | 单调递增: first 36→50 GB, last 42→53 GB, mid 44→53 GB | 碎片累积, 只升不降 |
| device mem_free | 最低值: first 4.53 GB, last 3.22 GB, mid 2.44 GB | 全部 stage 都很紧张! |

**与 PP=10 配置的关键差异**: PP=5 下各 stage 显存更均衡, 但**所有 stage 都接近 OOM**, 没有明显的 "安全 stage"。PP=10 时 first/last 有大量余量, 但 mid 极度紧张。

---

## 四、特定 Rank (节点/Device) 显存对比

### 4.1 全部 40 节点 Step 0 Device 0 对比

| PP Stage | 节点数 | allocated | 代表节点 |
|----------|--------|-----------|----------|
| Stage 0 (first, 10层) | 8 节点 | 28.85 GB | 10.233.123.60 等 (+1 本地节点) |
| Stage 4 (last, 12层) | 8 节点 | 33.13~33.14 GB | 10.237.63.63, 10.236.12.63 等 |
| Stage 1-3 (mid, 13层) | 24 节点 | 33.83 GB | 10.234.9.85, 10.235.111.87 等 |

### 4.2 PP first 同节点 16 张卡 (10.233.123.60, Step 1)

| Device | allocated (GB) | reserved (GB) | max_alloc (GB) | free (GB) |
|--------|----------------|---------------|----------------|-----------|
| 0 | 28.85 | 48.62 | 42.78 | 4.53 |
| 1 | 28.85 | 44.58 | 42.79 | 8.79 |
| 2 | 28.84 | 46.00 | 42.78 | 7.11 |
| 3 | 28.85 | 44.93 | 42.78 | 8.48 |
| 4 | 28.85 | 49.47 | 42.78 | 3.69 |
| 5 | 28.85 | 41.90 | 42.78 | 11.54 |
| 6 | 28.85 | 34.30 | 42.78 | 18.45 |
| 7 | 28.85 | 41.24 | 42.78 | 12.16 |
| 8 | 28.85 | 43.27 | 42.78 | 9.90 |
| 9 | 28.85 | 43.25 | 42.78 | 10.16 |
| 10 | 28.85 | 46.46 | 42.78 | 6.61 |
| 11 | 28.85 | 45.70 | 42.78 | 7.66 |
| 12 | 28.84 | 43.44 | 42.78 | 9.66 |
| 13 | 28.85 | 44.01 | 42.78 | 9.33 |
| 14 | 28.85 | 46.03 | 42.78 | 7.11 |
| 15 | 28.85 | 46.36 | 42.78 | 6.98 |

**卡间统计:**

| 指标 | 最小值 | 最大值 | 差异 |
|------|--------|--------|------|
| allocated | 28.84 GB | 28.85 GB | 一致 |
| reserved | 34.30 GB (dev6) | 49.47 GB (dev4) | **差 15.17 GB** |
| max_allocated | 42.78 GB | 42.79 GB | 一致 |
| free | 3.69 GB (dev4) | 18.45 GB (dev6) | **差 5.0 倍** |

### 4.3 PP mid 同节点 16 张卡 (10.234.9.85, Step 1)

| 指标 | 最小值 | 最大值 | 差异 |
|------|--------|--------|------|
| allocated | 33.82 GB | 33.83 GB | 一致 |
| reserved | 44.25 GB (dev0) | 52.30 GB (dev14) | 差 8.05 GB |
| max_allocated | 47.76 GB | 47.77 GB | 一致 |
| free | **999 MB (dev14)** | 9.12 GB (dev0) | **差 9.1 倍** |

**最危险**: dev14 free 仅 999 MB, 接近 OOM。

### 4.4 PP last 同节点 16 张卡 (10.237.63.63, Step 1)

| 指标 | 最小值 | 最大值 | 差异 |
|------|--------|--------|------|
| allocated | 33.13 GB | 33.14 GB | 一致 |
| reserved | 42.41 GB (dev2) | 53.40 GB (dev1) | **差 10.99 GB** |
| max_allocated | 46.96 GB | 46.96 GB | 完全一致 |
| free | **118 MB (dev1)** | 11.17 GB (dev15) | **差 94 倍** |

**最危险**: dev1 free 仅 118 MB! 距 OOM 仅 0.19% HBM。

---

## 五、静态 vs 动态显存拆解总结

### 5.1 PP first Stage 0 (8 节点 / 128 NPU)

| 类别            | 大小              | 计算方式                                         | 占 HBM |
| ------------- | --------------- | -------------------------------------------- | ----- |
| 静态显存 (权重+优化器) | 28.85 GB        | torch allocated                              | 47.1% |
| 动态峰值 (激活+梯度)  | ~13.9 GB        | max_allocated - allocated = 42.78 - 28.85    | 22.7% |
| PyTorch 缓存碎片  | ~7.2 GB         | max_reserved - max_allocated = 50.01 - 42.78 | 11.8% |
| 框架/驱动开销       | ~4.5~6.7 GB     | device occupy - torch reserved               | 7~11% |
| 剩余 free       | ~4.5 GB (step1) | —                                            | 7.3%  |

### 5.2 PP mid Stage 1-3 (24 节点 / 384 NPU)

| 类别 | 大小 | 计算方式 | 占 HBM |
|------|------|---------|--------|
| 静态显存 (权重+优化器) | 33.83 GB | torch allocated | 55.2% |
| 动态峰值 (激活+梯度) | ~13.9 GB | max_allocated - allocated = 47.76 - 33.83 | 22.7% |
| PyTorch 缓存碎片 | ~5.5 GB | max_reserved - max_allocated = 53.24 - 47.76 | 9.0% |
| 剩余 free | ~2.4~9 GB | — | 4~15% |

### 5.3 PP last Stage 4 (8 节点 / 128 NPU)

| 类别 | 大小 | 计算方式 | 占 HBM |
|------|------|---------|--------|
| 静态显存 (权重+优化器) | 33.13 GB | torch allocated | 54.1% |
| 动态峰值 (激活+梯度) | ~13.8 GB | max_allocated - allocated = 46.96 - 33.13 | 22.5% |
| PyTorch 缓存碎片 | ~6.4 GB | max_reserved - max_allocated = 53.36 - 46.96 | 10.4% |
| 剩余 free | ~3.2~11 GB | — | 5~18% |

---

## 六、与 PP=10 配置的对比

| 指标           | PP=5 (本日志)                      | PP=10 (上一日志)                  |
| ------------ | ------------------------------- | ----------------------------- |
| PP stages    | 5 (first:10, mid:13×3, last:12) | 10 (first:3, mid:7×8, last:2) |
| CP           | 32                              | 16                            |
| EP           | 128                             | 64                            |
| 每 stage 节点数  | 8                               | 4                             |
| allocated 范围 | 28.85~33.83 GB (**差 17%**)      | 12.03~34.51 GB (**差 187%**)   |
| **PP 均衡度**   | **较好**                          | **极不均衡**                      |
| 最紧张 stage    | 全部紧张 (free 都 < 12 GB)           | 仅 mid 紧张 (free < 1 GB)        |
| 最宽裕 stage    | 无 (所有 stage free < 18 GB)       | PP last (free ~35 GB)         |
| 最危险 free     | 118 MB (PP last dev1)           | 35 MB (PP mid)                |
| OOM 风险分布     | **分散在所有 stage**                 | 集中在 PP mid (32 节点)            |

---

## 七、核心结论与建议

### 7.1 核心结论

1. **静态显存完全稳定**: torch allocated 全程 4 步恒定不变。三类 stage: 28.85 GB (first, 10层) / 33.13 GB (last, 12层) / 33.83 GB (mid, 13层)。

2. **PP=5 的 stage 间均衡度远好于 PP=10**: allocated 最大差异仅 17% (vs PP=10 的 187%), 说明 PP=5 的层数分配 (10/13/13/13/12) 比 PP=10 (3/7×8/2) 更合理。

3. **动态峰值 (max_alloc - alloc) 大幅增加**: PP=5 下动态增量约 13.8~13.9 GB, 远大于 PP=10 的 6.8~7.0 GB, 且在所有 stage 高度一致（与层数无关: 10/12/13 层均 ~13.9 GB）。根因尚未完全确定。已排除"更多层导致更高峰值"（数据不支持）和"梯度缓冲延迟分配"（代码确认 `ParamAndGradBuffer` 始终在初始化时预分配）两个错误假设。CP/EP 配置差异（CP=32/EP=128 vs CP=16/EP=64）确认影响 bucket padding（PP10 静态中多 ~3 GB padding），但 ~7 GB 动态差值的完整来源需通过加日志实测确认（详见 PP5 vs PP10 分析文档）。

4. **所有 stage 都很紧张, 无安全余量**: PP=10 时 first/last 有 24~35 GB free, 但 PP=5 下所有 stage 的 free 都 < 18 GB, 最低到 118 MB。OOM 风险从 "集中在 mid" 变为 "分散在全部 stage"。

5. **PyTorch 缓存碎片仍然严重**: reserved 在步间大幅波动 (锯齿形), 同节点卡间 reserved 差异达 10~15 GB, 由 CP=32 的不同 rank 处理不等长序列导致。PP last dev1 (10.237.63.63) free 仅 118 MB, 源于 reserved=53.40 GB 碎片占用。

### 7.2 优化建议

1. **PP=5 比 PP=10 均衡度更好, 但整体更紧张**: 如果目标是减少 OOM 风险, PP=5 的均衡性更佳; 但如果目标是最大化吞吐, PP=10 的 first/last 有更多空间可以利用 (如增大 batch)。

2. **急需解决 PyTorch 缓存碎片**: 碎片 (reserved - allocated) 在所有 stage 都高达 5~21 GB, 占 HBM 的 8~34%。建议: (1) 每个 train step 后 `torch.cuda.empty_cache()`; (2) 设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`; (3) 或直接升级到支持更好内存管理的 PyTorch 版本。

3. **关注 CP rank 间数据均衡**: PP first 节点 dev4 (free 3.69 GB) vs dev6 (free 18.45 GB) 差 5 倍, PP last dev1 (free 118 MB) vs dev15 (free 11.17 GB) 差 94 倍。说明 CP=32 切分后序列长度极不均。建议优化 CP 切分策略。

4. **增大 DP 以分片优化器状态**: 当前 DP=1, Adam 优化器 m/v (FP32) 全量存储在每卡上。若增大 DP=2 并启用 optimizer sharding, 可将每卡静态显存降低约 30~40%, 大幅缓解 OOM 风险。
