# Kimi-K2.5 RL 训练显存分析报告

- **日志文件**: `log_aistudio-bsjvq9bu_2026-04-13_14-22-24.txt`
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
| 分配模式 | vllm[rollout]:d5t8p1e128 + megatron[actor]:d1t4p10c16e64 |

### 1.2 并行策略

| 并行维度                   | 值   | 说明                          |
| ---------------------- | --- | --------------------------- |
| TP (Tensor Parallel)   | 4   | 模型权重按 attention head 切分     |
| PP (Pipeline Parallel) | 10  | First:3层, Mid:7层×8, Last:2层 |
| EP (Expert Parallel)   | 64  | MoE expert 分布在 64 个 rank    |
| CP (Context Parallel)  | 16  | 128K 序列切分到 ~8K/rank         |
| DP (Data Parallel)     | 1   | 无数据并行, 优化器状态未分片             |

**PP 层数分配**: 61层总共分为 10 个 PP stage: Stage 0 (first) = 3层, Stage 1-8 (mid) = 7层/stage × 8 = 56层, Stage 9 (last) = 2层。每个 PP stage 占 4 个节点 (TP=4 × CP=16 = 64 ranks = 4 节点 × 16 NPU)。

### 1.3 模型参数分布

| PP Stage        | 层数        | 每 TP rank 参数量         | 说明                            |
| --------------- | --------- | --------------------- | ----------------------------- |
| Stage 0 (first) | 3 层       | 1.53B (1,529,796,080) | Embedding + 3 Transformer 层   |
| Stage 1-8 (mid) | 7 层/stage | 2.20B (2,202,583,040) | 7 MoE Transformer 层           |
| Stage 9 (last)  | 2 层       | 0.92B (922,917,888)   | 2 Transformer 层 + Output Head |

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

日志中发现三类不同 torch allocated 的节点, 分别对应 PP=10 中的 first / mid / last stage, 静态显存差异显著:

| PP Stage        | 代表节点          | 层数  | 节点数   | 参数量/TP | allocated | max_alloc | free(step4) |
| --------------- | ------------- | --- | ----- | ------ | --------- | --------- | ----------- |
| Stage 9 (last)  | 10.228.82.47  | 2 层 | 4 节点  | 0.92B  | 12.03 GB  | 18.84 GB  | 28.96 GB    |
| Stage 0 (first) | 10.237.142.68 | 3 层 | 4 节点  | 1.53B  | 15.71 GB  | 23.11 GB  | 24.64 GB    |
| Stage 1-8 (mid) | 10.239.50.27  | 7 层 | 32 节点 | 2.20B  | 34.51 GB  | 41.47 GB  | 0.48 GB     |

**关键发现**: PP mid (8 个 stage, 占 32 节点 / 80% 的 NPU) 的 allocated 是 PP last 的 2.87 倍。PP mid 节点已接近 OOM (free < 500 MB), 而 PP last 还剩 29 GB, PP first 还剩 25 GB。80% 的计算资源被最紧张的 mid stage 占据。

---

## 三、各 Step 显存增长趋势 (共 22 步, Step 0-21)

### 3.1 PP mid - Stage 1-8 (Node 10.239.50.27, device 0)

| Step   | allocated (GB) | reserved (GB) | max_alloc (GB) | max_resv (GB) | free (GB)  | occupy (GB) |
| ------ | -------------- | ------------- | -------------- | ------------- | ---------- | ----------- |
| 0      | 34.48          | 45.39         | 41.37          | 45.39         | 8.51       | 52.76       |
| 1      | 34.48          | 48.41         | 41.47          | 52.52         | 5.23       | 56.03       |
| 2      | 34.48          | 48.70         | 41.47          | 52.52         | 4.90       | 56.37       |
| 3      | 34.48          | 50.70         | 41.47          | 52.52         | 2.88       | 58.39       |
| **4**  | **34.51**      | **53.17 ★**   | **41.47**      | **53.17 ★**   | **0.48 ★** | **60.79 ★** |
| 5      | 34.52          | 48.71         | 41.47          | 53.17         | 4.93       | 56.34       |
| 6      | 34.52          | 47.86         | 41.47          | 53.17         | 5.77       | 55.50       |
| 7      | 34.51          | 48.74         | 41.47          | 53.17         | 4.89       | 56.38       |
| 8      | 34.51          | 52.27         | 41.47          | 53.17         | 1.36       | 59.91       |
| 9      | 34.52          | 47.01         | 41.47          | 53.17         | 6.63       | 54.64       |
| 10     | 34.52          | 45.70         | 41.47          | 53.17         | 7.96       | 53.31       |
| 11     | 34.51          | 48.72         | 41.47          | 53.17         | 4.93       | 56.34       |
| 12     | 34.51          | 46.15         | 41.47          | 53.17         | 7.49       | 53.78       |
| 13     | 34.51          | 50.99         | 41.47          | 53.17         | 2.64       | 58.63       |
| **14** | **34.52**      | **53.18 ★**   | **41.47**      | **53.18**     | **0.47 ★** | **60.80 ★** |
| 15     | 34.52          | 50.17         | 41.47          | 53.18         | 3.45       | 57.82       |
| 16     | 34.51          | 46.66         | 41.47          | 53.18         | 7.00       | 54.27       |
| 17     | 34.52          | 46.83         | 41.47          | 53.18         | 6.83       | 54.44       |
| 18     | 34.52          | 50.43         | 41.47          | 53.18         | 3.21       | 58.06       |
| 19     | 34.52          | 51.76         | 41.47          | 53.18         | 1.90       | 59.37       |
| 20     | 34.52          | 47.86         | 41.47          | 53.18         | 5.75       | 55.52       |
| 21     | 34.51          | 46.79         | 41.51          | 53.18         | 6.82       | 54.45       |

### 3.2 PP first - Stage 0 (Node 10.237.142.68, device 0)

| Step | allocated (GB) | reserved (GB) | max_alloc (GB) | max_resv (GB) | free (GB) | occupy (GB) |
|------|----------------|---------------|----------------|---------------|-----------|-------------|
| 0 | 15.70 | 23.31 | 17.67 | 23.31 | 30.54 | 30.73 |
| 1 | 15.70 | 29.80 | 22.70 | 29.80 | 23.67 | 37.59 |
| 2 | 15.70 | 31.93 | 23.11 | 31.93 | 21.51 | 39.76 |
| 3 | 15.70 | 27.98 | 23.11 | 31.93 | 25.48 | 35.79 |
| 4 | 15.71 | 28.67 | 23.11 | 31.93 | 24.64 | 36.63 |
| 5 | 15.72 | 32.66 | 23.11 | 32.66 | 20.73 | 40.54 |
| 8 | 15.72 | 30.37 | 23.11 | 32.66 | 23.06 | 38.21 |
| 11 | 15.72 | 33.59 | 23.11 | 33.59 | 19.84 | 41.43 |
| 14 | 15.72 | 33.78 | 23.11 | 33.78 | 19.63 | 41.64 |
| **17** | **15.72** | **35.31 ★** | **23.11** | **35.31 ★** | **18.06** | **43.21 ★** |
| 21 | 15.73 | 30.33 | 23.11 | 35.31 | 23.15 | 38.12 |

### 3.3 趋势总结

| 指标                         | 变化规律                                                             | 结论                   |
| -------------------------- | ---------------------------------------------------------------- | -------------------- |
| torch allocated (静态显存)     | 全程几乎恒定: PP mid 34.48→34.52 (+0.1%), PP first 15.70→15.73 (+0.2%) | 模型权重+优化器状态完全固定       |
| torch max_allocated (动态峰值) | Step 0→1 跳升后稳定: PP mid 41.37→41.47 GB, PP first 17.67→23.11 GB   | 激活+梯度峰值是确定性的         |
| torch reserved (缓存池)       | 大幅波动, 呈锯齿形, 逐步爬高: PP mid 45→53 GB, PP first 23→35 GB             | PyTorch 缓存碎片化是显存增长主因 |
| max_reserved (历史最高缓存)      | 单调递增, 只升不降: PP mid +17%, PP first +51%                           | 碎片无法释放, 持续累积         |
| device mem_free            | PP mid 最低 0.47 GB (step 14), PP first 最低 18 GB                   | PP mid 多次接近 OOM 边界   |

---

## 四、特定 Rank (节点/Device) 显存对比

### 4.1 全部 40 节点 Step 1 Device 0 对比

按 allocated 分组, 共发现三类节点, 与 PP stage 一一对应:

| PP Stage | 节点数 | allocated | occupy 范围 | free 范围 | 代表节点 |
|----------|--------|-----------|-------------|-----------|----------|
| Stage 9 (last, 2层) | 4 节点 | 12.02 GB | 27~32 GB | 29~34 GB | 10.228.82.47, 10.232.163.79, 10.234.161.60, 10.236.10.71 |
| Stage 0 (first, 3层) | 4 节点 | 15.70 GB | 29~46 GB | 15~27 GB | 10.227.100.73, 10.232.150.73, 10.236.60.56, 10.237.142.68 |
| Stage 1-8 (mid, 7层) | 32 节点 | 34.47~34.48 GB | 52~61 GB | 0.04~8.5 GB | (32 个节点, 占 80% 资源) |

**最危险节点 (Step 1)**: 10.231.212.49 — device occupy = 61.24 GB, mem_free = 35.3 MB, 距离 OOM 仅剩 0.06% HBM。10.233.89.56 — mem_free = 827 MB。

### 4.2 同一节点 16 张卡分布 (PP mid Stage 1-8, Step 4)

Node 10.239.50.27 (PP stage 中间, 7 MoE 层):

| Device | allocated (GB) | reserved (GB) | max_alloc (GB) | free (GB) |
|--------|----------------|---------------|----------------|-----------|
| 0 | 34.51 | 53.17 | 41.47 | 0.49 ★ |
| 1 | 34.50 | 47.88 | 41.47 | 5.93 |
| 2 | 34.51 | 45.40 | 41.47 | 8.24 |
| 3 | 34.51 | 47.40 | 41.47 | 6.55 |
| 4 | 34.51 | 46.98 | 41.47 | 6.62 |
| 5 | 34.51 | 47.89 | 41.47 | 5.94 |
| 6 | 34.51 | 49.84 | 41.47 | 3.77 |
| 7 | 34.51 | 52.55 | 45.34 ★ | 1.34 |
| 8 | 34.51 | 46.84 | 41.47 | 6.87 |
| 9 | 34.50 | 46.30 | 41.47 | 7.43 |
| 10 | 34.51 | 50.09 | 41.47 | 3.50 |
| 11 | 34.51 | 49.63 | 41.47 | 4.28 |
| 12 | 34.51 | 52.99 | 41.47 | 0.69 ★ |
| 13 | 34.51 | 47.81 | 41.47 | 6.11 |
| 14 | 34.51 | 48.40 | 41.47 | 5.18 |
| 15 | 34.51 | 50.32 | 41.47 | 3.57 |

**卡间差异统计:**

| 指标 | 最小值 | 最大值 | 差异 |
|------|--------|--------|------|
| allocated | 34.50 GB | 34.51 GB | 几乎一致 |
| reserved | 45.40 GB | 53.17 GB | 差 7.77 GB |
| max_allocated | 41.47 GB | 45.34 GB (dev7) | dev7 异常高 |
| free | 0.49 GB (dev0) | 8.24 GB (dev2) | 差 16.8 倍 |

**分析**: 同一节点 16 张卡的 allocated 完全一致, 说明静态显存由模型结构决定。但 reserved 差异高达 7.77 GB, 源于不同 CP rank (CP=16) 处理不同长度的数据切片, 导致 PyTorch 缓存碎片化程度不同。Device 7 的 max_allocated 异常高 (45.34 GB vs 41.47 GB), 可能是该 CP rank 分到了较长的序列片段。

### 4.3 同一节点 16 张卡分布 (PP first Stage 0, Step 4)

Node 10.237.142.68 (PP stage 0, Embedding + 3 层):

| 指标 | 最小值 | 最大值 | 差异 |
|------|--------|--------|------|
| allocated | 15.71 GB | 15.72 GB | 一致 |
| reserved | 25.50 GB (dev6) | 40.22 GB (dev12) | 差 14.72 GB |
| max_allocated | 22.69 GB | 25.55 GB (dev4) | 部分卡动态峰值更高 |
| free | 13.25 GB (dev12) | 27.99 GB (dev6) | 差 2.1 倍 |

PP first 节点的 reserved 差异更大 (14.72 GB), 但由于 allocated 基数较低 (15.7 GB), 即使最紧张的卡也有 13 GB free, 不存在 OOM 风险。

### 4.4 PP last Stage 9 卡间分布 (Step 4)

Node 10.228.82.47 (PP stage 9, 2 层 + Output Head):

| 指标 | 最小值 | 最大值 | 差异 |
|------|--------|--------|------|
| allocated | 12.02 GB | 12.03 GB | 一致 |
| reserved | 17.87 GB (dev15) | 24.83 GB (dev0) | 差 6.96 GB |
| max_allocated | 18.84 GB | 18.84 GB | 完全一致 |
| free | 28.96 GB (dev0) | 36.06 GB (dev15) | 最充裕 |

PP last 节点显存最为充裕, 所有卡 free 均 > 28 GB, 存在大量浪费空间。

---

## 五、Rollout 侧 (vLLM/SGLang) 显存

| 指标 | 值 |
|------|-----|
| Available KV cache memory | ~14.9 GiB (各节点 14.91~14.99 GiB) |
| CUDA Graph capturing 占用 | 0.36 GiB |
| mem_fraction_static | 0.65 |
| 推理引擎 | vLLM (rollout 配置: d5t8p1e128) |

---

## 六、静态 vs 动态显存拆解总结

### 6.1 PP mid Stage 1-8 (主要瓶颈, 32 节点 / 512 NPU)

| 类别            | 大小       | 计算方式                                         | 占 HBM |
| ------------- | -------- | -------------------------------------------- | ----- |
| 静态显存 (权重+优化器) | 34.5 GB  | torch allocated (训练后稳态)                      | 56.3% |
| 动态峰值 (激活+梯度)  | ~7.0 GB  | max_allocated - allocated = 41.47 - 34.48    | 11.4% |
| PyTorch 缓存碎片  | ~18.7 GB | max_reserved - max_allocated = 53.18 - 34.48 | 30.5% |
| 框架/驱动开销       | ~1.1 GB  | device occupy - torch reserved               | 1.8%  |

### 6.2 PP first Stage 0 (4 节点 / 64 NPU)

| 类别 | 大小 | 占 HBM |
|------|------|--------|
| 静态显存 | 15.7 GB | 25.6% |
| 动态峰值 | ~7.4 GB | 12.1% |
| PyTorch 缓存碎片 | ~12.2 GB | 19.9% |
| 剩余 free | ~25.9 GB | 42.3% |

### 6.3 PP last Stage 9 (4 节点 / 64 NPU)

| 类别 | 大小 | 占 HBM |
|------|------|--------|
| 静态显存 | 12.0 GB | 19.6% |
| 动态峰值 | ~6.8 GB | 11.1% |
| PyTorch 缓存碎片 | ~7.1 GB | 11.6% |
| 剩余 free | ~35.3 GB | 57.6% |

### 6.4 显存利用率全景

以总集群 640 NPU × 61.27 GB ≈ 39.2 TB HBM 计算:

| PP Stage | NPU 数 | 每卡 occupy (step4) | 总占用 | 每卡 free | 总浪费 |
|----------|--------|---------------------|--------|-----------|--------|
| Stage 0 (first) | 64 | ~37 GB | 2.3 TB | ~24 GB | 1.5 TB |
| Stage 1-8 (mid) | 512 | ~55 GB | 28.2 TB | ~6 GB | 3.1 TB |
| Stage 9 (last) | 64 | ~27 GB | 1.7 TB | ~34 GB | 2.2 TB |
| **合计** | **640** | **—** | **32.2 TB** | **—** | **6.8 TB** |

**集群显存利用率**: 32.2 / 39.2 TB = 82.1%。其中 PP first + last (8 节点) 浪费 3.7 TB, 占总浪费的 54%。

---

## 七、核心结论与建议

### 7.1 核心结论

1. **静态显存完全稳定**: torch allocated 全程 22 个 step 几乎不变 (±0.1%), 证明模型权重 + 优化器状态是完全固定的。三类 PP stage 静态显存分别为: 12 GB (last, 2层) / 15.7 GB (first, 3层) / 34.5 GB (mid, 7层)。

2. **动态峰值在 Step 1 后收敛**: max_allocated 在 Step 0→1 有一次跳升 (首次完整训练建立梯度缓冲), 之后几乎不变, 说明激活+梯度的峰值是确定性的。动态增量 (max_alloc - alloc) 约 6.8~7.4 GB, 各 PP stage 差异不大。

3. **显存危险来自 PyTorch 缓存池碎片化**: reserved 持续爬升 (max_reserved 只增不降): PP mid 从 45→53 GB (+18%), PP first 从 23→35 GB (+51%)。同节点 16 张卡的 reserved 差异高达 14.7 GB, 这是由 CP=16 中不同 rank 处理不等长序列切片导致的碎片化。

4. **PP=10 的 stage 不均衡是最大问题**: PP mid (7层, 8 个 stage, 32 节点) 占用 34.5 GB, 而 PP last (2层, 4 节点) 仅 12 GB, 差 2.87 倍。PP mid 占 80% 的 NPU, 多次出现 free < 500 MB 的极端情况; 而 PP first/last 共 8 节点的 128 张 NPU, 每卡浪费 24~35 GB 空间。

5. **部分节点已在 OOM 边缘**: Step 1 时 10.231.212.49 节点 free 仅 35.3 MB; Step 4/14 时 10.239.50.27 节点 device 0 free < 500 MB。随 step 增加碎片累积, max_reserved 只增不降, OOM 风险持续升高。

### 7.2 优化建议

1. **重新平衡 PP 层数切分**: 当前 PP=10, first=3层, mid=7层×8, last=2层。Mid stage 的 7 个 MoE 层占 34.5 GB, 而 last 的 2 层仅 12 GB。建议调整为更均匀的分配 (如减少 mid 层数, 增加 first/last 层数), 或尝试减少 PP stage 数 (如 PP=5 配合更大的 EP) 来减少不均衡。

2. **增大 DP 以分片优化器状态**: 当前 DP=1, data_parallel_sharding_strategy=no_shard, 优化器状态 (Adam m/v, FP32) 未分片, 占据大量静态显存。增大 DP=2 并启用分片可将每卡优化器显存减半。

3. **优化 PyTorch 缓存碎片**: 日志已配置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, 但碎片仍然严重 (reserved 比 allocated 多出 ~18 GB)。建议: (1) 设置 max_split_size_mb 限制分块大小; (2) 在每个 train step 结束后调用 torch.cuda.empty_cache(); (3) 监控 max_reserved 增长, 超阈值时触发清理。

4. **关注 CP rank 间的数据长度均衡**: 同节点 16 张卡 (对应 CP=16 的不同 rank) reserved 差异高达 14.7 GB, Device 7 的 max_allocated 异常高 (45.34 GB vs 41.47 GB)。说明 CP 切分后各 rank 的序列长度不均。建议检查 CP 切分策略, 确保序列在各 CP rank 间更均匀分配。

---

## 八、Sleep 完成后稳态显存分析

**分析日期**: 2026-04-28
**数据来源**: 同一日志文件中 "after offloaded weights" 检查点的显存日志
**参考数据**: 16B 模型 profiling (`KimiK25_Step0_vLLM_Sleep_显存分析.md`, `KimiK25_NPU_Profiling_动态显存分析.md`)

### 8.1 Colocate 模式下的 Sleep 流程

训练和推理共享同一组 NPU (Colocate 模式)。每次训练→推理切换时, Megatron 需要释放显存给 vLLM:

```
_write_weights_in_colocate_mode()  (weights_writer.py:691)
  ├─ group_tensors_by_shape_and_dtype(tensors)     # 创建 grouped 副本用于 IPC
  ├─ release_memory_occupation("weights")           # 触发 offload
  │   └─ offload_optimizer_states()                 # Adam exp_avg/exp_avg_sq → CPU
  │       ├─ offload_megatron_optimizer()
  │       ├─ gc.collect()
  │       └─ torch.npu.empty_cache()                # ← 第 1 次 empty_cache
  │   └─ offload_params()                           # 模型权重 → CPU
  │       ├─ offload_megatron_model_to_cpu()        # param_data/grad_data.storage().resize_(0)
  │       ├─ gc.collect()
  │       └─ torch.npu.empty_cache()                # ← 第 2 次 empty_cache
  ├─ print_current_gpu_status("after offloaded weights")  # ★ 测量点
  ├─ group_shared = [tensor.cuda().share_memory_()]  # IPC shared memory
  ├─ ... (权重交换) ...
  └─ torch.cuda.empty_cache()                       # ← 第 3 次 empty_cache
```

### 8.2 首次 Offload vs 稳态 Offload

| 指标 | 首次 Offload (训练前, 14:51) | 稳态 Offload (训练后, 17:07+) |
|------|---------------------------|------------------------------|
| torch allocated | 142~490 KB | **431 MB ~ 1.13 GB** |
| torch reserved | ~0 | 3.5~5.0 GB |
| device occupy | 4.52~4.87 GB | **8.9~12.9 GB** |
| 状态 | 干净, 无训练残留 | 训练产生的 tensor 未完全 offload |

**关键发现**: 首次 offload 非常干净 (allocated 仅数百 KB), 但训练过一轮后的稳态 offload, allocated 涨到 MB~GB 级别, 说明有 tensor 在训练过程中产生但 offload 流程没有覆盖到。

### 8.3 各 PP Stage 稳态 Offload 显存

#### 8.3.1 PP mid (Stage 1-8, 32 节点 / 512 NPU) — 最紧张

| 指标 | 值 | 说明 |
|------|-----|------|
| torch allocated | **1.13 GB** | 未 offload 的 tensor |
| torch reserved | 3.5~5.0 GB | 因 allocated 中的 live tensor 导致 segment 无法释放 |
| device occupy (非 torch) | ~7.6~7.8 GB | HCCL buffer + Driver/Runtime |
| **device occupy 总计** | **11.0~12.9 GB** | Megatron 训练的显存基线 |

#### 8.3.2 PP first (Stage 0, 4 节点 / 64 NPU)

| 指标 | 值 |
|------|-----|
| torch allocated | ~431 MB |
| device occupy | 9.3~9.9 GB |

#### 8.3.3 PP last (Stage 9, 4 节点 / 64 NPU)

| 指标 | 值 |
|------|-----|
| torch allocated | ~454 MB |
| device occupy | 8.9~9.8 GB |

### 8.4 稳态显存构成分析

以 PP mid 为例, 稳态 offload 后 device 总占用 ~12 GB 的拆解:

| 组件 | 大小 | 可优化 | 说明 |
|------|------|--------|------|
| **未 offload 的 tensor** | 1.13 GB | ✓ 需排查定位 | 训练中产生, offload 流程未覆盖 |
| **PyTorch reserved 碎片** | 2.4~3.9 GB | ✓ 清除 live tensor 后可释放 | 因 live tensor 钉住 segment |
| **HCCL 通信 buffer** | ~5.6 GB (估) | ✓ 可销毁通信组 | Megatron 更多通信组 (TP+PP+CP+EP) |
| **Driver/Runtime 开销** | ~2.0 GB | ✗ 硬件栈固有 | CANN 上下文, 页表等 |

### 8.5 "reserved 碎片" 的根因

empty_cache() 已调用 3 次, 但 reserved 仍高达 3.5~5.0 GB, 原因:

1. PyTorch 配置了 `expandable_segments:True`, 分配器按 segment (通常 2/4/8/16 MB) 为单位管理显存
2. 只要 segment 中有 **任何一个 live tensor**, 整个 segment 都无法释放
3. 1.13 GB 的 live tensor 散布在多个 segment 中, 导致大量 segment 被钉住
4. 只有清除这 1.13 GB 的 live tensor, empty_cache() 才能真正回收 reserved

### 8.6 集群级影响

| 优化项 | 单卡节省 | PP mid 512 卡节省 | 全集群 640 卡节省 |
|--------|---------|-------------------|------------------|
| 清除 1.13 GB 未 offload tensor | 1.13 GB | **578 GB** | — |
| 清除后 reserved 碎片可释放 | 2.4~3.9 GB | **1.2~2.0 TB** | — |
| 合计 (tensor + 碎片) | 3.5~5.0 GB | **1.8~2.6 TB** | — |
| HCCL buffer 回收 (若可行) | ~5.6 GB | 2.9 TB | ~3.6 TB |

对 PP mid (free 低至 0.47 GB) 而言, 单卡节省 3.5~5.0 GB 可以直接消除 OOM 风险。

### 8.7 诊断方案

已在 `megatron_backend.py` 的 `offload_params()` 中添加 `_dump_gpu_residual_tensors()` 诊断方法:

- **触发条件**: offload 完成后 `torch.cuda.memory_allocated() > 1 MB` 时, 在采样 rank (每 64 个取一个) 上自动调用
- **首次 offload 不触发** (allocated 仅数百 KB), **稳态 offload 触发** (allocated > 1 MB)

扫描 4 层来源:
1. **ParamAndGradBuffer** — 检查 `param_data` / `grad_data` 的 storage 是否 resize 到 0
2. **model named_parameters** — 遍历模型参数, 找仍在 GPU 上的
3. **model named_buffers** — 检查 LayerNorm running stats, position embedding 等
4. **optimizer states** — 检查 Adam 的 exp_avg / exp_avg_sq 是否搬到了 CPU

若以上仍解释不了 allocated 总量 (差 > 10 MB), 兜底使用 `gc.get_objects()` 扫描所有 GPU tensor。

输出格式:
```
[RESIDUAL] rank=X accounted=1100.0 MB / total_allocated=1130.0 MB (unaccounted=30.0 MB)
[RESIDUAL #0] 800.0 MB | model[0].expert_buffer[0].grad_data | dtype=float32 storage=...
[RESIDUAL #1] 300.0 MB | optim[0].state.exp_avg | dtype=float32 shape=[...]
```

### 8.8 优化路径 (分层)

| 优先级 | 优化项 | 预期单卡收益 | 集群收益 | 难度 |
|--------|--------|-------------|---------|------|
| **Tier 1** | 定位并 offload 1.13 GB 残留 tensor | ~1.13 GB + 2.4~3.9 GB 碎片释放 | ~1.8~2.6 TB (PP mid) | 中: 需等诊断日志 |
| **Tier 2** | 降低 HCCL_BUFFSIZE (600→200) | ~1.2 GB | ~768 GB | 低: 需评估通信性能 |
| **Tier 3** | offload 后销毁 Megatron 通信组 | ~5.6 GB | ~3.6 TB | 高: wake_up 需重建 |
| **已有** | `allocator_and_pools.clear()` (vLLM 侧) | 231 MB (16B 实测) | ~144~433 GB | 已验证 |

### 8.9 与 16B Profiling 数据的对比

| 指标                             | 16B 模型 (profiling)  | 1T 模型 (日志)        | 说明                            |
| ------------------------------ | ------------------- | ----------------- | ----------------------------- |
| vLLM sleep 后 device            | 4.88 GB             | —                 | 16B: TP=2, PP=4, world_size=8 |
| HCCL buffer (vLLM)             | 2.08 GB (3×600+281) | —                 | 16B 实测                        |
| Module 33 残留                   | 192 MB (不可优化)       | —                 | pluggable allocator 基础设施      |
| Megatron offload 后 torch alloc | —                   | 1.13 GB (PP mid)  | 1T: 训练残留                      |
| Megatron offload 后 device      | —                   | 11~13 GB (PP mid) | 含 HCCL + Driver               |
| 非 torch 开销                     | ~2.1 GB (Driver)    | ~7.6~7.8 GB       | 1T 规模更大, HCCL 组更多             |
