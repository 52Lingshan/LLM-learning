# Bailing V3 Megatron-Bridge 适配工作摘要

## 模型架构概览

Bailing V3 (`BailingMoeV3ForCausalLM`) 是一个混合 MoE 模型，交替使用 MLA（softmax 注意力）和 KDA（Kimi Delta Attention / 线性注意力）层。

| 参数 | 值 |
|------|-----|
| 层数 | 42 + 1 MTP |
| hidden_size | 2560 |
| num_attention_heads | 32 |
| num_key_value_heads | 32 |
| vocab_size | 157184 |
| num_experts | 512 (routed) + 1 (shared) |
| num_experts_per_tok | 8 |
| first_k_dense_replace | 2 (前2层为dense MLP) |
| layer_group_size | 6 (每6层1层MLA，5层KDA) |
| kv_lora_rank | 512 |
| q_lora_rank | 768 (MLA层有Q LoRA压缩) |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |
| rope_theta | 6000000 |
| rope_interleave | true |
| moe_intermediate_size | 768 |
| scoring_func | sigmoid |
| routed_scaling_factor | 2.5 |
| short_conv_kernel_size | 4 |
| num_nextn_predict_layers | 1 |

### 层类型分布

| 层索引 | 注意力类型 | MLP类型 |
|--------|-----------|---------|
| 0-1 | KDA | Dense |
| 2-4 | KDA | MoE |
| 5 | MLA | MoE |
| 6-10 | KDA | MoE |
| 11 | MLA | MoE |
| ... | 每6层循环 | MoE |
| 41 | MLA | MoE |
| 42 (MTP) | MLA | MoE |

### HF 权重命名

**MLA 层 (每6层1层)**:
- `model.layers.*.attention.q_a_proj.weight` — Q LoRA降维投影（q_lora_rank=768时）
- `model.layers.*.attention.q_a_layernorm.weight` — Q LoRA LayerNorm
- `model.layers.*.attention.q_b_proj.weight` — Q LoRA升维投影
- `model.layers.*.attention.kv_a_proj_with_mqa.weight` — KV下投影
- `model.layers.*.attention.kv_b_proj.weight` — KV上投影
- `model.layers.*.attention.kv_a_layernorm.weight` — KV LayerNorm
- `model.layers.*.attention.dense.weight` — 输出投影
- `model.layers.*.attention.g_proj.weight` — 门控投影

**KDA 层 (5/6层)**:
- `model.layers.*.attention.q_proj.weight` — Q投影
- `model.layers.*.attention.k_proj.weight` — K投影
- `model.layers.*.attention.v_proj.weight` — V投影
- `model.layers.*.attention.f_proj.weight` — 门控投影（传给chunk_kda的g参数）
- `model.layers.*.attention.g_proj.weight` — 输出门控（FusedRMSNormGated）
- `model.layers.*.attention.b_proj.weight` — beta投影
- `model.layers.*.attention.A_log` — 衰减率对数参数
- `model.layers.*.attention.dt_bias` — 时间步偏置参数
- `model.layers.*.attention.o_proj.weight` — 输出投影
- `model.layers.*.attention.o_norm.weight` — 输出归一化
- `model.layers.*.attention.q_conv1d.weight` — Q短卷积
- `model.layers.*.attention.k_conv1d.weight` — K短卷积
- `model.layers.*.attention.v_conv1d.weight` — V短卷积

**MoE 层**:
- `model.layers.*.mlp.gate.weight` — router权重
- `model.layers.*.mlp.gate.expert_bias` — router专家偏置
- `model.layers.*.mlp.experts.*.gate_proj/up_proj/down_proj.weight` — 专家MLP
- `model.layers.*.mlp.shared_experts.gate_proj/up_proj/down_proj.weight` — 共享专家

**MTP 层 (42)**:
- `model.layers.42.eh_proj.weight` — MTP嵌入投影
- `model.layers.42.enorm.weight` — MTP嵌入归一化
- `model.layers.42.hnorm.weight` — MTP隐藏归一化
- `model.layers.42.final_layernorm.weight` — MTP最终归一化

---

## 修改文件清单

### Megatron-LM 修改

| 文件                                                                        | 修改内容                                                                                                                                                                                    |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `megatron/core/transformer/transformer_config.py`                         | 1. 放宽 `rotary_interleaved + MLA` 限制：允许 `rope_type="rope"` 时使用<br>2. 添加 `'kda'` 作为 `experimental_attention_variant` 选项<br>3. 将 GDN 验证逻辑扩展为同时覆盖 `kda`                                     |
| `megatron/core/transformer/multi_latent_attention.py`                     | 传递 `rotary_interleaved` 参数给 MLA 的 `RotaryEmbedding` 构造函数                                                                                                                                |
| `megatron/core/ssm/kda.py`                                                | **新增**：`KimiDeltaAttention` 模块，实现 Bailing V3 的 KDA 层                                                                                                                                    |
| `megatron/core/models/gpt/experimental_attention_variant_module_specs.py` | 1. 添加 `KimiDeltaAttention` 导入<br>2. 添加 `get_kda_module_spec()` 函数<br>3. 更新 `get_experimental_attention_variant_module_spec()` 分派 KDA<br>4. 将 `'kda'` 加入 `is_linear_attention_variant()` |

### Megatron-Bridge 新增

| 文件 | 内容 |
|------|------|
| `src/megatron/bridge/models/bailing/__init__.py` | 导出 `BailingMoeV3Bridge`, `BailingMoeV3Provider` |
| `src/megatron/bridge/models/bailing/bailing_provider.py` | `BailingMoeV3Provider` 数据类，含完整模型配置默认值 |
| `src/megatron/bridge/models/bailing/bailing_bridge.py` | `BailingMoeV3Bridge`，注册 `BailingMoeV3ForCausalLM`，处理 MLA/KDA/MoE/MTP 权重映射 |
| `src/megatron/bridge/recipes/bailing/__init__.py` | 导出 `bailing_v3_pretrain_config` |
| `src/megatron/bridge/recipes/bailing/bailing_v3.py` | `bailing_v3_pretrain_config()` 配方函数（已补全所有训练参数，对齐 kimi_k25 配方） |
| `examples/decentralized_pg/pretrain_bailing_v3.py` | 预训练脚本（参考 pretrain_kimi25.py） |

### Megatron-Bridge 修改

| 文件 | 修改内容 |
|------|---------|
| `src/megatron/bridge/models/__init__.py` | 添加 Bailing 导入和 `__all__` 导出 |
| `src/megatron/bridge/recipes/__init__.py` | 添加 `from megatron.bridge.recipes.bailing import *` |

---

## MCore KDA 模块设计

### KDA vs GDN 对比

| 特性 | GDN (Qwen3-Next) | KDA (Bailing V3) |
|------|------------------|-------------------|
| 输入投影 | 合并 `in_proj` (QKVZ+BA) | 分离 `q_proj/k_proj/v_proj/f_proj/g_proj/b_proj` |
| 卷积 | 合并 `conv1d` (QKV) | 分离 `q_conv1d/k_conv1d/v_conv1d` |
| 衰减计算 | `g = -A_log.exp() * softplus(alpha + dt_bias)` | 直接传 `A_log` 和 `dt_bias` 给 `chunk_kda` |
| alpha | 来自 `in_proj` | 无 alpha 投影 |
| 输出门控 | `RMSNorm(x) * silu(gate)` (gate来自in_proj) | `FusedRMSNormGated(o, g_proj_output)` |
| 核心kernel | `chunk_gated_delta_rule` | `chunk_kda` |

### KimiDeltaAttention 模块结构

```
KimiDeltaAttention
├── q_proj (ColumnParallelLinear)  → qk_dim
├── k_proj (ColumnParallelLinear)  → qk_dim
├── v_proj (ColumnParallelLinear)  → v_dim
├── f_proj (ColumnParallelLinear)  → v_dim (gate for chunk_kda)
├── g_proj (ColumnParallelLinear)  → v_dim (output gate)
├── b_proj (ColumnParallelLinear)  → num_value_heads (beta)
├── q_conv1d (nn.Conv1d)
├── k_conv1d (nn.Conv1d)
├── v_conv1d (nn.Conv1d)
├── A_log (Parameter)
├── dt_bias (Parameter)
├── out_norm (LayerNorm/RMSNorm)
└── out_proj (RowParallelLinear)
```

---

## 权重映射表

### Megatron → HF 映射

| Megatron 参数 | HF 参数 | 层类型 |
|--------------|---------|--------|
| `embedding.word_embeddings.weight` | `model.word_embeddings.weight` | 全局 |
| `output_layer.weight` | `lm_head.weight` | 全局 |
| `decoder.final_layernorm.weight` | `model.norm.weight` | 全局 |
| `decoder.layers.*.input_layernorm.weight` | `model.layers.*.input_layernorm.weight` | 通用 |
| `decoder.layers.*.self_attention.linear_q_proj.weight` | `model.layers.*.attention.q_proj.weight` | MLA (q_lora_rank=None时) |
| `decoder.layers.*.self_attention.linear_q_down_proj.weight` | `model.layers.*.attention.q_a_proj.weight` | MLA (q_lora_rank≠None时) |
| `decoder.layers.*.self_attention.linear_q_up_proj.weight` | `model.layers.*.attention.q_b_proj.weight` | MLA (q_lora_rank≠None时) |
| `decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight` | `model.layers.*.attention.q_a_layernorm.weight` | MLA (q_lora_rank≠None时) |
| `decoder.layers.*.self_attention.linear_kv_down_proj.weight` | `model.layers.*.attention.kv_a_proj_with_mqa.weight` | MLA |
| `decoder.layers.*.self_attention.linear_kv_up_proj.weight` | `model.layers.*.attention.kv_b_proj.weight` | MLA |
| `decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight` | `model.layers.*.attention.kv_a_layernorm.weight` | MLA |
| `decoder.layers.*.self_attention.linear_proj.weight` | `model.layers.*.attention.dense.weight` | MLA |
| `decoder.layers.*.self_attention.linear_qkv.weight` | QKV合并映射 | KDA |
| `decoder.layers.*.self_attention.f_proj.weight` | `model.layers.*.attention.f_proj.weight` | KDA |
| `decoder.layers.*.self_attention.g_proj.weight` | `model.layers.*.attention.g_proj.weight` | KDA |
| `decoder.layers.*.self_attention.b_proj.weight` | `model.layers.*.attention.b_proj.weight` | KDA |
| `decoder.layers.*.self_attention.A_log` | `model.layers.*.attention.A_log` | KDA |
| `decoder.layers.*.self_attention.dt_bias` | `model.layers.*.attention.dt_bias` | KDA |
| `decoder.layers.*.self_attention.q_conv1d.weight` | `model.layers.*.attention.q_conv1d.weight` | KDA |
| `decoder.layers.*.self_attention.k_conv1d.weight` | `model.layers.*.attention.k_conv1d.weight` | KDA |
| `decoder.layers.*.self_attention.v_conv1d.weight` | `model.layers.*.attention.v_conv1d.weight` | KDA |
| `decoder.layers.*.self_attention.out_proj.weight` | `model.layers.*.attention.o_proj.weight` | KDA |
| `decoder.layers.*.self_attention.out_norm.weight` | `model.layers.*.attention.o_norm.weight` | KDA |
| `decoder.layers.*.mlp.router.weight` | `model.layers.*.mlp.gate.weight` | MoE |
| `decoder.layers.*.mlp.router.expert_bias` | `model.layers.*.mlp.gate.expert_bias` | MoE |
| `decoder.layers.*.pre_mlp_layernorm.weight` | `model.layers.*.post_attention_layernorm.weight` | MoE |
| `decoder.layers.*.mlp.experts.linear_fc1.weight*` | `model.layers.*.mlp.experts.*.gate_proj+up_proj` (GatedMLP) | MoE (TEGroupedMLP) |
| `decoder.layers.*.mlp.experts.linear_fc2.weight*` | `model.layers.*.mlp.experts.*.down_proj.weight` | MoE (TEGroupedMLP) |
| `decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight` | `model.layers.*.mlp.experts.*.gate_proj+up_proj` (GatedMLP) | MoE (SequentialMLP) |
| `decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight` | `model.layers.*.mlp.experts.*.down_proj.weight` | MoE (SequentialMLP) |
| `mtp.layers.0.eh_proj.weight` | `model.layers.42.eh_proj.weight` | MTP |
| `mtp.layers.0.enorm.weight` | `model.layers.42.enorm.weight` | MTP |
| `mtp.layers.0.hnorm.weight` | `model.layers.42.hnorm.weight` | MTP |
| `mtp.layers.0.final_layernorm.weight` | `model.layers.42.final_layernorm.weight` | MTP |

---

## Pretrain 调试记录（2025-05-09）

### 问题 1：Expert 权重未加载导致 NaN

**现象**：运行 pretrain 脚本，iteration 1 报 NaN loss。Dump 数据显示 `experts.local_experts.0.linear_fc1` 输出全为 NaN（输入正常，输出含 e+37 极大值），router 输出全零。

**根因**：Recipe (`bailing_v3.py`) 设置 `moe_grouped_gemm = False`，模型使用 `SequentialMLP`（参数命名 `local_experts.E.linear_fc1.weight`），但 bridge 只定义了 TEGroupedMLP 格式的映射（`linear_fc1.weight*`）。所有 expert fc1/fc2 权重在 HF→Megatron 加载时被静默跳过，权重未初始化 → NaN。

**修复**：

1. **`bailing_bridge.py`**：添加 SequentialMLP 格式的专家权重映射（参照 `qwen3_moe_bridge.py`）：
   ```python
   # SequentialMLP format (local_experts.E.linear_fc1.weight)
   GatedMLPMapping(
       megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
       gate="model.layers.*.mlp.experts.*.gate_proj.weight",
       up="model.layers.*.mlp.experts.*.up_proj.weight",
   )
   AutoMapping(
       megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
       hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
   )
   ```

2. **`model_bridge.py`**：`_megatron_local_name_to_global()` 添加 SequentialMLP 的 EP local→global 专家索引转换：
   ```python
   # EP for SequentialMLP (local_experts.N.xxx.weight)
   if ".mlp.experts.local_experts." in param_name and get_pg_size(ep_group) > 1:
       match = re.match(r"(.+\.local_experts\.)(\d+)(\..+)", param_name)
       if match:
           prefix, local_idx_str, suffix = match.groups()
           global_idx = num_experts_per_rank * ep_group.rank() + int(local_idx_str)
           param_name = f"{prefix}{global_idx}{suffix}"
   ```

3. **`param_mapping.py`**：`_normalize_expert_param_name()` 扩展支持 SequentialMLP 命名（`local_experts.N.xxx` 而非尾部数字 `weightN`）。

### 问题 2：MLA Q LoRA 权重未加载导致 NaN

**现象**：修复问题 1 后仍报 NaN。权重检查发现 `linear_q_down_proj.weight` 在所有 rank 上均为 NaN。

**根因**：Bailing V3 的 MLA 层使用了 `q_lora_rank=768`（Q LoRA 压缩），模型参数为 `linear_q_down_proj` + `linear_q_up_proj`，但 bridge 之前只映射了 `linear_q_proj`（非 LoRA 路径，MLA 层中不存在该参数），导致 Q LoRA 权重全部未加载。

**修复**：在 `bailing_bridge.py` 的 `param_mappings` 字典中添加 MLA Q LoRA 映射：
```python
# When q_lora_rank is set (LoRA path):
"decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.attention.q_a_proj.weight",
"decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.attention.q_b_proj.weight",
"decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.attention.q_a_layernorm.weight",
```

**对应关系**（参照 DeepSeek bridge）：

| Megatron 参数                          | HF 参数                            | 说明                                                                        |
| ------------------------------------ | -------------------------------- | ------------------------------------------------------------------------- |
| `linear_q_down_proj.weight`          | `attention.q_a_proj.weight`      | Q LoRA 降维 (hidden_size → q_lora_rank)，TELinear parallel_mode='duplicated' |
| `linear_q_up_proj.weight`            | `attention.q_b_proj.weight`      | Q LoRA 升维 (q_lora_rank → num_heads * q_head_dim)，ColumnParallel           |
| `linear_q_up_proj.layer_norm_weight` | `attention.q_a_layernorm.weight` | Q LoRA LayerNorm，融合在 linear_q_up_proj 内                                   |

### 修改文件汇总

| 文件 | 修改内容 |
|------|---------|
| `Megatron-Bridge/src/megatron/bridge/models/bailing/bailing_bridge.py` | 添加 SequentialMLP 专家映射 + MLA Q LoRA 映射 |
| `Megatron-Bridge/src/megatron/bridge/models/conversion/model_bridge.py` | `_megatron_local_name_to_global()` 添加 SequentialMLP EP 索引转换 |
| `Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py` | `_normalize_expert_param_name()` 支持 SequentialMLP 命名 |

### 调试方法

1. **权重 NaN 检查**：在 `setup.py` 的模型构建后、训练前，遍历所有参数检查 NaN/Inf：
   ```python
   for name, param in model.named_parameters():
       has_nan = torch.isnan(param).any().item()
       has_inf = torch.isinf(param).any().item()
       if has_nan or has_inf:
           print(f"[WEIGHT CORRUPTED] {name}: nan={has_nan}, inf={has_inf}")
   ```

2. **Forward dump**：在 MoE 各子模块入口/出口打印 tensor 统计信息（shape, dtype, sum, mean, max, min），定位 NaN 首次出现的位置。

3. **权重加载日志**：检查 `WARNING: No mapping found for megatron_param` 和 `WARNING: Can't find X in hf_keys` 消息，识别缺失映射。

---

## 已知限制和后续步骤

### 需要验证的问题

1. **MLA `g_proj` 门控**: Bailing 的 MLA 层有 `g_proj`（`gated_attention_proj_granularity_type=head_wise`），但 MCore 的 MLA 不支持 `attention_output_gate`（会抛 `NotImplementedError`）。可能需要：
   - 在 MCore MLA 中添加输出门控支持，或
   - 暂时跳过 `g_proj` 权重加载

2. **KDA 权重映射**: 当前使用简单的 `AutoMapping` 映射 KDA 权重。MCore 的 KDA 模块使用分离的投影（与 HF 结构匹配），应该可以直接工作，但需要验证 TP 分片是否正确。

3. **`transformer_layer_spec`**: Provider 当前使用 `get_gpt_decoder_block_spec`。对于混合 MLA+KDA 架构，需要使用 `get_transformer_block_with_experimental_attention_variant_spec` 来构建异构层规格。Bridge 中已设置此参数。

4. **MTP 层映射**: MTP 层（layer 42）使用 MLA 注意力 + MoE，需要确保 MTP 层的权重映射正确。

### 测试步骤

1. 在远程服务器运行 bridge 转换 UT，验证 HF→Megatron 权重转换
2. 使用 mock data 运行 `pretrain_bailing_v3.py`，验证 forward pass
3. 使用真实权重进行权重加载测试

### 远程路径

- Megatron-Bridge: `/mnt/sfs_turbo/wlp02367507/ling3/Megatron-Bridge/`
- Megatron-LM: `/mnt/sfs_turbo/wlp02367507/ling3/Megatron-LM/`
- 模型权重: `/sfs_turbo/pretrained_models/bailing_v3/sft_ring_flash_v3_formal_0429_rc53_baseline_100w_safegate_tkv3_wthink_ep8cp8pp4_iter_0000651/`