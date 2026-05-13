# BailingV3 Tiny vs Flash 架构对比

> Tiny: `sft_ring_tiny_v3_formal_0402_rc52_baseline_100w_safegate_0403_iter_0000324`
> Flash: `sft_ring_flash_v3_formal_0429_rc53_baseline_100w_safegate_tkv3_wthink_ep8cp8pp4_iter_0000651`

## 1. 整体规模对比

| 参数 | Tiny | Flash |
|------|------|-------|
| 总权重文件大小 | 16.4 GB (~15.3B) | 255.0 GB (~237B) |
| 权重 key 数 | 9,686 | 63,783 |
| hidden_size | 1536 | 2560 |
| num_hidden_layers | 24 | 42 |
| num_attention_heads | 16 | 32 |
| num_key_value_heads | 16 | 32 |
| intermediate_size | 4608 | 6144 |
| vocab_size | 157184 | 157184 |

Flash 约为 Tiny 的 **15.5 倍**参数量。

## 2. 核心结构差异（config.json 逐项对比）

### 2.1 层分组与排布

| 参数 | Tiny | Flash |
|------|------|-------|
| layer_group_size | **4** | **6** |
| first_k_dense_replace | **1** | **2** |

**含义：**
- Tiny: 每4层一组 → 3×KDA + 1×MLA，第0层用 Dense MLP
- Flash: 每6层一组 → 5×KDA + 1×MLA，第0~1层用 Dense MLP

Flash 的 KDA:MLA 比例更高（5:1 vs 3:1），线性注意力占比更大，推理吞吐更高。

### 2.2 MoE 配置

| 参数 | Tiny | Flash |
|------|------|-------|
| num_experts | **128** | **512** |
| num_experts_per_tok | 8 | 8 |
| moe_intermediate_size | **512** | **768** |
| moe_shared_expert_intermediate_size | **512** | **768** |
| n_group | 8 | 8 |
| topk_group | 4 | 4 |
| routed_scaling_factor | 2.5 | 2.5 |

Flash 专家数量是 Tiny 的 4 倍（512 vs 128），每个专家中间维度也更大（768 vs 512）。

### 2.3 Q 投影结构

| 参数 | Tiny | Flash |
|------|------|-------|
| q_lora_rank | **256** | **null** |

**这是最关键的结构差异之一。**

- **Tiny**: Q 使用低秩投影 `q_a_proj(1536→256) → RMSNorm → q_b_proj(256→16*192)`，即 MLA 风格的双投影
- **Flash**: Q 直接全量投影 `q_proj(2560→32*192)`，无低秩压缩

Flash 的 MLA 只压缩了 KV（kv_lora_rank=512 保留），Q 端不做压缩。这意味着 Flash 的 Q 投影参数量更大，但推理时 Q 不需要缓存，所以不影响 KV cache 收益。

### 2.4 其他 config 差异

| 参数 | Tiny | Flash |
|------|------|-------|
| head_dim | 128 | 128 |
| qk_head_dim | 192 | 192 |
| qk_nope_head_dim | 128 | 128 |
| qk_rope_head_dim | 64 | 64 |
| kv_lora_rank | 512 | 512 |
| v_head_dim | 128 | 128 |
| num_nextn_predict_layers | 1 | 1 |
| mtp_loss_scaling_factor | 0 | 0 |
| max_position_embeddings | 131072 | 131072 |
| rope_theta | 6000000 | 6000000 |
| rope_interleave | true | true |
| kda_safe_gate | true | true |
| kda_lower_bound | -5.0 | -5.0 |
| no_kda_lora | true | true |
| use_qk_norm | true | true |
| gated_attention_proj_granularity_type | head_wise | head_wise |

**以上参数两者完全一致，架构范式相同。**

## 3. 建模代码差异（modeling_bailing_moe_v3.py）

两个版本的建模代码仅有 **42 行差异**，全部集中在 `BailingMoeV3KimiDeltaAttention` 类中：

### 3.1 KDA 门控投影命名

| | Tiny | Flash |
|---|------|-------|
| f 投影 | `f_a_proj` + `f_b_proj=None` | `f_proj` (单层) |
| g 投影 | `g_a_proj` + `g_b_proj=None` | `g_proj` (单层) |

Tiny 使用 `f_a_proj`/`g_a_proj` 命名（预留了 LoRA 分解的 `f_b_proj`/`g_b_proj` 位但设为 None），Flash 直接命名为 `f_proj`/`g_proj`，更简洁。

### 3.2 KDA 算子调用方式（核心差异）

**Tiny 版本：**
```python
# 1. 先在外面计算 gate
g = self.f_a_proj(hidden_states)
g = fused_kda_gate(g, self.A_log.view(1, 1, -1, 1), self.head_dim, g_bias=self.dt_bias)

# 2. 调用 chunk/fused_recurrent 时传入 safe_gate 参数
o, recurrent_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    initial_state=recurrent_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    safe_gate=self.safe_gate,       # <-- 传入 safe_gate
    lower_bound=self.lower_bound,
    cu_seqlens=cu_seqlens,
)
```

**Flash 版本：**
```python
# 1. 只做线性投影，不做 fused_kda_gate
g = self.f_proj(hidden_states)
g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)

# 2. 调用 chunk/fused_recurrent 时传入 A_log 和 dt_bias，由 kernel 内部计算 gate
o, recurrent_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    initial_state=recurrent_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    A_log=self.A_log,              # <-- 传入 kernel 内部
    dt_bias=self.dt_bias,          # <-- 传入 kernel 内部
    use_gate_in_kernel=True,       # <-- kernel 内部做 gate
    cu_seqlens=cu_seqlens,
)
```

**差异本质：**
- Tiny: gate 计算**外置** — 先用 `fused_kda_gate` 算出 g，再传入 kernel
- Flash: gate 计算**内置** — 传入原始投影 + A_log + dt_bias，由 CUDA kernel 内部融合计算

Flash 的方式更高效：将 gate 计算（A_log 指数衰减 + dt_bias 偏置 + sigmoid）融合到 KDA kernel 中，减少一次 kernel launch 和显存读写。这也是 "flash" 命名的由来 — **fused kernel 优化**。

### 3.3 配置导入路径

| Tiny | Flash |
|------|-------|
| `from .configuration_bailing_moe_v2_5 import BailingMoeV3Config` | `from .configuration_bailing_moe_v3 import BailingMoeV3Config` |

Flash 使用了更新的配置模块名（v2_5 → v3），但 configuration 代码内容完全一致。

### 3.4 fused_kda_gate 依赖

| Tiny | Flash |
|------|-------|
| `from fla.ops.kda.gate import fused_kda_gate` | 无此导入 |

Flash 不再需要单独的 gate 算子，因为已融合进 KDA kernel。

## 4. Chat Template 与 Think 模式

| | Tiny | Flash |
|---|------|-------|
| chat_template 位置 | tokenizer_config.json 内嵌 | 独立 chat_template.jinja 文件 |
| reasoning_content 支持 | 有 | 有 |
| think token (/\</think\>) | 有 (156903/156904) | 有 (156903/156904) |

两者都支持 think 模式，token 完全一致。Flash 从路径名 `wthink` 也可确认训练时包含 thinking 数据。

## 5. 训练配置推断

从路径名推断训练配置差异：

| | Tiny | Flash |
|---|------|-------|
| 训练日期 | 0402~0403 | 0429 |
| RC 版本 | rc52 | rc53 |
| 数据量 | 100w | 100w |
| safegate | 有 | 有 |
| think 训练 | 未明确 | wthink (明确包含) |
| EP/CP/PP | 未标注 | ep8cp8pp4 (8路专家并行+8路上下文并行+4路流水线并行) |
| 迭代步数 | 324 | 651 |

Flash 使用了更复杂的分布式策略 (EP+CP+PP)，适合大模型训练。

## 6. 适配训练框架的关键差异总结

从 Megatron-Core 适配角度，Tiny → Flash 需要注意：

1. **Q 投影结构不同** — Tiny 用 q_lora_rank=256 的低秩投影，Flash 用全量投影（q_lora_rank=null），权重不兼容
2. **KDA kernel 调用方式不同** — Flash 用 `use_gate_in_kernel=True`，需要 fla-core 支持融合 gate 的 KDA kernel
3. **layer_group_size 不同** — 4 vs 6，影响 KDA/MLA 层的交替模式
4. **first_k_dense_replace 不同** — 1 vs 2，影响前几层使用 Dense MLP 还是 MoE
5. **专家数量和维度不同** — 128/512 vs 768/768，EP 并行策略需要调整
6. **权重规模** — Flash 约为 Tiny 的 15.5 倍，需要 TP/EP/CP/PP 组合并行