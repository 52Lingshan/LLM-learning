# Qwen3.5 vs BailingV3 模型架构对比

> 基于远程权重路径 `/sfs_turbo/pretrained_models/bailing_v3/sft_ring_tiny_v3_formal_0402_rc52_baseline_100w_safegate_0403_iter_0000324` 的 config.json 和 modeling_bailing_moe_v3.py 分析

## 1. 整体架构范式

两者都采用 **Hybrid MoE + 线性注意力混合** 架构，但具体实现差异较大：

| 特性 | Qwen3.5-397B-A17B | BailingV3 (tiny) |
|------|-------------------|-------------------|
| 总参数/激活参数 | 397B / 17B | ~3B (hidden=1536) |
| 层数 | 60 | 24 |
| hidden_size | 4096 | 1536 |
| vocab_size | 248,320 | 157,184 |
| max_position | 262,144 (YaRN→1M) | 131,072 |

## 2. 核心区别：注意力机制

**这是最大的差异点。**

### 2.1 线性注意力层

| | Qwen3.5 | BailingV3 |
|---|---------|-----------|
| 类型 | **Gated DeltaNet** | **KDA** (Kimi Delta Attention) |
| V heads | 64 | — |
| QK heads | 16 | — |
| head_dim | 128 | 128 |
| 实现依赖 | — | fla-core (`fla.ops.kda`) |
| Short Conv | — | kernel_size=4 (q/k/v 各一个) |
| 衰减机制 | — | A_log (可学习, init 1~16) + dt_bias |
| 门控 | — | f_proj (no_kda_lora时直接投影) + g_proj |
| 输出归一化 | — | FusedRMSNormGated (sigmoid门控) |

### 2.2 Softmax 注意力层

| | Qwen3.5 | BailingV3 |
|---|---------|-----------|
| 类型 | **标准 GQA** | **MLA** (Multi-Latent Attention) |
| Q heads | 32 | 16 |
| KV heads | 2 | 16 (但通过MLA压缩) |
| head_dim | 256 | qk_head_dim=192 (nope=128 + rope=64) |
| RoPE dim | 64 | 64 |
| KV 压缩 | 无 | kv_lora_rank=512 (kv_a_proj → kv_b_proj) |
| Q 压缩 | 无 | q_lora_rank=256 (q_a_proj → RMSNorm → q_b_proj) |
| Gated proj | 无 | head_wise (g_proj: hidden→num_heads, sigmoid门控) |

**MLA 核心流程 (BailingV3):**
```
Q: hidden → q_a_proj(1536→256) → RMSNorm → q_b_proj(256→16*192)
   → split(qk_nope=128, qk_rope=64)

K: hidden → kv_a_proj_with_mqa(1536→512+64)
   → split(kv_compressed=512, k_rot=64)
   → kv_compressed → kv_a_layernorm → kv_b_proj(512→16*(128+128))
   → split(k_pass=128, v=128)

V: 从 kv_b_proj 输出中分离

K_final = cat(k_pass, k_rot.expand)  # [B, 16, S, 192]
Q_final = cat(q_pass, q_rot)          # [B, 16, S, 192]
```

### 2.3 层间排布

| | Qwen3.5 | BailingV3 |
|---|---------|-----------|
| 分组模式 | 15组 × (3×DeltaNet + 1×Attention) | layer_group_size=4, 每4层: 3×KDA + 1×MLA |
| 判断逻辑 | — | `(layer_idx + 1) % layer_group_size == 0` → Attention, 否则 → Linear |

BailingV3 的 DecoderLayer.__init__ 中:
```python
self.attention_layer_type = (
    "attention"
    if (layer_idx + 1) % config.layer_group_size == 0
    or layer_idx >= config.num_hidden_layers // config.layer_group_size * config.layer_group_size
    else "linear_attention"
)
```

## 3. MoE 路由

| | Qwen3.5 | BailingV3 |
|---|---------|-----------|
| 总专家数 | 512 | 128 |
| 激活专家数 | 10 routed + 1 shared | 8 routed + 1 shared |
| 专家中间维度 | 1024 | 512 |
| 共享专家中间维度 | — | 512 (1个共享专家) |
| 路由评分 | — | sigmoid |
| 路由选择 | — | **noaux_tc** + group_limited_topk |
| 分组数 | — | n_group=8, topk_group=4 |
| expert_bias | — | True (可训练偏置) |
| routed_scaling_factor | — | 2.5 |
| 归一化 | — | norm_topk_prob=True |
| seq_aux | — | True (序列级辅助损失) |
| first_k_dense_replace | — | 1 (第0层用Dense MLP) |

**BailingV3 路由流程:**
```
hidden → Gate.linear(hidden, weight) → logits (fp32)
       → sigmoid(logits) → scores
       → scores + expert_bias → group_limited_topk:
           1. scores reshape → (tokens, n_group, experts_per_group)
           2. 每组取 top2 求和 → group_scores
           3. 取 topk_group=4 个组 → group_mask
           4. mask 掉非选中组的专家
           5. 在剩余专家中取 top_k=8
       → gather scores → normalize → * routed_scaling_factor
```

## 4. 其他差异

### 4.1 MTP (Multi-Token Prediction)

| | Qwen3.5 | BailingV3 |
|---|---------|-----------|
| 层数 | multi-steps | 1层 (num_nextn_predict_layers=1) |
| 结构 | — | eh_proj(cat(embed, hidden)) → Attention(MLA) + MoE |
| loss scaling | — | mtp_loss_scaling_factor=0 (即不参与训练loss) |

### 4.2 位置编码

| | Qwen3.5 | BailingV3 |
|---|---------|-----------|
| RoPE theta | — | 6,000,000 (比Qwen3的1M大6倍) |
| RoPE interleave | — | True (交错式: q.view(..., d//2, 2).transpose) |
| partial_rotary_factor | — | 0.5 (仅对 qk_rope_head_dim=64 维度施加RoPE) |

### 4.3 归一化与门控

| | Qwen3.5 | BailingV3 |
|---|---------|-----------|
| QK Norm | — | use_qk_norm=True |
| GroupNorm | — | group_norm_size=1 (实际退化为RMSNorm) |
| Gated attention | — | head_wise (g_proj: hidden→num_heads, sigmoid门控注意力输出) |
| linear_silu | — | True |

### 4.4 KDA 特有参数

| 参数 | 值 | 说明 |
|------|-----|------|
| kda_safe_gate | True | 安全门控,防止衰减爆炸 |
| kda_lower_bound | -5.0 | 衰减下界 |
| no_kda_lora | True | 不使用LoRA分解,直接全量投影 |
| use_kda_lora | False | 同上 |
| short_conv_kernel_size | 4 | 短卷积核大小 |

## 5. 与 Qwen3 MoE 的对比参考

BailingV3 相比 Qwen3 MoE（如 Qwen3-30B-A3B）的进化方向与 Qwen3.5 一致：

| | Qwen3 MoE | Qwen3.5 | BailingV3 |
|---|-----------|---------|-----------|
| 注意力 | 纯标准 GQA | Gated DeltaNet + 标准 GQA | KDA + MLA |
| MoE 路由 | 标准 topk | 改进路由 | noaux_tc + group_limited_topk |
| KV Cache | 标准 | 标准 | MLA 压缩 (大幅节省) |
| 线性注意力 | 无 | Gated DeltaNet | KDA |

## 6. 适配训练框架的关键工作量

如果要在 Megatron-Core 等框架中适配 BailingV3，主要工作量在：

1. **MLA 注意力** — kv_a_proj / kv_b_proj 的低秩压缩结构，以及 RoPE interleave 的特殊处理
2. **KDA 线性注意力** — 需要移植 fla-core 的 chunk_kda / fused_recurrent_kda 算子，以及 ShortConvolution + A_log 衰减机制
3. **混合层排布** — 每4层交替 KDA/MLA，需要框架支持异构层
4. **MoE 路由** — group_limited_topk + expert_bias + sigmoid 评分，与标准 topk 路由不同
5. **MTP** — eh_proj 跨层投影结构