# KimiK25 VLM 图像与文本处理流程

> 基于 `modeling_kimi25.py` 源码分析

## 1. 模型架构概览

```
KimiK25ForConditionalGeneration
├── vision_tower (MoonViT3dPretrainedModel)  # 视觉编码器
├── mm_projector (MLP / PatchMergerMLP)       # 图像特征投影层
└── language_model (DeepseekV3ForCausalLM)    # 语言模型
```

---

## 2. Forward 方法完整流程

### 2.1 输入参数

```python
def forward(
    self,
    input_ids: torch.LongTensor | None = None,      # 文本 token IDs [batch_size, seq_len]
    pixel_values: torch.FloatTensor | None = None,   # 图像像素值 [batch, channel, height, width]
    grid_thws: torch.Tensor | None = None,           # 图像网格尺寸 [batch, 3] (time, height, width)
    attention_mask: torch.Tensor | None = None,      # 注意力掩码 [batch_size, seq_len]
    position_ids: torch.LongTensor | None = None,    # 位置编码
    past_key_values: list | None = None,             # KV 缓存（推理时使用）
    inputs_embeds: torch.FloatTensor | None = None,  # 预计算的 embedding
    labels: torch.LongTensor | None = None,          # 训练标签
    ...
):
```

### 2.2 处理流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 1: 获取文本 Embedding                            │
│                                                                             │
│  inputs_embeds = self.get_input_embeddings()(input_ids)                     │
│  input_ids: [batch_size, seq_len]                                           │
│      ↓                                                                      │
│  inputs_embeds: [batch_size, seq_len, hidden_dim]                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 2: 提取图像特征                                   │
│                                                                             │
│  if pixel_values is not None and len(pixel_values) > 0:                     │
│      image_features = self._extract_image_features(pixel_values, grid_thws) │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Vision Tower (MoonViT3dPretrainedModel)                              │  │
│  │                                                                       │  │
│  │  pixel_values: [batch, channel, height, width]                        │  │
│  │       ↓                                                               │  │
│  │  Patch Embedding (MoonVision3dPatchEmbed)                             │  │
│  │       ↓                                                               │  │
│  │  Transformer Blocks (MoonViTEncoderLayer × N)                         │  │
│  │       │                                                               │  │
│  │       ├── attention_qkvpacked()                                       │  │
│  │       │     └── eager_attention() or flash_attention()                │  │
│  │       │                                                               │  │
│  │       └── MLP                                                         │  │
│  │       ↓                                                               │  │
│  │  LayerNorm                                                            │  │
│  │       ↓                                                               │  │
│  │  image_features: list[Tensor]                                         │  │
│  │  每个 Tensor: [num_image_tokens, vt_hidden_size]                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 3: 图像特征投影                                   │
│                                                                             │
│  if self.mm_projector:                                                      │
│      image_features = self.mm_projector(image_features)                     │
│                                                                             │
│  维度转换: vt_hidden_size → text_hidden_size                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 4: 合并文本和图像 Embedding                        │
│                                                                             │
│  inputs_embeds, attention_mask, labels, position_ids =                      │
│      self._merge_input_ids_with_image_features(...)                         │
│                                                                             │
│  合并逻辑:                                                                    │
│    1. 找到 input_ids 中的图像占位符位置 (media_placeholder_token_id)           │
│    2. 计算每个图像 token 需要扩展的位置                                        │
│    3. 创建扩展后的 final_embedding [batch, max_embed_dim, dim]               │
│    4. 将文本 embedding 填入非图像位置                                         │
│    5. 将图像 features 填入图像 token 位置                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 5: 语言模型前向                                   │
│                                                                             │
│  outputs = self.language_model(                                             │
│      attention_mask=attention_mask,                                         │
│      position_ids=position_ids,                                             │
│      inputs_embeds=inputs_embeds,  # 合并后的 embedding                      │
│      ...                                                                    │
│  )                                                                          │
│                                                                             │
│  language_model = DeepseekV3ForCausalLM (DeepSeek-V3 架构)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 关键函数详解

### 3.1 `_extract_image_features`

```python
def _extract_image_features(self, pixel_values: torch.Tensor,
                            grid_thws: torch.Tensor) -> list[torch.Tensor]:
    """
    提取图像特征

    Args:
        pixel_values: 图像像素值 [batch, channel, height, width]
        grid_thws: 图像网格尺寸 [batch, 3]，每行为 (time, height, width)

    Returns:
        image_features: list[Tensor]，每个 Tensor 形状为 [num_image_tokens, vt_hidden_size]
    """
    target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
    pixel_values = pixel_values.to(target_dtype)

    image_features = self.vision_tower(pixel_values, grid_thws)
    return image_features
```

### 3.2 `_merge_input_ids_with_image_features`

```python
def _merge_input_ids_with_image_features(
    self,
    image_features: list[torch.Tensor],  # 图像特征列表
    inputs_embeds: torch.Tensor,          # 文本 embedding [batch, seq_len, dim]
    input_ids: torch.Tensor,              # 包含图像占位符的 token IDs
    attention_mask: torch.Tensor,         # 注意力掩码
    labels: torch.Tensor | None,          # 标签
):
    """
    将图像特征合并到文本 embedding 中

    核心逻辑:
    1. 找到 input_ids 中的图像占位符 (media_placeholder_token_id)
    2. 计算每个图像占位符需要扩展的 token 数量
    3. 创建扩展后的 embedding 矩阵
    4. 填充文本和图像 embedding
    """
    # 获取图像 token 数量
    feature_lengths = [x.shape[0] for x in image_features]

    # 创建 token 占用表，用于计算扩展后的位置
    _token_occupation_table = torch.ones_like(input_ids.flatten())
    _token_occupation_table[input_ids.flatten() == image_token_index] = \
        torch.tensor(feature_lengths, dtype=torch.long, device=input_ids.device)

    # 计算最大扩展长度
    max_embed_dim = _token_occupation_table.sum(-1).max().item()

    # 创建扩展后的 embedding
    final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim, ...)
    final_attention_mask = torch.zeros(batch_size, max_embed_dim,
                                        dtype=attention_mask.dtype, ...)  # ← 这里会报错如果 attention_mask=None

    # 填充文本 embedding
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[...]

    # 填充图像 features
    final_embedding[image_to_overwrite] = image_features.reshape(-1, embed_dim)

    return final_embedding, final_attention_mask, final_labels, position_ids
```

---

## 4. Vision Tower 架构

### 4.1 MoonViT3dPretrainedModel

```
MoonViT3dPretrainedModel
├── patch_embed (MoonVision3dPatchEmbed)     # Patch 嵌入层
├── encoder (MoonViT3dEncoder)               # Transformer 编码器
│   ├── rope_2d (Rope2DPosEmbRepeated)       # 2D 旋转位置编码
│   └── blocks (ModuleList[MoonViTEncoderLayer])  # Transformer 块
└── final_layernorm (LayerNorm)              # 最终归一化
```

### 4.2 MoonViTEncoderLayer

```python
class MoonViTEncoderLayer(nn.Module):
    def forward(self, hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        # 注意力计算
        hidden_states = self.attention_qkvpacked(
            hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

### 4.3 注意力实现

```python
# 支持两种注意力实现
VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,  # Flash Attention (GPU)
    "eager": eager_attention,                   # Eager Attention (通用)
}

def eager_attention(q, k, v, q_cu_seqlens, k_cu_seqlens, max_seqlen_q, max_seqlen_k):
    """
    Eager Attention - O(N²) 内存复杂度

    ⚠️ 在 NPU 上 Flash Attention 不可用时会回退到此实现
    ⚠️ 大图像序列长度时容易 OOM
    """
    # 创建注意力掩码
    attention_mask = torch.zeros(...)

    # 计算注意力权重 - O(N²) 内存！
    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask

    # Softmax - OOM 可能发生点
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)

    attn_output = attn_weight @ v
    return attn_output
```

---

## 5. 数据流总结

```
输入数据:
┌──────────────────┐    ┌──────────────────────┐
│  input_ids       │    │  pixel_values         │
│  [B, text_len]   │    │  [B, C, H, W]         │
│  包含图像占位符    │    │  图像像素值            │
│  (media_token)   │    │                       │
└────────┬─────────┘    └──────────┬───────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│  Text Embedding  │    │  Vision Tower        │
│  [B, text_len,   │    │  (MoonViT3d)          │
│   hidden_dim]    │    │                       │
└────────┬─────────┘    │  ⚠️ OOM 风险点:        │
         │              │  eager_attention()    │
         │              │  O(N²) 内存           │
         │              └──────────┬───────────┘
         │                         │
         │                         ▼
         │              ┌──────────────────────┐
         │              │  MM Projector        │
         │              │  (MLP/PatchMerger)   │
         │              │  vt_dim → text_dim   │
         │              └──────────┬───────────┘
         │                         │
         │                         ▼
         │              ┌──────────────────────┐
         │              │  image_features      │
         │              │  list of [N, dim]    │
         │              └──────────┬───────────┘
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  _merge_input_ids_with_    │
         │  image_features()          │
         │                            │
         │  ⚠️ 需要 attention_mask    │
         │  不能为 None               │
         │                            │
         │  替换图像占位符为实际特征    │
         │  [B, merged_len, dim]      │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  DeepSeekV3 LLM            │
         │  (语言模型前向)             │
         └────────────────────────────┘
```

---

## 6. 已知问题与修复

### 6.1 attention_mask=None 导致 AttributeError

**问题描述**:
```python
# actor_function.py 中调用模型时传递 attention_mask=None
output_orig = model(
    input_ids=input_ids_rmpad,
    attention_mask=None,  # ← 传递了 None
    ...
)

# modeling_kimi25.py:959 中直接访问 dtype
final_attention_mask = torch.zeros(batch_size, max_embed_dim,
                                   dtype=attention_mask.dtype,  # ← None.dtype 报错!
                                   device=inputs_embeds.device)
```

**修复方案**:
```python
# 在调用 _merge_input_ids_with_image_features 之前添加
if attention_mask is None:
    attention_mask = torch.ones(
        input_ids.shape[:2],
        dtype=torch.long,
        device=input_ids.device
    )
```

### 6.2 Vision Tower OOM 问题

**问题描述**:
- NPU 上 Flash Attention 不可用
- 回退到 Eager Attention (O(N²) 内存)
- 大图像序列长度时容易 OOM

**可能的解决方案**:
1. 设置内存碎片优化: `export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:32"`
2. 减小 micro_batch_size
3. 启用视觉塔重计算
4. 实现 NPU 兼容的高效注意力

---

## 7. 相关文件

| 文件 | 用途 |
|------|------|
| `modeling_kimi25.py` | KimiK25 VLM 模型定义 |
| `configuration_kimi_k25.py` | 模型配置 |
| `actor_function.py` | RL 训练中的模型调用 |
| `megatron_backend.py` | Megatron 后端训练逻辑 |

---

## 8. 参考资料

- [LLaVA 模型架构](https://github.com/haotian-liu/LLaVA)
- [DeepSeek-V3 论文](https://arxiv.org/abs/2412.19437)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)