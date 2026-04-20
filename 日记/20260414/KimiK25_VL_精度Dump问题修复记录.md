# KimiK2.5 VL 精度 Dump 问题修复记录

> 日期：2026-04-14
> 目标：修复 msprobe 无法 dump Vision Tower 数据的问题

## 1. 问题背景

在 KimiK2.5 VL 模型的精度对比测试中，使用 msprobe 进行前向传播 dump 时，发现 Vision Tower (ViT) 部分的数据没有被记录。

初始状态：
- msprobe dump 结果：4453 个文件
- Vision/ViT 相关文件：0 个
- 只记录了 language_model 的数据

## 2. 问题分析与解决过程

### 2.1 问题一：input_ids 只有 4 个 token

**现象**：
```
input_ids: shape=torch.Size([1, 4]), dtype=torch.int64
```

**原因**：
`prepare_vl_input.py` 中调用 KimiK25Processor 的方式不正确。格式 0/1 (`images` + `text`) 不支持 system prompt，导致生成的文本序列不完整。

**解决方案**：
修改 `prepare_vl_input.py`，先用 `tokenizer.apply_chat_template` 构造完整文本，再用 `medias` 格式调用 processor：

```python
# 修改前
processed_input = _call_kimi_processor(
    processor,
    system_content=_SYSTEM_PROMPT,
    user_text=problem_text,
    pil_images=pil_images,
    return_tensors="pt",
)

# 修改后
messages = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    {"role": "user", "content": problem_text},
]
full_text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
medias = [{"type": "image", "image": img} for img in pil_images]
processed_input = processor(
    medias=medias,
    text=full_text,
    return_tensors="pt",
)
```

**结果**：
```
input_ids: shape=torch.Size([1, 78]), dtype=torch.int64  # 正确
```

### 2.2 问题二：multi_modal_input 未传递给模型

**现象**：
Hook 日志显示只有 `language_model.*` 层被触发，`vision_tower.*` 完全没有执行。

**原因**：
`test_kimik25_megatron_forward.py` 中调用 `prepare_data_and_fwd_function` 时，没有传递 `multi_modal_inputs` 参数，导致图像数据没有传递给模型。

**解决方案**：
修改 `test_kimik25_megatron_forward.py`，提取并传递 `multi_modal_inputs`：

```python
# 提取 multi_modal_input
multi_modal_inputs = None
if 'multi_modal_input' in sequence_sample.meta
    multi_modal_inputs = sequence_sample.metadata['multi_modal_input']

# 传递给 prepare_data_and_fwd_function
mb_inputs, forward_step, total_loss_weight = prepare_data_and_fwd_function(
    sequence_sample, mb_spec, megatron_backend.model_optimizer, None,
    multi_modal_inputs=multi_modal_inputs  # 关键修复
)
```

**验证**：
堆栈跟踪中出现 `_extract_image_features`，确认 Vision Tower 被执行。

### 2.3 问题三：attention_mask 类型错误

**现象**：
```
invalid attenMask dtype[DT_INT64], only support bool or uint8.
```

**原因**：
`actor_function.py` 中生成 attention_mask 时使用了 `.long()`，生成 int64 类型，但 NPU Flash Attention 需要 bool 或 uint8 类型。

**解决方案**：
修改远程服务器上的 `actor_function.py`：

```bash
sed -i 's/attn_mask = (input_ids_rmpad != _kimi_pad_token_id).long()/attn_mask = (input_ids_rmpad != _kimi_pad_token_id).bool()/g' \
    /sfs_turbo/hw/shiyang/code/kimi25vl_maijia_new/Asystem-HybridEngine/asystem_runtime/rl_function/actor_function.py
```

**验证**：
```
shy lm_attention_mask: tensor([[ True,  True,  True,  ..., False, False, False]], device='npu:0')
```

### 2.4 问题四：反向传播 OOM

**现象**：
```
RuntimeError: NPU out of memory. Tried to allocate 2.63 GiB
```

**原因**：
`megatron_backend.train()` 会执行完整的前向+反向传播，反向传播需要大量显存。

**解决方案**：
修改 `test_kimik25_megatron_forward.py`，只执行前向传播：

```python
# 修改前
outputs, stats = megatron_backend.train(
    mb_inputs,
    forward_step,
    lambda seq_sample: sum([...])
)

# 修改后
def data_iterator(mb_inputs):
    for mb_input in mb_inputs:
        yield mb_input

with torch.no_grad():
    batch_iter = data_iterator(mb_inputs)
    output_tensor, loss_func = forward_step(batch_iter, model)
```

## 3. 最终结果

### 3.1 前向传播成功

```
[Result] Megatron forward completed
[Result] outputs: <class 'torch.Tensor'>
[Result] output shape: torch.Size([167, 163840])
[Result] output sum:  4.288683e+07
[Result] output mean: 1.567428e+00

[DEBUG] Layer matching summary:
  Total triggered layers: 281
  Dumped layers: 34
```

### 3.2 msprobe dump 包含 ViT 数据

```
[INFO] dump.json data keys (4961 entries):
  [vision/vit]: 86 entries     ✅ Vision Tower 数据
    - Module.module.vision_tower.patch_embed.proj.Conv2d.forward.0...
    - Module.module.vision_tower.encoder.blocks.0.norm0.LayerNorm.forward.0...
  [mlp/moe]: 121 entries       ✅ 包含 ViT MLP
  [layernorm]: 126 entries     ✅ 包含 ViT LayerNorm
  [attention]: 53 entries      ✅ 注意力层
  [embedding]: 4 entries       ✅ 语言模型 embedding
```

## 4. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `prepare_vl_input.py` | 修改 processor 调用方式，使用 `medias` + `apply_chat_template` |
| `test_kimik25_megatron_forward.py` | 添加 `multi_modal_inputs` 传递，改为只执行前向传播 |
| `actor_function.py` (远程) | attention_mask 类型从 `.long()` 改为 `.bool()` |

## 5. 关键代码路径

### 5.1 数据流

```
prepare_vl_input.py
    └── processor(medias=..., text=full_text)  # 生成 pixel_values, image_grid_thw
        ↓
test_kimik25_megatron_forward.py
    └── sequence_sample.metadata['multi_modal_input'] = [mm_dict]
        ↓
prepare_data_and_fwd_function(..., multi_modal_inputs=...)
    └── mcore_model_forward_packed()
        └── model(input_ids=..., attention_mask=..., **multi_modal_inputs)
            ↓
modeling_kimi_k25_vl.py
    └── forward()
        ├── _extract_image_features(pixel_values, grid_thws)  # Vision Tower
        │   └── vision_tower(pixel_values, grid_thws)
        ├── mm_projector(image_features)
        └── language_model(inputs_embeds=...)  # Language Model
```

### 5.2 KimiK25VLModel 架构

```
KimiK25VLModel
├── vision_tower (MoonViT3dPretrainedModel)  # HF 原生模块
│   ├── patch_embed
│   ├── encoder.blocks[0-26]
│   └── final_layernorm
├── mm_projector                              # HF 原生模块
└── language_model (GPTModel)                 # Megatron 模块
    ├── embedding
    ├── decoder.layers[0-7]
    └── output_layer
```

## 6. 精度对比结果分析

### 6.1 对比结果摘要

运行 `compare_dumps.py` 后发现 HF 和 Megatron 之间存在显著差异：

| 指标 | HF | Megatron |
|------|-----|----------|
| 触发层数 | 1670 | 280 |
| input_ids shape | `[1, 78]` | `[1, 2137]` |
| hidden states shape | `[1, 167, 7168]` | `[2137, 1, 7168]` |
| 有效序列长度 | 167 (78文本 + 89图像) | 2137 (2048 padding + 89图像) |

### 6.2 关键发现

#### 6.2.1 维度顺序差异
- **HF**: `[batch, seq_len, hidden_dim]`
- **Megatron**: `[seq_len, batch, hidden_dim]`

这是 Megatron 的标准格式，需要在对比时进行 transpose。

#### 6.2.2 序列长度差异
- **HF**: 只有有效 token (78 文本 + 89 图像 token = 167)
- **Megatron**: padding 到 2048 + 89 图像 token = 2137

Megatron 侧的 padding 会影响 sum 统计量，但不应该影响有效位置的值。

#### 6.2.3 数据流分析（Layer 0）

```
Layer                           HF Shape           MEG Shape            CosineSim   Status
-------------------------------------------------------------------------------------------
embed_tokens (input)            [1, 78]            [1, 2137]            0.89        ***
embed_tokens (output)           [1, 78, 7168]      [1, 2137, 7168]      0.85        ***
layers.0.input_layernorm (in)   [1, 167, 7168]     [2137, 1, 7168]      1.00        OK ✓
layers.0.input_layernorm (out)  [1, 167, 7168]     [2137, 1, 7168]      1.00        OK ✓
layers.0.self_attn (out)        [1, 167, 7168]     [2137, 1, 7168]     -0.11        *** MISMATCH
layers.0.post_attn_ln (in)      [1, 167, 7168]     [2137, 1, 7168]      0.50        ***
layers.0.mlp (out)              [1, 167, 7168]     [2137, 1, 7168]     -0.03        ***
```

### 6.3 问题定位

**第一个匹配点**：`layers.0.input_layernorm` 的 input/output cosine similarity = 1.0

这说明：
1. ✅ Embedding 层权重正确
2. ✅ 图像特征合并正确（167 token = 78 文本 + 89 图像）
3. ✅ RMSNorm 实现一致

**差异起点**：`layers.0.self_attn` 输出的 cosine similarity = -0.11

可能原因：
1. **Attention 实现差异**：HF 使用标准 attention，Megatron 使用 NPU Flash Attention
2. **Attention mask 处理**：Megatron 的 attention_mask 可能没有正确应用到有效 token
3. **位置编码差异**：YarnRotaryEmbedding 的实现可能有差异

### 6.4 下一步排查方向

1. **检查 attention_mask**：确认 Megatron 侧的 attention_mask 是否正确标记了有效 token
2. **对比 attention 输出**：在 attention 内部插入更细粒度的 hook，对比 Q/K/V 和 attention weights
3. **检查位置编码**：对比 rotary_pos_emb 的输出
4. **验证 Flash Attention**：尝试使用 eager attention 代替 Flash Attention

### 6.5 Vision Tower 对比结果

运行 `compare_vision_tower.py` 后，确认 Vision Tower 完全匹配：

```
====================================================================================================
Vision Tower Layer Comparison (Output Statistics)
====================================================================================================
Layer Name                                                      HF sum        MEG sum      Diff%     Status
----------------------------------------------------------------------------------------------------
vision_tower.patch_embed.proj                             3.654201e+05   3.654201e+05      0.00%         OK
vision_tower.patch_embed.pos_emb                          3.728524e+05   3.728524e+05      0.00%         OK
vision_tower.patch_embed                                  3.728524e+05   3.728524e+05      0.00%         OK
vision_tower.encoder.blocks.0.norm0                       5.313054e+04   5.313054e+04      0.00%         OK
vision_tower.encoder.blocks.0.wqkv                        8.248708e+05   8.248708e+05      0.00%         OK
vision_tower.encoder.blocks.0.wo                          7.400112e+04   7.400112e+04      0.00%         OK
vision_tower.encoder.blocks.13.*                          ...            ...               0.00%         OK
vision_tower.encoder.blocks.26.*                          ...            ...               0.00%         OK
mm_projector.pre_norm                                     2.686125e+04   2.686125e+04      0.00%         OK
mm_projector.proj.*                                       ...            ...               0.00%         OK
====================================================================================================

Summary:
  Vision Tower layers compared: 31
  Matched (diff < 1%):          31
  Mismatched (diff > 1%):       0
```

**结论**：
- ✅ Vision Tower 全部 31 层完全匹配（差异 0.00%）
- ✅ mm_projector 完全匹配
- ✅ 图像特征提取和投影实现一致
- ✅ 权重转换正确

## 7. 问题定位总结

### 7.1 已排除的问题

| 组件 | 状态 | 说明 |
|------|------|------|
| Vision Tower | ✅ 正确 | 31 层全部匹配，差异 0% |
| mm_projector | ✅ 正确 | 图像特征投影一致 |
| Embedding 层 | ✅ 正确 | input_layernorm 输出 cosine sim = 1.0 |
| RMSNorm | ✅ 正确 | LayerNorm 实现一致 |

### 7.2 确认的问题

| 组件 | 状态 | 差异 |
|------|------|------|
| Language Model Attention | ❌ 不匹配 | self_attn 输出 cosine sim = -0.11 |
| Language Model MLP | ❌ 不匹配 | 差异从 attention 传播而来 |

### 7.3 根本原因分析

问题出在 **Language Model 的 Attention 层**：

1. **序列长度差异**：
   - HF: 167 token (78 文本 + 89 图像)
   - Megatron: 2137 token (2048 padding + 89 图像)

2. **Attention Mask 问题**：
   - Megatron 使用 `[1, 2137]` 的 mask
   - 可能没有正确区分有效 token 和 padding

3. **Flash Attention 差异**：
   - Megatron 使用 NPU Flash Attention
   - HF 使用标准 attention 实现

### 7.4 数据流对比

```
组件                          HF Shape           MEG Shape            Cosine Sim   状态
-------------------------------------------------------------------------------------------
embed_tokens (output)         [1, 78, 7168]      [1, 2137, 7168]      0.85         差异
+ 图像特征合并                [1, 167, 7168]     [2137, 1, 7168]      -            -
input_layernorm (output)      [1, 167, 7168]     [2137, 1, 7168]      1.00         ✅ 匹配
self_attn (output)            [1, 167, 7168]     [2137, 1, 7168]     -0.11         ❌ 差异点
mlp (output)                  [1, 167, 7168]     [2137, 1, 7168]     -0.03         ❌ 传播
```

## 8. 后续工作

1. **深入分析 attention 层**：对比 Q/K/V 投影和 attention weights
2. **验证 attention_mask**：确认 padding 位置的 mask 是否正确应用到有效 token
3. **检查位置编码**：YarnRotaryEmbedding 实现是否有差异
4. **尝试 eager attention**：排除 Flash Attention 的影响

## 9. 参考文件

- 数据处理流程：`D:\knowledge\大模型\多模态\KimiK25_VLM_图像文本处理流程.md`
- 测试脚本目录：`D:\work\KimiK2.5\precision_dump\`
- 远程代码路径：`/sfs_turbo/hw/shiyang/code/kimi25vl_maijia_new/`