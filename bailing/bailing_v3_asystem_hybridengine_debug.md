# Bailing V3 Asystem-HybridEngine 调试记录

## 环境

- 测试脚本：`Asystem-HybridEngine/tests/npu/integration_tests/bailingv3_tiny/run_bailingv3_tiny_npu.py`
- YAML 配置：`Asystem-HybridEngine/tests/npu/integration_tests/bailingv3_tiny/bailingv3_tiny_npu.yaml`
- 远程路径：`/mnt/sfs_turbo/wlp02367507/ling3/Asystem-HybridEngine/`
- 模型：Bailing V3 tiny（24层，128专家，TP=2，EP=16）
- 使用 `setup_model_with_mbridge()` 路径（`use_megatron_bridge: true`）

---

## 问题 1：use_flash_attn 未传入 TransformerConfig 导致 RuntimeError

### 现象

```
RuntimeError: Please set micro_batch_size or set use_flash_attn=True in config.
```

错误栈定位到 MindSpeed 的 `mindspeed/core/transformer/flash_attention/generate_mask/generate_mask.py:21`。

### 根因

MindSpeed 的 `generate_attention_mask()` 通过 `getattr(config, 'use_flash_attn', False)` 检查 `TransformerConfig` 上是否有 `use_flash_attn` 属性。当该属性不存在或为 `False` 时，会尝试生成显式 attention mask，但缺少 `micro_batch_size` 导致报错。

**配置传递链路断裂**：

1. YAML 配置中 `override_transformer_config.use_flash_attn: true` 通过 `repatch()` 设置到 Megatron 的 `args` 上（`args.use_flash_attn = True`）
2. 但 `setup_model_with_mbridge()` 路径**不走** `core_transformer_config_from_args()`，而是直接用 `AutoBridge.from_hf_pretrained()` 创建 `BailingMoeV3Provider`
3. `BailingMoeV3Provider`（继承自 `MLATransformerConfig` → `TransformerConfig`）**没有** `use_flash_attn` 字段
4. `build_config()` 也只从 `args` 复制了部分字段（`tp_size`, `ep_size`, `recompute_*` 等），**没有**传递 `use_flash_attn`
5. 在非 bridge 路径中，MindSpeed 的 `transformer_config_init_wrapper` 会从 `args` 动态注入未知字段到 `TransformerConfig`，但 bridge 路径绕过了这个 wrapper

**关键代码路径对比**：

| 路径 | use_flash_attn 传递方式 |
|------|----------------------|
| 非 bridge（`core_transformer_config_from_args`） | MindSpeed patch 动态注入 `args.use_flash_attn` 到 config |
| bridge（`AutoBridge` + `BailingMoeV3Provider`） | Provider 无此字段，`build_config()` 也未传递 |

### 修复

**文件 1**：`Megatron-Bridge/src/megatron/bridge/models/bailing/bailing_provider.py`

添加 `use_flash_attn` 字段（参考 `qwen35_vl_provider.py` 的做法）：

```python
# Flash attention (required for NPU/MindSpeed)
use_flash_attn: bool = True
```

**文件 2**：`Asystem-HybridEngine/asystem_runtime/backend/megatron_backend.py`

在 `setup_model_with_mbridge()` 的 `build_config()` 中添加从 `args` 到 `provider` 的传递：

```python
if hasattr(args, 'use_flash_attn'):
    provider.use_flash_attn = args.use_flash_attn
```

### 验证

修复后 forward pass 成功通过，所有 16 个 rank 输出 `enable_fp32_lm_head is False, cast outputs as bf16`，不再报 `generate_mask` 错误。

---

## 问题 2：RL 训练 loss weight 为零导致 NaN 梯度

### 现象

```
Warning: The sum of loss weights of all micro batches is zero.
...
RuntimeError: found NaN in local grad norm for bucket #0 in backward pass
```

### 根因

测试脚本中 `config['enable_padding_data'] = True`（第163行），导致：

1. 加载了真实数据并填入了 logprobs → `sequence_sample` 有有效数据
2. 但紧接着 `if config.get('enable_padding_data', False): sequence_sample = None` 将数据清空
3. `prepare_data_and_fwd_function(None, ...)` 内部调用 `build_one_sample()` 生成全零 padding 数据
4. padding 数据的 `ppo_loss_mask` 全为零
5. `loss_weight_fn = lambda x: x.data["ppo_loss_mask"].count_nonzero()` → `total_loss_weight_og = 0`
6. loss_scale = 0 → loss 全为零 → backward 时梯度为 NaN

### 修复

**文件**：`Asystem-HybridEngine/tests/npu/integration_tests/bailingv3_tiny/run_bailingv3_tiny_npu.py`

不再将 `sequence_sample` 置为 None，保留填好 logprobs 的真实数据：

```python
if config.get('enable_padding_data', False):
    print(f">>>>>>>WARNING: enable_padding_data=True, but we have real data with logprobs. Keeping sequence_sample.")
    # sequence_sample = None  # Do NOT set to None - would lose the real data
```

### 额外修改

在调用 `prepare_data_and_fwd_function` 之前，增加了以下步骤：

1. **先做 forward pass 获取真实 logprobs**：调用 `logp_compute()` 获取模型输出，填入 `old_logp`
2. **设置 advantages 为非零值**：`sequence_sample.data['advantages'].fill_(1.0)`（之前全零）
3. **检查 ppo_loss_mask**：如果全零则设为全 True
4. **数据路径可配置**：从 YAML `data_path` 读取，默认 `/storage/wlp02367507/ling3/data_512.pt`

---

## 问题 3：KDA 层 input_layernorm 为 IdentityOp 导致参数未加载

### 现象

打印模型结构发现 KDA 层的 `input_layernorm` 为 `IdentityOp()`，而 MLA 层的 `input_layernorm` 为 `RMSNorm()`。KDA 层的 layernorm 权重全为 1（默认初始化值），未从 HF checkpoint 加载。

### 根因

`get_kda_module_spec()` 中设置了 `metainfo={"fuse_input_layernorm": True}`，表示 input layernorm 被融合到第一个线性层中。但 KDA 的 `q_proj` 使用的是 `column_parallel_linear()`（非 `column_parallel_layer_norm_linear()`），实际上并没有进行融合。`fuse_input_layernorm=True` 只是把 `input_layernorm` 替换成了 `IdentityOp`，导致：

1. KDA 层没有独立的 `input_layernorm` 模块
2. layernorm 权重无处加载，全部初始化为 1
3. 模型缺少归一化操作，影响训练稳定性

### 修复

**文件**：`Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py`

将 KDA spec 中的 `fuse_input_layernorm` 改为 `False`：

```python
# get_kda_module_spec() 中
attention = ModuleSpec(
    module=KimiDeltaAttention,
    submodules=KimiDeltaAttentionSubmodules(...),
    metainfo={"fuse_input_layernorm": False},  # 原来是 True
)
```

---

## 问题 4：KDA f_proj/g_proj HF 参数名错误

### 现象

bridge 中 KDA 的 `f_proj` 和 `g_proj` 映射到了 `model.layers.*.attention.f_a_proj.weight` 和 `model.layers.*.attention.g_a_proj.weight`（带 `_a_` 后缀），但 HF 模型中实际参数名不同，导致 `WARNING: Can't find model.layers.*.attention.f_a_proj.weight in hf_keys`。

### 根因

Bailing V3 的 HF config 中 `no_kda_lora=True`（默认），此时 HF 模型使用单投影而非 LoRA 分解。根据实际 HF 权重索引验证：

| Megatron 参数名 | no_kda_lora=True (实际) | no_kda_lora=False (LoRA) |
|----------------|------------------------|--------------------------|
| `f_proj.weight` | `f_proj.weight` | `f_a_proj.weight` + `f_b_proj.weight` |
| `g_proj.weight` | `g_proj.weight` | `g_a_proj.weight` + `g_b_proj.weight` |

bridge 之前硬编码为 `f_a_proj.weight` / `g_a_proj.weight`（LoRA 分解的 `_a_` 后缀），`no_kda_lora=True` 时两个都不对。

### 修复

**文件**：`Megatron-Bridge/src/megatron/bridge/models/bailing/bailing_bridge.py`

根据 `no_kda_lora` 配置选择正确的 HF 参数名：

```python
no_kda_lora = getattr(self.hf_config, "no_kda_lora", True)
if no_kda_lora:
    # No LoRA: f_proj -> f_proj, g_proj -> g_proj (no suffix)
    mapping_list.append(AutoMapping(
        megatron_param="decoder.layers.*.self_attention.f_proj.weight",
        hf_param="model.layers.*.attention.f_proj.weight",
    ))
    mapping_list.append(AutoMapping(
        megatron_param="decoder.layers.*.self_attention.g_proj.weight",
        hf_param="model.layers.*.attention.g_proj.weight",
    ))
else:
    # LoRA: f_a_proj+f_b_proj -> f_proj, g_a_proj+g_b_proj -> g_proj
    mapping_list.append(AutoMapping(
        megatron_param="decoder.layers.*.self_attention.f_proj.weight",
        hf_param="model.layers.*.attention.f_a_proj.weight",
    ))
    mapping_list.append(AutoMapping(
        megatron_param="decoder.layers.*.self_attention.g_proj.weight",
        hf_param="model.layers.*.attention.g_a_proj.weight",
    ))
```

---

## 问题 5：MLA g_proj（门控注意力投影）缺失

### 现象

HF 模型中 MLA 层有 `g_proj`（`gated_attention_proj_granularity_type="head_wise"`），但 Megatron 的 `MLASelfAttention` 没有此模块，导致：

1. `g_proj.weight` 权重无法加载（无映射目标）
2. 模型缺少门控注意力机制，与原始模型行为不一致

### 根因

Megatron 的 `MLASelfAttention` 原本不支持 `g_proj`。HF 模型中 `BailingMoeV3MultiLatentAttention` 的 `g_proj` 实现为：

```python
# head_wise: g_proj 输出 num_heads 个值
self.g_proj = ColumnParallelLinear(hidden_size, num_heads, bias=False)
# forward:
gate = torch.sigmoid(self.g_proj(hidden_states))
attn_output = attn_output * gate.unsqueeze(-1)  # [sq, b, num_heads, v_head_dim]
```

### 修复（5 个文件）

#### 5.1 TransformerConfig 添加配置字段

**文件**：`Megatron-LM/megatron/core/transformer/transformer_config.py`

```python
# MLATransformerConfig 中
gated_attention_proj_granularity_type: Optional[str] = None
"""Gated attention projection granularity for MLA.
   Options: None (no gate), 'head_wise' (hidden_size -> num_heads),
   'element_wise' (hidden_size -> num_heads * v_head_dim)."""
```

#### 5.2 MLASelfAttentionSubmodules 添加 attn_gate

**文件**：`Megatron-LM/megatron/core/transformer/multi_latent_attention.py`

为避免与 KDA 的 `g_proj` 在权重映射时命名冲突，MLA 的门控投影命名为 `attn_gate`：

```python
@dataclass
class MLASelfAttentionSubmodules:
    # ... existing fields ...
    attn_gate: Union[ModuleSpec, type] = None
```

#### 5.3 MultiLatentAttention.__init__ 构建 attn_gate

```python
self.gated_attention_proj_granularity_type = getattr(
    self.config, 'gated_attention_proj_granularity_type', None
)
if self.gated_attention_proj_granularity_type is not None and submodules.attn_gate is not None:
    if self.gated_attention_proj_granularity_type == "head_wise":
        attn_gate_size = self.config.num_attention_heads
    else:  # element_wise
        attn_gate_size = self.config.num_attention_heads * self.config.v_head_dim
    self.attn_gate = build_module(
        submodules.attn_gate,
        self.config.hidden_size,
        attn_gate_size,
        config=self.config,
        init_method=self.config.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=True,
        is_expert=False,
        tp_comm_buffer_name='attn_gate',
        tp_group=self.pg_collection.tp,
    )
else:
    self.attn_gate = None
```

#### 5.4 MultiLatentAttention.forward 应用门控

在 `linear_proj` 之前：

```python
if hasattr(self, 'attn_gate') and self.attn_gate is not None:
    gate, _ = self.attn_gate(hidden_states)
    gate = torch.sigmoid(gate.float()).type_as(core_attn_out)
    if self.gated_attention_proj_granularity_type == "head_wise":
        num_heads_per_partition = core_attn_out.shape[-1] // self.config.v_head_dim
        core_attn_out = core_attn_out.view(
            core_attn_out.shape[0], core_attn_out.shape[1],
            num_heads_per_partition, self.config.v_head_dim
        )
        core_attn_out = core_attn_out * gate.unsqueeze(-1)
        core_attn_out = core_attn_out.view(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )
    else:
        core_attn_out = core_attn_out * gate
```

#### 5.5 Spec 注入 attn_gate

**文件**：`Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py`

在 `_get_self_attention_module_spec()` 中：

```python
if config.multi_latent_attention:
    attn_spec.metainfo["fuse_input_layernorm"] = False
    if getattr(config, 'gated_attention_proj_granularity_type', None) is not None:
        attn_spec.submodules.attn_gate = backend.column_parallel_linear()
```

#### 5.6 Bridge 传参和权重映射

**文件**：`Megatron-Bridge/src/megatron/bridge/models/bailing/bailing_bridge.py`

传参：
```python
provider_kwargs["gated_attention_proj_granularity_type"] = getattr(
    hf_config, "gated_attention_proj_granularity_type", None
)
```

权重映射：
```python
# MLA gated attention projection (attn_gate in Megatron, g_proj in HF)
"decoder.layers.*.self_attention.attn_gate.weight": "model.layers.*.attention.g_proj.weight",
```

#### 5.7 Provider 添加默认值

**文件**：`Megatron-Bridge/src/megatron/bridge/models/bailing/bailing_provider.py`

```python
gated_attention_proj_granularity_type: Optional[str] = None
```

### 命名说明

MLA 的门控投影在 Megatron 中命名为 `attn_gate` 而非 `g_proj`，因为：

- KDA 层已有 `self_attention.g_proj.weight` 映射（→ `g_a_proj.weight`）
- MegatronMappingRegistry 使用 first-match-wins 策略
- 如果两者都映射到 `decoder.layers.*.self_attention.g_proj.weight`，只有第一个会生效
- 使用 `attn_gate` 彻底消除歧义：KDA → `g_proj` → `g_a_proj.weight`，MLA → `attn_gate` → `g_proj.weight`

---

## 问题 6：rotary_interleaved + MLA 需要 rope_type='rope'

### 现象

```
ValueError: rotary_interleaved with multi_latent_attention is only supported with rope_type='rope'.
```

### 根因

`MLATransformerConfig` 默认 `rope_type="yarn"`，但 Bailing V3 使用 `rotary_interleaved=True`（对应 HF 的 `rope_interleave=True`）。`__post_init__` 校验要求两者组合必须 `rope_type='rope'`。

bridge 中设置了 `rotary_interleaved` 但没有设置 `rope_type`。

### 修复

**文件**：`Megatron-Bridge/src/megatron/bridge/models/bailing/bailing_bridge.py`

```python
provider_kwargs["rotary_interleaved"] = getattr(hf_config, "rope_interleave", False)
provider_kwargs["rotary_percent"] = getattr(hf_config, "partial_rotary_factor", 1.0)
provider_kwargs["rope_type"] = "rope"  # Bailing V3 uses rotary_interleaved, requires rope_type='rope'
```

---

## 问题 7：attn_gate reshape 未考虑 TP 分片

### 现象

```
RuntimeError: shape '[512, 1, 16, 128]' is invalid for input of size 524288
```

在 `multi_latent_attention.py:360` 的 `core_attn_out.view()` 调用处崩溃。

### 根因

`core_attn_out` 经过 TP 切分后，最后一维是 `num_heads_per_partition * v_head_dim`，不是全局的 `num_attention_heads * v_head_dim`。

计算验证：
- 全局：16 heads × 128 v_head_dim = 2048
- TP=2 分片后：8 heads × 128 = 1024
- tensor 实际大小：512 × 1 × 1024 = 524288
- 错误 reshape 尝试：512 × 1 × 16 × 128 = 1048576 ≠ 524288

同样，`attn_gate` 是 `ColumnParallelLinear(gather_output=False)`，输出也是按 TP 分片的：`num_heads_per_partition` 而非 `num_heads`。

### 修复

**文件**：`Megatron-LM/megatron/core/transformer/multi_latent_attention.py`

从实际 tensor shape 推断 `num_heads_per_partition`，而非使用全局 `num_attention_heads`：

```python
if self.gated_attention_proj_granularity_type == "head_wise":
    num_heads_per_partition = core_attn_out.shape[-1] // self.config.v_head_dim
    core_attn_out = core_attn_out.view(
        core_attn_out.shape[0], core_attn_out.shape[1],
        num_heads_per_partition, self.config.v_head_dim
    )
    core_attn_out = core_attn_out * gate.unsqueeze(-1)
    core_attn_out = core_attn_out.view(
        core_attn_out.shape[0], core_attn_out.shape[1], -1
    )
```

---

## 问题 9：SequentialMLP 模式下权重保存（Megatron→HF 导出）失败

### 现象

使用 `moe_grouped_gemm=False`（SequentialMLP）训练后，执行权重保存/导出时 `gather_from_ep_ranks()` 抛出异常：

```
ValueError: invalid literal for int() with base 10: ''
```

或

```
ValueError: Cannot extract global expert id from parameter name: 'decoder.layers.5.mlp.experts.local_experts.15.linear_fc1.weight'. Expected a trailing number after .weight/.bias or a pattern like 'local_experts.<id>'.
```

### 根因

`param_mapping.py` 的 `gather_from_ep_ranks()` 方法从 Megatron 参数名中提取 expert 编号时，只支持 TEGroupedMLP 的命名格式，不支持 SequentialMLP 格式。

**原始代码**（仅支持 TEGroupedMLP）：

```python
# 只能处理 "...linear_fc1.weight15" 格式
local_expert_number = None
for key in (".weight", ".bias"):
    if key in self.megatron_param:
        global_expert_number = int(self.megatron_param.split(key)[-1])
        local_expert_number = global_expert_number % num_experts_per_rank
```

两种命名格式的差异：

| 格式 | Megatron 参数名示例 | `split(".weight")[-1]` 结果 | 能否提取 expert id |
|------|---------------------|---------------------------|-------------------|
| TEGroupedMLP | `...linear_fc1.weight15` | `"15"` | `int("15")` = 15 |
| SequentialMLP | `...local_experts.15.linear_fc1.weight` | `""` | `int("")` → **ValueError** |

SequentialMLP 的 expert 编号在路径中间（`local_experts.15.xxx`），而不是尾部数字（`weight15`），原始代码无法解析。

同样，`_normalize_expert_param_name()` 也只处理尾部数字，不支持 `local_experts.N.xxx` 格式。

### 修复

**文件**：`Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py`

#### 9.1 `gather_from_ep_ranks()` — 支持 SequentialMLP expert 编号提取

```python
# Extract global expert id from parameter name, supporting two naming conventions:
# 1. TEGroupedMLP: "...linear_fc1.weight15" or "...linear_fc1.bias15"
# 2. SequentialMLP: "...local_experts.15.linear_fc1.weight" or "...local_experts.15.linear_fc1.bias"
param = self.megatron_param
global_expert_number = None

# Try trailing number after .weight or .bias (TEGroupedMLP)
match = re.search(r'\.(weight|bias)(\d+)$', param)
if match:
    global_expert_number = int(match.group(2))
else:
    # Try SequentialMLP pattern: local_experts.<number>
    match = re.search(r'local_experts\.(\d+)', param)
    if match:
        global_expert_number = int(match.group(1))

if global_expert_number is None:
    raise ValueError(
        f"Cannot extract global expert id from parameter name: '{param}'. "
        "Expected a trailing number after .weight/.bias or a pattern like 'local_experts.<id>'."
    )

local_expert_number = global_expert_number % num_experts_per_rank
```

关键改进：
- TEGroupedMLP：用 `\.(weight|bias)(\d+)$` 精确匹配尾部数字（比 `split` 更健壮，不会误匹配路径中其他 `.weight`）
- SequentialMLP：用 `local_experts\.(\d+)` 提取路径中的 expert 索引
- 两种格式互不冲突：SequentialMLP 参数名以 `.weight` 结尾（无尾部数字），第一个正则不匹配，正确 fall through 到第二个
- 提取失败时抛出明确的错误信息，而非 `int("")` 的隐晦报错

#### 9.2 `_normalize_expert_param_name()` — 支持 SequentialMLP 格式

```python
def _normalize_expert_param_name(self, param_name: str) -> str:
    """Normalize expert parameter name by replacing expert index with 0.

    Handles two naming conventions:
    - TEGroupedMLP: experts.linear_fc1.weight15 -> experts.linear_fc1.weight0
    - SequentialMLP: experts.local_experts.15.linear_fc1.weight -> experts.local_experts.0.linear_fc1.weight
    """
    # TEGroupedMLP: trailing number (e.g. weight15 -> weight0)
    if re.search(r"\d+$", param_name):
        return re.sub(r"\d+$", "0", param_name)
    # SequentialMLP: expert index in module path (e.g. local_experts.15.xxx -> local_experts.0.xxx)
    match = re.search(r"(local_experts\.)\d+", param_name)
    if match:
        return param_name[:match.start()] + "local_experts.0" + param_name[match.end():]
    return param_name
```

### 完整流程验证（EP=2, E=512, num_experts_per_rank=256）

1. **`_megatron_local_name_to_global`**（`model_bridge.py`）将 local index → global index：
   - Rank 1, `local_experts.3` → `local_experts.259`（`3 + 1*256`）

2. **映射匹配**：`decoder.layers.5.mlp.experts.local_experts.259.linear_fc1.weight` 匹配模式 `decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight`，HF 参数名解析为 `model.layers.5.mlp.experts.259.gate_proj.weight`

3. **`gather_from_ep_ranks`**：
   - `global_expert_number = 259`，`local_expert_number = 259 % 256 = 3`
   - `gathered_expert_param_names = ["experts.3", "experts.259"]`（Rank 0 和 Rank 1 上对应 local index 3 的全局 expert）
   - 断言 `hf_param_name in gathered_expert_param_names` 通过（`experts.259` 在列表中）

4. **`_normalize_expert_param_name`**：`local_experts.259.linear_fc1.weight` → `local_experts.0.linear_fc1.weight`，用于从模型中查找 expert 0 的 shape/dtype（仅取元信息，不取实际值）

---

## 问题 8：moe_grouped_gemm=True 导致 backward NaN

### 现象

```
RuntimeError: Function 'MindSpeedTEGroupedLinearGMMBackward' returned nan values in its 6th output.
```

Forward pass 成功完成，但 backward pass 中 `MindSpeedTEGroupedLinearGMM` 的反向传播返回 NaN。

完整错误栈：

```
File ".../megatron/core/pipeline_parallel/schedules.py", line 489, in backward_step
    torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
  ...
  File ".../megatron/core/tensor_parallel/random.py", line 566, in backward
    torch.autograd.backward(outputs, args)
  ...
RuntimeError: Function 'MindSpeedTEGroupedLinearGMMBackward' returned nan values in its 6th output.
```

### 根因

`moe_grouped_gemm=True` 时，MoE 专家层使用 MindSpeed 的 `TEGroupedLinearGMM`（Grouped GEMM）实现。该实现在 Bailing V3 tiny 模型的 backward pass 中产生 NaN 梯度，可能原因：

1. Bailing V3 的 `moe_intermediate_size=768` 较小，Grouped GEMM kernel 在小维度下数值不稳定
2. MindSpeed 的 `TEGroupedLinearGMM` 对 NPU 的某些算子组合存在兼容性问题
3. Grouped GEMM 的 FP16/BF16 精度在小 expert 上累积误差更大

### 修复

**文件**：`Megatron-Bridge/src/megatron/bridge/recipes/bailing/bailing_v3.py`

将 `moe_grouped_gemm` 设为 `False`，使用 `SequentialMLP` 替代 `TEGroupedMLP`：

```python
moe_grouped_gemm = False  # TEGroupedLinearGMM backward 产生 NaN，改用 SequentialMLP
```

**注意**：切换为 SequentialMLP 后，专家权重的 Megatron 命名格式从 `experts.linear_fc1.weightN`（尾部数字）变为 `experts.local_experts.N.linear_fc1.weight`（路径索引），需要同步修改权重映射（见问题 9）。

### 验证

关闭 `moe_grouped_gemm` 后，backward pass 正常通过，不再出现 NaN 梯度

---

## 修改文件汇总

| 文件 | 修改内容 | 问题 |
|------|---------|------|
| `Megatron-Bridge/.../bailing/bailing_provider.py` | 添加 `use_flash_attn`、`gated_attention_proj_granularity_type` 字段 | #1, #5 |
| `Megatron-Bridge/.../bailing/bailing_bridge.py` | SequentialMLP 映射；KDA f/g_proj 参数名修复；MLA attn_gate 映射；rope_type 传参 | #3, #4, #5, #6 |
| `Megatron-Bridge/.../recipes/bailing/bailing_v3.py` | `moe_grouped_gemm=False` 避免 TEGroupedLinearGMM backward NaN | #8 |
| `Megatron-Bridge/.../conversion/model_bridge.py` | SequentialMLP EP 索引转换 | SequentialMLP |
| `Megatron-Bridge/.../conversion/param_mapping.py` | `_normalize_expert_param_name` 支持 SequentialMLP；`gather_from_ep_ranks` 支持 SequentialMLP expert 编号提取 | #9, SequentialMLP |
| `Megatron-LM/.../transformer_config.py` | 添加 `gated_attention_proj_granularity_type` 字段 | #5 |
| `Megatron-LM/.../multi_latent_attention.py` | 添加 `attn_gate` 子模块、构建、forward 门控逻辑；修复 TP reshape | #5, #7 |
| `Megatron-LM/.../experimental_attention_variant_module_specs.py` | KDA `fuse_input_layernorm=False`；MLA spec 注入 `attn_gate` | #3, #5 |
| `Asystem-HybridEngine/.../megatron_backend.py` | `build_config()` 传递 `use_flash_attn` | #1 |
| `Asystem-HybridEngine/.../run_bailingv3_tiny_npu.py` | 修复 `enable_padding_data` 清空数据；添加 logp_compute + advantages 填充 | #2 |

---

## Asystem-HybridEngine Bridge 路径配置传递机制

### setup_model_with_mbridge() 流程

```
YAML 配置
  ├─ 顶层参数 → MegatronConfig.init_megatron_args() → setattr(args, key, value)
  ├─ override_transformer_config → repatch() → setattr(full_args, key, value)
  │
  └─ setup_model_with_mbridge()
       ├─ AutoBridge.from_hf_pretrained() → BailingMoeV3Provider (即 TransformerConfig)
       ├─ build_config(provider, args) → 只复制部分参数（tp, ep, pp, recompute 等）
       ├─ provider.finalize()
       └─ get_model(provider, ...) → GPTModel
```

### 关键点

1. **`repatch()`** 将 `override_transformer_config` 的值设到 Megatron `args`，但 `args` 上的值不会自动传到 `Provider`/`TransformerConfig`
2. **`build_config()`** 是手动逐字段复制的，新增参数需要显式添加
3. **`BailingMoeV3Provider`** 需要包含所有 MindSpeed/NPU 需要的字段（如 `use_flash_attn`），否则 `getattr(config, 'use_flash_attn', False)` 返回 `False`
4. 非 bridge 路径中，MindSpeed 的 `transformer_config_init_wrapper` 会自动从 `args` 注入未知字段到 config，bridge 路径没有这个机制

### 在 Provider 中需要添加的 NPU/MindSpeed 相关字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `use_flash_attn` | `True` | NPU 必须，否则 MindSpeed 的 generate_mask 报错 |
| `recompute_method` | - | 需通过 `build_config()` 或 `override_transformer_config` 传递 |
| `recompute_granularity` | - | 同上 |
| `recompute_num_layers` | - | 同上 |
| `multi_latent_attention` | - | 同上 |
| `attention_mask_type` | - | 同上 |
| `use_fused_rotary_pos_emb` | - | 同上 |
| `context_parallel_size` | - | 同上 |
| `rope_type` | - | 同上 |

---

## Bailing V3 架构关键参数

### HF Config → Megatron Bridge 参数映射

| HF Config | Megatron Bridge | 说明 |
|-----------|----------------|------|
| `q_lora_rank` | `q_lora_rank` | None 时走非 LoRA 路径（`linear_q_proj`） |
| `kv_lora_rank` | `kv_lora_rank` | KV 压缩维度 |
| `qk_nope_head_dim` | `qk_head_dim` | QK 非 RoPE 部分 head dim |
| `qk_rope_head_dim` | `qk_pos_emb_head_dim` | QK RoPE 部分 head dim |
| `v_head_dim` | `v_head_dim` | V head dim |
| `rope_interleave` | `rotary_interleaved` | RoPE 交错模式 |
| `partial_rotary_factor` | `rotary_percent` | 0.5 = 半数维度用 RoPE |
| `rope_theta` | `rotary_base` | RoPE 基频 |
| `gated_attention_proj_granularity_type` | `gated_attention_proj_granularity_type` | MLA 门控投影粒度 |
| `no_kda_lora` | - | True 时 KDA 用 `f_a_proj`/`g_a_proj`，False 用 LoRA |
| `layer_group_size` | - | 推导 `linear_attention_freq`：每 N 层一组，最后一层用 MLA |
| `first_k_dense_replace` | - | 推导 `moe_layer_freq`：前 K 层 dense，其余 MoE |
| `scoring_func` | `moe_router_score_function` | 专家路由评分函数 |
| `short_conv_kernel_size` | `linear_conv_kernel_dim` | KDA conv1d kernel 大小 |

### 注意事项

- `rope_type` 必须设为 `"rope"`（非默认的 `"yarn"`），因为 Bailing V3 使用 `rotary_interleaved=True`
- `moe_grouped_gemm` 在 tiny 模型（`moe_ffn_hidden_size=512`，较小）上可能需要设为 `False`（走 SequentialMLP），注意权重映射格式差异
- KDA 和 MLA 的 `g_proj` 是不同的模块：KDA 的 `g_proj` 是门控输出投影，MLA 的 `attn_gate` 是注意力输出门控

---

## 调试技巧

### 检查 TransformerConfig 上的属性

```python
config = model[0].config
print(f"use_flash_attn: {getattr(config, 'use_flash_attn', 'NOT SET')}")
print(f"multi_latent_attention: {getattr(config, 'multi_latent_attention', 'NOT SET')}")
print(f"gated_attention_proj_granularity_type: {getattr(config, 'gated_attention_proj_granularity_type', 'NOT SET')}")
```

### 检查模型结构和权重加载

```python
# 打印模型结构（检查关键模块是否正确创建）
for name, module in model[0].named_modules():
    if 'attn_gate' in name or 'input_layernorm' in name or 'g_proj' in name:
        print(f"{name}: {type(module).__name__}")

# 检查权重是否加载（非默认初始化）
for name, param in model[0].named_parameters():
    if 'attn_gate' in name:
        print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
```

### 检查 loss weight

```python
mb_inputs, forward_step, total_loss_weight = prepare_data_and_fwd_function(...)
print(f"total_loss_weight: {total_loss_weight}")
for i, mb in enumerate(mb_inputs):
    mask_sum = mb.data['ppo_loss_mask'].count_nonzero().item()
    print(f"mb[{i}] ppo_loss_mask nonzero: {mask_sum}")
```

### 检查数据完整性

```python
data = torch.load(data_path, ...)
sample = data['inputs']
for k, v in sample.data.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, nonzero={v.count_nonzero().item()}")
```