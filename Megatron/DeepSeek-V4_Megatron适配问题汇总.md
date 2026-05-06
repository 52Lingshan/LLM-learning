# DeepSeek V4 Megatron-LM + Megatron-Bridge NPU 适配问题汇总

> 日期: 2026-04-28
> 环境: 华为昇腾 910 NPU (16卡单机), MindSpeed, Megatron-LM + Megatron-Bridge
> 模型: DeepSeek-V4-Flash-Base (291B MoE, 缩减为4层测试)
> 并行策略: TP=2, PP=1, DP=8, EP=16

---

## Phase 1: HF Config 加载问题

### 1. `quantization_config.to_dict()` 报 NoneType 错误

**现象**: transformers 的 `to_diff_dict()` 在创建默认实例时调用 `quantization_config.to_dict()`，但 `quantization_config` 为 None。

**原因**: `DeepseekV4Config.__init__` 中将 `quantization_config=None` 直接赋值为实例属性，transformers 基类序列化时尝试调用 `.to_dict()`。

**解决**: 在 `configuration_deepseek_v4.py` 中，仅当 `quantization_config is not None` 时才通过 `kwargs` 传给基类，不设为实例属性。

```python
def __init__(self, ..., quantization_config=None, **kwargs):
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    # 不要 self.quantization_config = quantization_config
```

**文件**: `Megatron-Bridge/examples/decentralized_pg/configuration_deepseek_v4.py`

---

### 2. `DeepseekV4ForCausalLM` 在 transformers 中找不到

**现象**: Bridge 注册的 source 类名是 `"DeepseekV4ForCausalLM"`，但远程环境的 transformers 版本太旧，没有原生 V4 支持。

**解决**: 在远程模型目录的 `config.json` 中添加 `auto_map` 字段，让 transformers 通过 `trust_remote_code` 自动加载：

```json
"auto_map": {
    "AutoConfig": "configuration_deepseek_v4.DeepseekV4Config",
    "AutoModelForCausalLM": "modeling_deepseek_v4.DeepseekV4ForCausalLM"
}
```

同时在模型目录放置 `configuration_deepseek_v4.py` 文件。

**文件**: 远程 `config.json`, `configuration_deepseek_v4.py`

---

### 3. `trust_remote_code` 未传递到第二次 config 加载

**现象**: `AutoBridge.from_hf_pretrained()` 内部通过 `_ConfigOnlyPretrainedShim` 第二次读取 config，但 `trust_remote_code` 默认 False，导致 auto_map 不生效。

**解决**: 显式传入 `trust_remote_code=True`：

```python
cfg.model = AutoBridge.from_hf_pretrained(
    args.hf_path, trust_remote_code=True
).to_megatron_provider(load_weights=args.load_weights)
```

**文件**: `pretrain_deepseek_v4.py`

---

### 4. 内联 `_DeepseekV4Config` 空类覆盖了 auto_map 注册

**现象**: pretrain 脚本中有一个空的 `_DeepseekV4Config` 类通过 `AutoConfig.register` 注册，优先级高于 auto_map，导致加载的 config 缺少 V4 字段。

**解决**: 删除 pretrain 脚本中的内联 config 类注册 hack，完全依赖 `trust_remote_code` + auto_map。

**文件**: `pretrain_deepseek_v4.py`

---

## Phase 2: V3 Bridge 继承问题

### 5. `first_k_dense_replace` 为 None 导致乘法 TypeError

**现象**: V4 Bridge 继承 V3 Bridge，V3 的 `provider_bridge()` 中有 `first_k_dense_replace * ...` 乘法运算，但 V4 config 中该字段为 None。

**解决**: 在 `DeepseekV4Config` 中将 `first_k_dense_replace` 默认值设为 `0`（而非 None）。

```python
first_k_dense_replace=0,  # V4 不使用，但 V3 bridge 需要 int
```

**文件**: `configuration_deepseek_v4.py`

---

### 6. `v_head_dim` 和 `kv_lora_rank` 为 None

**现象**: V3 bridge 中对这些值做乘法运算，但 V4 config 中可能未显式设置。

**解决**: 在 `DeepseekV4Config` 中确保默认值为 512：

```python
kv_lora_rank=512,
v_head_dim=512,
```

**文件**: `configuration_deepseek_v4.py`

---

### 7. `task.megatron_module` 在 None task 上调用

**现象**: `build_conversion_tasks()` 返回的列表中有 None 条目（V4 不需要 V3 的某些权重映射），但后续代码直接访问 `.megatron_module`。

**解决**: 在 `model_bridge.py` 中添加 None 检查：

```python
if task is None or task.megatron_module is None:
    continue
```

**文件**: `Megatron-Bridge/src/megatron/bridge/models/conversion/model_bridge.py` (~行 782, 880)

---

## Phase 3: NPU 环境适配

### 8. `use_precision_aware_optimizer` 需要 TE FusedAdam

**现象**: NPU 没有 NVIDIA Transformer Engine，FusedAdam 不可用。

**解决**: 禁用精度感知优化器：

```python
cfg.optimizer.use_precision_aware_optimizer = False
```

**文件**: `pretrain_deepseek_v4.py`

---

### 9. NullTokenizer 不接受 `trust_remote_code` kwarg

**现象**: Bridge 为所有 tokenizer 传入 `trust_remote_code`，但 NullTokenizer 的 `__init__` 不接受额外参数。

**解决**:
1. 改用 HuggingFace tokenizer（模型目录自带 tokenizer）：
```python
cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
cfg.tokenizer.tokenizer_model = args.hf_path
```
2. 同时给 NullTokenizer 加 `**kwargs` 防止其他场景报错。

**文件**: `pretrain_deepseek_v4.py`, `null_tokenizer.py`

---

## Phase 4: DSV4 Attention 模块适配

### 10. 模型使用 MLASelfAttention 而非 DSV4Attention

**现象**: V3 bridge 设置标准 MLA spec，V4 bridge 继承后没有覆盖 `transformer_layer_spec`，导致实例化的是 MLA 而非 DSV4 attention，shape 不匹配。

**解决**: 在 V4 bridge 的 `provider_bridge()` 中显式设置 experimental attention variant spec：

```python
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
```

**文件**: `deepseek_v4_bridge.py`

---

### 11. `rotary_scaling_factor` assert 失败 (40 不在 {4, 16})

**现象**: Miles 遗留代码中 `rope.py` 有硬编码的 assert，只允许 `rotary_scaling_factor` 为 4 或 16，但 V4 使用 40。

**解决**: 删除 Miles 遗留的硬编码 assert。

**文件**: `Megatron-LM/megatron/core/transformer/experimental_attention_variant/ops/rope.py`

---

### 12. `attn_sink.dtype` assert 失败 (bf16 != float32)

**现象**: `attn_sink` 声明为 `torch.float32`，但混合精度训练自动转为 bf16，assert 失败。

**解决**: 移除 dtype assert，在使用时显式转为 float32：

```python
# 改前
assert self.attn_sink.dtype == torch.float32
# 改后
attn_sink = self.attn_sink.float()
# 后续所有引用 self.attn_sink 改为 attn_sink
```

**文件**: `dsv4.py`

---

### 13. `all_reduce_grad_fp32` 参数不存在

**现象**: Miles 自定义的 `copy_to_tensor_model_parallel_region()` 扩展参数，Megatron-LM 原版不支持。

**解决**: 删除 `all_reduce_grad_fp32=True` 参数调用。

**文件**: `dsv4.py`

---

### 14. DSV4 forward 返回值格式不匹配

**现象**: `transformer_layer.py` 的 `bias_dropout_add` 期望 `(output, bias)` 元组，但 DSV4 forward 只返回单个 tensor。报错 `ValueError: too many values to unpack (expected 2)`。

**解决**: 修改返回值：

```python
# 改前
return output
# 改后
return output, None  # DSV4 没有 bias
```

**文件**: `dsv4.py` (行 309)

**注意**: 修改后需要清理远程 `__pycache__` 目录，否则旧的 .pyc 缓存可能导致改动不生效。

---

## Phase 5: Loss 计算 & 训练循环

### 15. TE cross entropy 不可用

**现象**: `cross_entropy_fusion_impl` 默认 `"te"`，但 NPU 没有 Transformer Engine 的 cross entropy 实现。

**解决**: 改用 native 实现：

```python
cfg.model.cross_entropy_fusion_impl = "native"
```

**文件**: `pretrain_deepseek_v4.py`

---

### 16. model 和 dataset 的 seq_length 不一致

**现象**: 修改 model seq_length 后，dataset seq_length 仍取默认值，validation 报 assert 错误。

**解决**: 同步设置 dataset seq_length：

```python
cfg.model.seq_length = args.seq_length
cfg.dataset.seq_length = args.seq_length  # 必须同步
```

**文件**: `pretrain_deepseek_v4.py`

---

### 17. NPU OOM (显存不足)

**现象**: 4层 DSV4 (256 experts, head_dim=512) 在 61GB NPU 上 OOM。

**解决** (组合措施):
1. 开启 full activation recompute：
```python
cfg.model.recompute_granularity = "full"
cfg.model.recompute_method = "uniform"
cfg.model.recompute_num_layers = cfg.model.num_layers
```
2. 减小 seq_length（4096 → 1024）

**文件**: `pretrain_deepseek_v4.py`

---

### 18. NaN loss / NaN grad 导致训练中断

**现象**: 第一次 iteration 所有 rank 报 NaN loss 和 NaN grad，`rerun_state_machine` 和 DDP grad check 抛出异常。

**原因**: Mock 数据 + 移植后的数值精度问题（待深入排查）。

**临时解决** (先让训练循环跑通):
```python
cfg.rerun_state_machine.check_for_nan_in_loss = False
cfg.ddp.check_for_nan_in_grad = False
```

**后续排查方向**:
- DSV4 `dense_attn_torch` 对全 -inf score 行的处理
- `attn_sink` 权重是否正确加载（TP 分片对齐）
- compressor 模块权重映射是否完整
- 使用真实数据替代 mock data 后是否还有 NaN

**文件**: `pretrain_deepseek_v4.py`

---

## 附录: 完整代码修改清单

### A. Megatron-LM 修改点

#### A1. 新增文件: DSV4 Attention 主模块

**文件**: `megatron/core/transformer/experimental_attention_variant/dsv4.py` (326行)

核心类 `DeepSeekV4Attention(MegatronModule)`，完整的 DSV4 注意力实现：
- Q 投影链: `wq_a` → `q_norm` → `wq_b` → unflatten → RMSNorm → RoPE
- KV 投影链: `wkv` → `kv_norm` → RoPE → (可选 FP8 QAT)
- 稀疏索引: 滑动窗口 topk + 压缩 topk（compress_ratio > 0 时）
- 注意力计算: 3种后端（dense/sparse/tilelang，由 `MEGATRON_SPARSE_ATTN_IMPL` 环境变量切换）
- 输出投影: 分组 LoRA (`wo_a` + `wo_b`)，`n_groups=8`, `o_lora_rank=1024`
- `attn_sink`: 每头注意力沉参数（fp32），参与 softmax 分母计算
- `sharded_state_dict()`: 支持 TP 分片 checkpoint

```python
class DeepSeekV4Attention(MegatronModule):
    def __init__(self, config, layer_number=1, ...):
        # Q: dim -> q_lora_rank -> n_heads * head_dim
        self.wq_a = TELinear(dim, q_lora_rank, parallel_mode="duplicated")
        self.q_norm = TENorm(q_lora_rank)
        self.wq_b = ColumnParallelLinear(q_lora_rank, n_heads * head_dim)
        
        # KV: dim -> head_dim (共享所有头)
        self.wkv = TELinear(dim, head_dim, parallel_mode="duplicated")
        self.kv_norm = TENorm(head_dim)
        
        # Output: 分组 LoRA
        self.wo_a = ColumnParallelLinear(n_heads * head_dim // n_groups, n_groups * o_lora_rank)
        self.wo_b = RowParallelLinear(n_groups * o_lora_rank, dim)
        
        # 压缩器和索引器（按层配置）
        if compress_ratio:
            self.compressor = DeepSeekV4Compressor(...)
            if compress_ratio == 4:
                self.indexer = V4Indexer(...)

    def forward(self, hidden_states, ...):
        # ... 完整前向逻辑
        return output, None  # Megatron 接口要求 (output, bias) 元组

def _dsv4_attention_module_spec(config, backend=None):
    return ModuleSpec(module=DeepSeekV4Attention, submodules=None,
                      metainfo={"fuse_input_layernorm": False})
```

---

#### A2. 新增文件: ops/ 子目录 (9个文件)

**`ops/__init__.py`** (1行): 空初始化

**`ops/rope.py`** (81行): RoPE 实现
- `precompute_freqs_cis()`: 带 YaRN 平滑的旋转频率预计算
- `apply_rotary_emb()`: 原地旋转嵌入（支持 inverse 参数）
- `wrapped_precompute_freqs_cis()`: 从 TransformerConfig 读取参数的包装

**`ops/cp_utils.py`** (130行): 上下文并行工具
- `all_gather_cp()`: CP 组内 all-gather
- `get_compress_topk_idxs_cp()`: 压缩 topk 索引（CP 感知）
- `get_freqs_cis_for_cp()`: CP 分片的 RoPE 频率
- `get_q_positions_for_cp()`: CP 分片的 query 位置
- `get_window_topk_idxs_cp()`: 滑动窗口 topk 索引（CP 感知）

**`ops/attention_core.py`** (143行): 注意力计算核心
- `sparse_attn_torch()`: 纯 PyTorch 稀疏注意力（gather 实现）
- `dense_attn_torch()`: 纯 PyTorch 稠密注意力（mask 实现）
- `sparse_attn_tilelang()`: TileLang 内核（autograd.Function 包装）
- 共同特性: 支持 `attn_sink` 参与 softmax 分母、`topk_idxs=-1` 表示无效位置

**`ops/compressor.py`** (185行): KV 压缩器
- `DeepSeekV4Compressor`: 按 `compress_ratio` 压缩 KV 序列
  - ratio=4 (CSA): 滑动窗口重叠变换 + softmax 池化
  - ratio=128 (HCA): 高度压缩 + softmax 池化
  - FP32 精度参数: `ape`, `wkv`, `wgate`, `norm`
  - 支持 CP 并行的重叠变换

**`ops/hyper_connection.py`** (167行): 超连接路由
- `DeepSeekV4HyperConnectionUtil`: 块内多流路由
  - `hc_pre_raw()`: Sinkhorn 求解最优传输 → 加权汇聚
  - `hc_post_raw()`: 流输出重组
  - `block_expand()` / `block_head()`: 块边界处理
- `HCHeadParams(MegatronModule)`: 全局 HC 头参数（fp32）
  - `hc_head_fn`, `hc_head_base`, `hc_head_scale`

**`ops/v4_indexer.py`** (135行): DSA V4 索引器
- `V4Indexer(MegatronModule)`: 仅用于 compress_ratio=4 的 C4 层
  - Q 投影 → 压缩 KV → TileLang 相似度计算 → topk 索引
  - 参数: `index_n_heads=64`, `index_head_dim=128`, `index_topk=512`
  - 从 Miles 移植时移除了 `indexer_replay_manager` 依赖（替换为 no-op）

**`ops/qat.py`** (31行): FP8 量化感知训练模拟
- `fp8_simulate_qat()`: 模拟 FP8 量化误差

**`ops/utils.py`** (15行): 工具函数
- `rotate_activation()`: Hadamard 旋转（依赖可选的 `fast_hadamard_transform`）

---

#### A3. 新增文件: ops/kernel/ 子目录 (7个文件, ~1180行)

所有内核文件依赖 `tilelang`（可选），作为高性能后端：

| 文件 | 行数 | 功能 |
|------|------|------|
| `tilelang_indexer_fwd.py` | 194 | 索引器前向 TileLang kernel |
| `tilelang_indexer_bwd.py` | 237 | 索引器反向 TileLang kernel |
| `tilelang_indexer.py` | 90 | 索引器 autograd 包装 |
| `tilelang_sparse_mla_fwd.py` | 189 | 稀疏 MLA 前向 kernel |
| `tilelang_sparse_mla_bwd.py` | 282 | 稀疏 MLA 反向 kernel |
| `sinkhorn.py` | 95 | HC Sinkhorn 最优传输 kernel |
| `act_quant.py` | 94 | FP8 激活量化 kernel |

---

#### A4. 修改文件: `transformer_config.py`

**修改 1**: 扩展 `experimental_attention_variant` 类型（~行 245）
```python
# 改前
experimental_attention_variant: Optional[Literal['gated_delta_net', 'dsa']] = None
# 改后
experimental_attention_variant: Optional[Literal['gated_delta_net', 'dsa', 'dsv4']] = None
```

**修改 2**: 新增 DSV4 配置字段（~行 270 后）
```python
dsv4_hc_mult: int = 4                          # HC 流数
dsv4_hc_sinkhorn_iters: int = 20               # Sinkhorn 迭代次数
dsv4_hc_eps: float = 1e-6                      # HC 数值稳定 epsilon
dsv4_compress_ratios: Optional[list] = None    # 每层压缩比 [0,4,128,...]
dsv4_compress_rope_theta: float = 160000.0     # 压缩层 RoPE 基
dsv4_swiglu_limit: float = 0.0                 # SwiGLU 激活钳制
dsv4_o_groups: int = 8                         # 输出投影分组数
dsv4_o_lora_rank: int = 1024                   # 输出 LoRA 秩
dsv4_n_hash_layers: int = 3                    # hash 层数
dsv4_window_size: int = 128                    # 滑动窗口大小
```

---

#### A5. 修改文件: `experimental_attention_variant_module_specs.py`

**修改 1**: 导入 DSV4
```python
from megatron.core.transformer.experimental_attention_variant.dsv4 import (
    DeepSeekV4Attention, _dsv4_attention_module_spec,
)
```

**修改 2**: 注册 DSV4 分支（~行 144）
```python
elif config.experimental_attention_variant == "dsv4":
    return _dsv4_attention_module_spec(config, backend=backend)
```

---

#### A6. 修改文件: `null_tokenizer.py`

`__init__` 添加 `**kwargs` 接受额外参数：
```python
def __init__(self, vocab_size, **kwargs):  # 新增 **kwargs
```

---

### B. Megatron-Bridge 修改点

#### B1. 新增文件: `deepseek_v4_bridge.py`

核心类 `DeepSeekV4Bridge(DeepSeekV3Bridge)`，注册为 Bridge：

```python
@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class DeepSeekV4Bridge(DeepSeekV3Bridge):
    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        # 覆盖 V3 的标准 MLA spec → DSV4 experimental attention variant
        provider.experimental_attention_variant = "dsv4"
        provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
        # DSA 索引器、HC、KV 压缩、输出投影等参数映射...
        return provider

    def mapping_registry(self):
        # 过滤掉 V3 独有的 Q/KV 投影映射
        # 添加 V4 attention + compressor + indexer + HC 映射
        return MegatronMappingRegistry(*(common + v4_attention + v4_global))
```

**V3 被替换的参数** (`_V3_LAYERNORM_MEGATRON_PARAMS`):
- `linear_q_down_proj`, `linear_q_up_proj` (及其 layer_norm_weight)
- `linear_kv_down_proj`, `linear_kv_up_proj` (及其 layer_norm_weight)
- `q_layernorm`, `kv_layernorm`
- `linear_proj`

---

#### B2. 修改文件: `common.py` — 新增 V4 参数映射

**`get_v4_attention_mapping_list()`**: V4 注意力层参数 (HF ↔ Megatron)

| Megatron 参数 | HF 参数 | 说明 |
|--------------|---------|------|
| `self_attention.wq_a.weight` | `self_attn.wq_a.weight` | Q 低秩投影 A |
| `self_attention.q_norm.weight` | `self_attn.q_norm.weight` | Q 归一化 |
| `self_attention.wq_b.weight` | `self_attn.wq_b.weight` | Q 低秩投影 B |
| `self_attention.wkv.weight` | `self_attn.wkv.weight` | KV 投影 |
| `self_attention.kv_norm.weight` | `self_attn.kv_norm.weight` | KV 归一化 |
| `self_attention.wo_a.weight` | `self_attn.wo_a.weight` | 输出投影 A |
| `self_attention.wo_b.weight` | `self_attn.wo_b.weight` | 输出投影 B |
| `self_attention.attn_sink` | `self_attn.attn_sink` | 注意力沉 |
| `self_attention.compressor.ape` | `self_attn.compressor.ape` | 压缩器位置编码 |
| `self_attention.compressor.wkv.weight` | `self_attn.compressor.wkv.weight` | 压缩器 KV |
| `self_attention.compressor.wgate.weight` | `self_attn.compressor.wgate.weight` | 压缩器门控 |
| `self_attention.compressor.norm.weight` | `self_attn.compressor.norm.weight` | 压缩器归一化 |
| `self_attention.indexer.linear_wq_b.weight` | `self_attn.indexer.wq_b.weight` | 索引器 Q 投影 |
| `self_attention.indexer.linear_weights_proj.weight` | `self_attn.indexer.weights_proj.weight` | 索引器权重投影 |
| `self_attention.indexer.compressor.*` | `self_attn.indexer.compressor.*` | 索引器内压缩器 |
| `*.hc_attn_fn/base/scale` | `*.hc_attn_fn/base/scale` | 层级 HC 注意力参数 |
| `*.hc_ffn_fn/base/scale` | `*.hc_ffn_fn/base/scale` | 层级 HC FFN 参数 |
| `mlp.router.tid2eid` | `mlp.topk.tid2eid` | MoE 路由映射 |

**`get_v4_global_mapping_list()`**: 全局 HC 头参数

| Megatron 参数 | HF 参数 |
|--------------|---------|
| `decoder.hc_head_params.hc_head_fn` | `model.hc_head_fn` |
| `decoder.hc_head_params.hc_head_base` | `model.hc_head_base` |
| `decoder.hc_head_params.hc_head_scale` | `model.hc_head_scale` |

---

#### B3. 修改文件: `model_bridge.py`

在权重加载循环中添加 None task 防护（~行 782, 880）：
```python
if task is None or task.megatron_module is None:
    continue
```

---

#### B4. 新增文件: `configuration_deepseek_v4.py`

`DeepseekV4Config(PretrainedConfig)` — 兼容旧版 transformers 的 V4 配置类：

关键设计：
- 支持 `compress_ratios` → `layer_types` 自动转换
- 3种层类型: `sliding_attention`(ratio=0), `compressed_sparse_attention`(ratio=4), `heavily_compressed_attention`(ratio=128)
- `quantization_config` 特殊处理（仅 not None 时传给基类）
- V3 兼容字段: `first_k_dense_replace=0`, `v_head_dim=512`, `kv_lora_rank=512`

---

#### B5. 新增/修改文件: `pretrain_deepseek_v4.py`

4层缩减 DSV4 预训练脚本，主要配置项：

```python
# 模型加载
cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True)
    .to_megatron_provider(load_weights=True)

# Tokenizer
cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
cfg.tokenizer.tokenizer_model = args.hf_path

# 并行 (16 NPU)
cfg.model.tensor_model_parallel_size = 2
cfg.model.expert_model_parallel_size = 16
cfg.model.sequence_parallel = True

# NPU 适配
cfg.optimizer.use_precision_aware_optimizer = False       # 无 TE FusedAdam
cfg.model.cross_entropy_loss_fusion = True
cfg.model.cross_entropy_fusion_impl = "native"            # 无 TE cross entropy

# 显存优化
cfg.model.recompute_granularity = "full"
cfg.model.recompute_method = "uniform"
cfg.model.recompute_num_layers = cfg.model.num_layers
# seq_length = 1024 (从 4096 缩减)

# 同步 dataset seq_length
cfg.dataset.seq_length = args.seq_length

# 调试: 禁用 NaN 检查
cfg.rerun_state_machine.check_for_nan_in_loss = False
cfg.ddp.check_for_nan_in_grad = False
```

---

### C. 代码量统计

| 仓库 | 新增文件 | 修改文件 | 新增代码行 |
|------|---------|---------|-----------|
| **Megatron-LM** | 19 (dsv4.py + ops/9 + kernel/8 + __init__) | 3 (transformer_config, module_specs, null_tokenizer) | ~2500 |
| **Megatron-Bridge** | 2 (v4_bridge, configuration) | 3 (common, model_bridge, pretrain) | ~700 |
| **合计** | 21 | 6 | **~3200** |

---

## 关键经验总结

1. **V4 继承 V3 Bridge 的陷阱**: V3 设置的 spec/config 会被 V4 继承，必须显式覆盖不适用的部分（如 `transformer_layer_spec`）。

2. **Miles 遗留代码清理**: Miles 框架的自定义扩展（如 `all_reduce_grad_fp32`、硬编码 assert）需要逐一移除。

3. **NPU 无 Transformer Engine**: 所有依赖 TE 的功能（FusedAdam、TE cross entropy、precision aware optimizer）都需要替换或禁用。

4. **Megatron attention 接口约定**: forward 必须返回 `(output, bias)` 元组，bias 为 None 也要显式返回。

5. **远程 .pyc 缓存**: 修改 Python 文件后务必清理远程 `__pycache__`，否则旧字节码可能仍被使用。

6. **config.json auto_map**: 旧版 transformers 不支持新模型类型时，通过 `auto_map` + `trust_remote_code=True` 解决。

7. **显存优化组合**: 对大 MoE 模型，需要同时使用 full recompute + 缩短 seq_length + EP 并行才能装下。
