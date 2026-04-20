# 权重交换方案NPU适配总结

## 一、Disk方案（FileWeightExchangeReader/FileWeightExchangeWriter）

### NPU适配情况：✅ 无需适配

### 原因：

**1. 纯文件I/O操作**：该方案仅涉及磁盘读写操作
   - 训练侧：`save_hf_checkpoint(path)` 保存权重到磁盘
   - 推理侧：`update_weights_from_disk(path, load_format)` 从磁盘加载权重

**2. 无硬件相关代码**：
```python
# weights_reader.py
class FileWeightExchangeReader(WeightExchangeReader):
    def update_weights(self, step_id, **kwargs):
        self.inference_backend.update_weights_from_disk(
            kwargs["path"], kwargs.get("load_format")
        )

# weights_writer.py
class FileWeightExchangeWriter(WeightExchangeWriter):
    def write_weights(self, **kwargs):
        self.train_backend.save_hf_checkpoint(kwargs["path"])
```

**3. 依赖底层框架**：实际的磁盘读写由推理引擎（vLLM/SGLang）和训练框架实现，这些框架已经适配了NPU

### 结论
Disk方案完全不需要NPU适配，可直接使用。

---

## 二、AllGather-HF方案

### NPU适配情况：✅ 已完成适配

### 适配点：

#### 1. AllGather通信操作
- **位置**：`convert/mcore/base.py:get_full_tensor()`
- **代码**：
```python
def get_full_tensor(weight: torch.Tensor, dim: int = 0):
    train_tp_size = mpu.get_tensor_model_parallel_world_size()
    if train_tp_size != 1:
        tp_group = mpu.get_tensor_model_parallel_group()
        new_v = [torch.zeros_like(weight) for i in range(train_tp_size)]
        dist.all_gather(new_v, weight, group=tp_group, async_op=False)
        weight = torch.cat(new_v, dim=dim)
    return weight
```

- **适配说明**：
  - `dist.all_gather` 是PyTorch分布式通信的标准API
  - NPU通过 `torch_npu` 提供了完整的分布式通信支持
  - HCCL（Huawei Collective Communication Library）实现了与NCCL相同的集合通信原语
  - **无需修改代码**，PyTorch会自动选择正确的通信后端（NCCL或HCCL）

#### 2. AllGather对象通信
- **位置**：`meta_resolver.py`
- **代码**：
```python
global_metadata: List[Dict[str, Any]] = [None] * dist.get_world_size()
dist.all_gather_object(global_metadata, meta)
```

- **适配说明**：
  - `all_gather_object` 用于收集元数据对象
  - NPU完全支持该操作
  - **无需修改代码**

#### 3. 权重转换中的AllGather使用
- **位置**：`convert/mcore/bailing_moe_linear.py`
- **用途**：在MoE模型中，对expert权重进行AllGather以获取完整权重
- **适配说明**：已通过 `get_full_tensor()` 函数统一处理，NPU无需额外适配

### 结论
AllGather-HF方案已完成NPU适配，无需额外修改。

---

## 三、P2P方案

### NPU适配情况：⚠️ 需要适配，已完成

### 适配点详解：

#### 1. 通信后端初始化 ✅ 已适配

**位置**：`util.py:init_weights_update_group()`

**适配代码**：
```python
from transformers.utils.import_utils import is_torch_npu_available

def init_weights_update_group(...):
    options = None
    if is_torch_npu_available():
        import torch_npu
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {
            "hccl_buffer_size": int(os.getenv("GROUP_HCCL_BUFFER_SIZE", "600"))
        }
    
    group = init_custom_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=rank,
        group_name=group_name,
        pg_options=options  # NPU专用配置
    )
    return group
```

**适配要点**：
- 检测NPU环境：`is_torch_npu_available()`
- 使用HCCL替代NCCL：`ProcessGroupHCCL.Options()`
- 配置HCCL buffer size（默认600MB，可通过环境变量调整）

---

#### 2. P2P通信操作 ✅ 已适配

**位置**：`nccl_comm.py`, `nccl_stream_batch.py`

**核心代码**：
```python
# 使用PyTorch标准P2P API
dist.P2POp(dist.isend, tensor, peer, group=group)
dist.P2POp(dist.irecv, tensor, peer, group=group)

# 或使用同步版本
dist.send(tensor, peer, group=group)
dist.recv(tensor, peer, group=group)
```

**适配说明**：
- PyTorch的P2P通信API在NPU上由HCCL实现
- `dist.isend/irecv/send/recv` 在NPU环境下自动使用HCCL
- **无需修改代码**，PyTorch自动选择正确的后端

---

#### 3. CUDA Stream适配 ⚠️ 需要注意

**位置**：`nccl_comm.py:_get_comm_streams()`

**代码**：
```python
_COMM_STREAMS_PER_DEVICE: Dict[int, List[torch.cuda.Stream]] = {}

def _get_comm_streams() -> List[torch.cuda.Stream]:
    if not torch.cuda.is_available():
        return []
    
    device_index = torch.cuda.current_device()
    streams = _COMM_STREAMS_PER_DEVICE.get(device_index)
    if streams is None:
        streams = [
            torch.cuda.Stream(device=device_index) 
            for _ in range(NUM_COMM_STREAMS)
        ]
        _COMM_STREAMS_PER_DEVICE[device_index] = streams
    return streams
```

**适配说明**：
- NPU使用 `torch_npu.npu.Stream` 而非 `torch.cuda.Stream`
- **当前代码使用 `torch.cuda.Stream`，在NPU上可能需要适配**
- **建议修改**：
```python
if is_torch_npu_available():
    import torch_npu
    streams = [torch_npu.npu.Stream(device=device_index) 
               for _ in range(NUM_COMM_STREAMS)]
else:
    streams = [torch.cuda.Stream(device=device_index) 
               for _ in range(NUM_COMM_STREAMS)]
```

---

#### 4. 权重转换适配 ✅ 已适配

**位置**：`convert/mcore/converter.py`, `convert/vllm/converter.py`

**关键适配点**：

**a) Expert权重转置**
```python
# mcore/converter.py
def _convert_expert_weight_param(self, name, parameter, layer_number):
    return [
        (f"mlp.experts.{expert_id}.{name}", 
         param.t() if is_torch_npu_available() else param)
        for name, param in self._convert_linear(name, parameter)
    ]
```

**原因**：NPU的矩阵乘法布局与GPU不同，需要转置expert权重

**b) Gate-Up Projection融合**
```python
def _fuse_gate_up_proj(self, name: str) -> bool:
    return True if is_torch_npu_available() else False
```

**原因**：NPU环境下启用gate_up projection融合以优化性能

**c) LayerNorm权重处理**
```python
if is_torch_npu_available() and "input_layernorm.weight" in name:
    name = name.replace("decoder.", "model.")
    return [(name, parameter)]
```

**原因**：修正NPU环境下的命名映射

---

#### 5. 并行配置适配 ✅ 已适配

**位置**：`weights_reader.py`

**代码**：
```python
if is_torch_npu_available():
    self.dp_size = 1
    if self.inference_backend.config.ep_size > 1:
        self.dp_size = self.inference_backend.config.ep_size // self.tp_size // self.pp_size
    self.infer_world_size = self.num_engines * self.tp_size * self.pp_size * self.dp_size
```

**适配说明**：
- NPU环境下支持EP（Expert Parallelism）
- 调整world_size计算以包含DP维度

---

#### 6. 环境变量配置 ✅ 已适配

**位置**：`weights_writer.py`

**代码**：
```python
self.init_every_step = os.getenv("WEIGHT_EXCHANGE_GROUP_INIT_EVERY", "1") == "1" 
                        if is_torch_npu_available() else False
self.batch_flag = os.getenv("WEIGHT_EXCHANGE_GROUP_BATCH_FLAG", "0") == "1" 
                   if is_torch_npu_available() else True
```

**适配说明**：
- NPU环境下默认每步重新初始化通信组（`init_every_step=True`）
- NPU环境下默认使用批量模式（`batch_flag=True`）

---

## 四、总结对比表

| 方案 | NPU适配状态 | 主要适配点 | 是否需要额外工作 |
|------|------------|-----------|----------------|
| **Disk** | ✅ 无需适配 | 无 | 否 |
| **AllGather-HF** | ✅ 已完成适配 | 无（PyTorch自动处理） | 否 |
| **P2P** | ⚠️ 已完成大部分适配 | 1. HCCL初始化<br>2. 权重转置<br>3. Gate-Up融合<br>4. LayerNorm命名<br>5. EP并行配置<br>6. Stream适配（需注意） | 是（Stream可能需要微调） |

---

## 五、PPT内容建议

### 标题：权重交换方案NPU适配总结

#### 1. Disk方案
- ✅ 无需适配
- 纯文件I/O操作，与硬件无关
- 依赖底层框架的NPU支持

#### 2. AllGather-HF方案
- ✅ 已完成适配
- 使用PyTorch标准分布式API
- HCCL自动替代NCCL，无需代码修改

#### 3. P2P方案
- ⚠️ 已完成核心适配
- **关键适配点**：
  - HCCL通信组初始化（buffer size配置）
  - Expert权重转置（NPU矩阵布局）
  - Gate-Up Projection融合（性能优化）
  - LayerNorm命名修正
  - EP并行配置支持
  - CUDA Stream → NPU Stream（需验证）

#### 4. 技术亮点
- 自动检测NPU环境
- 条件执行适配逻辑
- 保持GPU/NPU代码兼容性
- 可配置的HCCL参数

#### 5. 建议
- 验证NPU Stream在多流并发场景下的表现
- 测试HCCL buffer size对性能的影响
- 监控P2P通信在NPU上的稳定性

---

## 六、代码文件结构

```
weights_exchange/
├── nccl_comm.py          # P2P通信实现
├── nccl_stream_batch.py  # 流式批量传输
├── util.py               # HCCL初始化
├── transfer_plan.py      # 传输计划生成
├── weights_reader.py     # 权重读取（推理侧）
├── weights_writer.py     # 权重写入（训练侧）
└── convert/              # 权重格式转换
    ├── mcore/            # Megatron-Core转换
    ├── vllm/             # vLLM转换
    └── sglang/           # SGLang转换
```

---

## 七、关键技术细节

### 1. HCCL通信组初始化
- 使用 `torch_npu._C._distributed_c10d.ProcessGroupHCCL` 替代NCCL
- 配置HCCL buffer size（默认600MB，可通过环境变量`GROUP_HCCL_BUFFER_SIZE`调整）
- 代码位置：`util.py:init_weights_update_group()`

### 2. 多流并发传输
- 创建64个CUDA Stream池用于并发P2P通信
- 每个peer rank分配固定stream，保证同peer操作顺序性
- 不同peer使用不同stream实现并发执行

### 3. 递归分区传输算法
- 时间复杂度：O(log N)轮次
- 每轮将partition分成两半，交替发送/接收
- 避免通信死锁，提高传输效率
- 代码位置：`nccl_stream_batch.py`

### 4. 两阶段P2P通信
- Phase 1: 前半部分发送，后半部分接收
- Phase 2: 前半部分接收，后半部分发送
- 全局barrier确保同步

---

## 八、环境变量配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `GROUP_HCCL_BUFFER_SIZE` | 600 | HCCL通信buffer大小（MB） |
| `WEIGHT_EXCHANGE_GROUP_INIT_EVERY` | 1 (NPU) / 0 (GPU) | 是否每步重新初始化通信组 |
| `WEIGHT_EXCHANGE_GROUP_BATCH_FLAG` | 0 (NPU) / 1 (GPU) | 是否使用批量模式 |

---

## 九、测试建议

### 1. 功能测试
- 验证Disk方案在NPU上的正确性
- 验证AllGather-HF方案在NPU上的正确性
- 验证P2P方案在NPU上的正确性

### 2. 性能测试
- 测试不同HCCL buffer size对性能的影响
- 测试多流并发在NPU上的性能表现
- 对比GPU和NPU的权重交换性能

### 3. 稳定性测试
- 长时间运行测试
- 大规模集群测试
- 异常情况恢复测试

---

## 十、参考资料

- PyTorch分布式通信文档
- HCCL开发指南
- torch_npu API文档
- Megatron-Core文档
- vLLM-Ascend文档
