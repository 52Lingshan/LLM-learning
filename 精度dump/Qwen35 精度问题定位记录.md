## 问题背景
在qwen35精度对齐过程中，使用Asys单测用例作为实验启动脚本，使用HF作为golden基线dump输入输出统计量和megatron单测用例进行对比。发现output无法对齐。暂定为精度问题进行定位。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774233317522-fcda0c90-08d5-4396-bf59-2f7a9c4f75d7.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774233290859-8118aba1-618c-43e8-9242-2c990facbd21.png)

## 脚本位置
source /sfs_turbo/hw/jingyu_P/qwen35_train/test_scripts/set_env.sh

HF脚本：

/sfs_turbo/hw/jingyu_P/msprobe/test_huggingface.py

```python
#!/usr/bin/env python3
"""
Hugging Face Qwen3.5 单测用例 - 截断到512并追加结束符mask
"""

import torch
import os
from transformers import AutoModel, AutoProcessor

# ========== 配置 ==========
model_name = "/sfs_turbo/models/Qwen3_5_35B_A3B"
data_path = "/storage/cqq/0.pt"
max_len = 512

# ========== 加载模型和 Processor ==========
print(f"Loading model from {model_name}...")
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None,
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
)

# 获取实际的 eos_token_id（Qwen3.5 通常是 151643 或 0）
eos_token_id = getattr(processor, 'eos_token_id', 0) or 0
print(f"[Config] eos_token_id: {eos_token_id}")

# 使用 accelerate（如果需要）
try:
    from accelerate import Accelerator
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    print("[Config] Using accelerator")
except ImportError:
    print("[Config] Accelerator not available, using single device")

# ========== 加载数据 ==========
print(f"[DEBUG] Loading data from {data_path}...")
forward_args = torch.load(data_path, weights_only=False, map_location='cpu')
sequence_sample = forward_args['inputs']

# 提取原始数据
raw_input_ids = sequence_sample.data['packed_input_ids']
raw_attention_mask = sequence_sample.data.get('packed_attention_mask', torch.ones_like(raw_input_ids))

print(f"[DEBUG] Original shapes: input_ids={raw_input_ids.shape}, mask={raw_attention_mask.shape}")

# ========== 截断到 512 ==========
raw_input_ids = raw_input_ids[:max_len]
raw_attention_mask = raw_attention_mask[:max_len]

print(f"[DEBUG] After truncate to {max_len}: input_ids={raw_input_ids.shape}, mask={raw_attention_mask.shape}")

# ========== 关键：追加结束符 mask=0 ==========
# 在 mask 末尾追加 0（表示该位置不计算 loss/attention）
raw_attention_mask = torch.cat([
    raw_attention_mask, 
    torch.zeros(1, dtype=raw_attention_mask.dtype, device=raw_attention_mask.device)
])

# input_ids 也要追加对应的结束符（保持长度一致）
raw_input_ids = torch.cat([
    raw_input_ids,
    torch.tensor([eos_token_id], dtype=raw_input_ids.dtype, device=raw_input_ids.device)
])

print(f"[DEBUG] After append eos: input_ids={raw_input_ids.shape}, mask={raw_attention_mask.shape}")
print(f"[DEBUG] Mask last 5 values: {raw_attention_mask[-5:].tolist()}")
print(f"[DEBUG] Input IDs last 5 values: {raw_input_ids[-5:].tolist()}")

# 加 batch 维度并移到 NPU/GPU
device = 'npu:0' if torch.npu.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu')
input_ids = raw_input_ids.unsqueeze(0).to(device)
attention_mask = raw_attention_mask.unsqueeze(0).to(device)

print(f"[DEBUG] Final input shape: {input_ids.shape}, device: {device}")

#--------------hook begin
import torch
import os
from functools import wraps

# 全局文件句柄，避免重复打开关闭
_log_file = None

def get_log_file():
    """获取或创建日志文件"""
    global _log_file
    if _log_file is None:
        rank = os.environ.get("RANK", "0")
        # 按 rank 区分文件，避免多进程写入冲突
        log_path = f"hook_log_rank{rank}.txt"
        _log_file = open(log_path, "w", encoding="utf-8")
        print(f"[Hook] 日志写入: {log_path}")
    return _log_file

def close_log_file():
    """关闭日志文件"""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None

def com(tensor, file_handle):
    """计算并写入张量统计信息"""
    ab = torch.abs(tensor)
    try:
        file_handle.write(f">sum:,{torch.sum(ab).item():.6e}\n")
    except:
        file_handle.write("This tensor do not support sum and abs!\n")
    try:
        file_handle.write(f">mean:,{torch.mean(ab).item():.6e}\n")
    except:
        file_handle.write("This tensor do not support mean!\n")
    try:
        file_handle.write(f">max:,{torch.max(ab).item():.6e}\n")
    except:
        file_handle.write("This tensor do not support max!\n")
    try:
        file_handle.write(f">min:,{torch.min(ab).item():.6e}\n")
    except:
        file_handle.write("This tensor do not support min!\n")

def print_tensor(name, tensors, file_handle, indent=0):
    """递归打印张量信息到文件"""
    prefix = "  " * indent
    
    if isinstance(tensors, torch.Tensor):
        file_handle.write(f"{prefix}{name} {list(tensors.shape)} {tensors.dtype}\n")
        file_handle.write(f"{prefix}{tensors}\n")
        if tensors.dtype != torch.bool:
            com(tensors, file_handle)
            
    elif isinstance(tensors, tuple) or isinstance(tensors, list):
        file_handle.write(f"{prefix}{name} (tuple/list, len={len(tensors)}):\n")
        for i, tensor in enumerate(tensors):
            print_tensor(f"{name}[{i}]", tensor, file_handle, indent + 1)
            
    else:
        file_handle.write(f"{prefix}{name} type: {type(tensors)}\n")

def hook_func(name, module):
    """创建 hook 函数，捕获输入输出并写入文件"""
    def hook_function(module, inputs, outputs):
        # 可选：只在特定 rank 记录
        # if os.environ.get("RANK", "0") not in ["0", 0]:
        #     return
        # import traceback
        # traceback.print_stack()
        f = get_log_file()
        
        f.write("=" * 70 + "\n")
        f.write(f"[{name}] {module.__class__.__name__}\n")
        f.write("-" * 70 + "\n")
        f.write("INPUTS:\n")
        print_tensor("inputs", inputs, f)
        
        f.write("-" * 70 + "\n")
        f.write("OUTPUTS:\n")
        print_tensor("outputs", outputs, f)
        f.write("=" * 70 + "\n\n")
        f.flush()  # 及时刷新到磁盘
        
    return hook_function

def hook_for_model(model):
    """为模型所有模块注册 forward hook"""
    for name, module in model.named_modules():
        module.register_forward_hook(hook_func(f'[forward]: {name}', module))
    
    # 注册退出时关闭文件
    import atexit
    atexit.register(close_log_file)
hook_for_model(model)
#--------------hook end
try:
    from msprobe.pytorch import seed_all, PrecisionDebugger
    seed_all()
    debugger = PrecisionDebugger(config_path="/sfs_turbo/hw/jingyu_P/msprobe/config_hf.json")
    debugger.start(model)
    use_debugger = True
except ImportError:
    use_debugger = False
    print("[DEBUG] MSProbe not available, skipping debugger")

# ========== 前向推理 ==========
print("[INFO] Starting forward pass...")
model.eval()
with torch.no_grad():
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )

print(f"[INFO] Forward pass completed")
print(f"[DEBUG] Output logits shape: {output.logits.shape if hasattr(output, 'logits') else 'N/A'}")

# 停止 debugger
if use_debugger:
    debugger.stop()
    debugger.step()
    print("[DEBUG] Debugger finished")

# 打印输出信息
if hasattr(output, 'logits'):
    logits = output.logits
    print(f"[Result] Logits shape: {logits.shape}")
    print(f"[Result] Logits dtype: {logits.dtype}")
    print(f"[Result] Logits device: {logits.device}")
    
    # 计算下一个 token 的预测（仅作验证）
    next_token_logits = logits[0, -1, :]
    predicted_token_id = torch.argmax(next_token_logits).item()
    print(f"[Result] Predicted next token ID: {predicted_token_id}")

print("[INFO] Test completed successfully")
```

megatron脚本：

/sfs_turbo/hw/jingyu_P/msprobe/run_dump25Bforward.sh

/sfs_turbo/hw/jingyu_P/msprobe/test_qwen35_bridge_dump.py

```bash
source /storage/hw/lanzeshun/Ascend/ascend-toolkit/set_env.sh
source /storage/hw/lanzeshun/Ascend/cann-8.5.0/set_env.sh

# source /mnt/sfs_turbo/hw/lanzeshun/Ascend/ascend-toolkit/set_env.sh
# source /mnt/sfs_turbo//hw/lanzeshun/Ascend/cann-8.5.0/set_env.sh
# pip install mindstudio-probe --pre

export ASCEND_DEVICE_ID=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export GLOO_SOCKET_IFNAME="eth0"
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TRITON_DISABLE_MULTI_BUFFER=1



cd /sfs_turbo/hw/jingyu_P/qwen35_train/0314/Asystem-HybridEngine/tests/npu
python /sfs_turbo/hw/jingyu_P/msprobe/test_qwen35_bridge_dump.py \
2>&1 | tee /sfs_turbo/hw/jingyu_P/qwen35_train/logs/qwen35_dump_v2.log
```

```python
import mindspeed.megatron_adaptor
import torch
import os
import torch.multiprocessing as mp
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from asystem_runtime.backend.megatron_config import MegatronConfig
import unittest
from asystem_runtime.utils.test_utils import init_dist, find_free_port
import yaml
from functools import partial
import tempfile 
from pathlib import Path


import torch

import torch
import torch.nn as nn

#增加初始化
from msprobe.pytorch import seed_all, PrecisionDebugger
seed_all()
debugger = PrecisionDebugger(config_path="/sfs_turbo/hw/jingyu_P/msprobe/dump_weights/config.json")



from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print('Available Qwen types:')
for k in sorted(CONFIG_MAPPING.keys()):
    if 'qwen' in k.lower():
        print(f'  {k}')
#--------------hook begin
import torch
def com(tensor):
    ab = torch.abs(tensor)
    try:
        print(">sum:,%e" %torch.sum(ab).item())
    except:
        print("This tensor do not support sum and abs!")
    try:
        print(">mean:,%e" %torch.mean(ab).item())
    except:
        print("This tensor do not support mean!")
    try:
        print(">max:,%e" %torch.max(ab).item())
    except:
        print("This tensor do not support max!")
    try:
        print(">min:,%e" %torch.min(ab).item())
    except:
        print("This tensor do not support min!")

def print_tensor(name, tensors):
    if isinstance(tensors, torch.Tensor):
        print(name, tensors.shape, tensors.dtype)
        print(tensors)
        if tensors.dtype != torch.bool:
            com(tensors)
    elif isinstance(tensors, tuple) or isinstance(tensors, list):
        for tensor in tensors:
            print_tensor(name, tensor)
    else:
        print(name, type(tensors))
import os
def hook_func(name, module):
    def hook_function(module, inputs, outputs):
        if name == "[forward]: model.embed_tokens":
            print(len(inputs))
            # exit()
        #if os.environ["RANK"] in ["0", 0]:
        print("---------------------------------------------------------")
        print_tensor(name + " inputs", inputs)
        print("---------------------------------------------------------")
        print_tensor(name + " outputs", outputs)
    
    return hook_function

def hook_for_model(model):
    for name, module in model.named_modules():
        module.register_forward_hook(hook_func('[forward]: ' + name, module)) 
#--------------hook end


def _run(rank, world_size, config):

    # import torch
        
    # os.environ["ASCEND_DEVICE_ID"] = str(rank)
    # torch.npu.set_device(rank)

    # print(f"Rank {rank} 绑定到NPU {rank}")
    
    # import transformers
    # from transformers import Qwen3ForCausalLM
    
    # if not hasattr(transformers, 'Qwen3_5ForConditionalGeneration'):
    #     transformers.Qwen3_5ForConditionalGeneration = Qwen3ForCausalLM
    # if not hasattr(transformers, 'Qwen3_5ForCausalLM'):
    #     transformers.Qwen3_5ForCausalLM = Qwen3ForCausalLM
    from asystem_runtime.backend.megatron_backend import MegatronBackend
    init_dist(rank, world_size)
    os.environ['HYBRID_IGNORE_LOAD_CHECK'] = "1"
    loss_configs= {
        "adaptive_kl_horizon": 10000,
        "adaptive_kl_target": 6,
        "eps_clip": 0.2,
        "kl_ctl": 0.0,
        "temperature": 1,
        "token_normalize_scope": "dp",
    }
    config['loss_configs'] = loss_configs
    port = int(os.environ['RANK']) + 4444

    megatron_backend = MegatronBackend(config)
    # Initialize Megatron backend
    print(f">>>>>>>JY Initialize Megatron backend")
    megatron_backend.initialize()

    # set_up model
    print(f">>>>>>>JY set_up model")
    megatron_backend.setup_model_with_mbridge()
    # 获取 DDP 包装的模型引擎
    model_engine = megatron_backend.model_engine[0]

    # 从 DDP 中提取原始模型
    model = model_engine.module  # 关键修复：DDP 使用 .module 而非 .model

    # print("################")
    # print(model.parameters())
    # print(f"huyiming : model={model.parameters()}")
    
    # if os.environ["RANK"] in ["0", 0]:
    #     for name, data in model.named_parameters(): 
    #         print(f"huyiming : print model.named_parameters")
    #         print(name)
    #         print(data)
    # print( megatron_backend.model_engine[0]) 
    # register_debug_hooks(megatron_backend.model_engine[0])


    # megatron_backend.update_ref_model("/storage/pretrained_models/ling-mini-2.5-256k-mcore-tp1pp1ep16etp1-te-gemm-vllm/")



    from asystem_runtime.utils.global_configs import config as global_config 

    #构造输入样本
    data = torch.load('/storage/cqq/0.pt', weights_only=False,map_location="cpu")
    # /storage/cqq/0.pt
    # /mnt/sfs_turbo/cqq/0.pt

    sequence_sample = data['inputs']
    device = torch.cuda.current_device()
    max_len = 511
    seqlens = getattr(sequence_sample, "seqlens", None)

    def _sum_seqlens(v):
        return sum(sum(x) for x in v)

    # Determine effective truncation length for logprobs-related fields.
    trunc_len = max_len
    for key in ["ppo_loss_mask", "old_logp", "advantages", "kl_rewards"]:
        if key in sequence_sample.data and torch.is_tensor(sequence_sample.data[key]):
            trunc_len = min(trunc_len, sequence_sample.data[key].shape[0])
    if "logprobs" in data and torch.is_tensor(data["logprobs"]):
        trunc_len = min(trunc_len, data["logprobs"].shape[0])
    if isinstance(seqlens, dict):
        for key in ["ppo_loss_mask", "old_logp", "advantages", "kl_rewards"]:
            if key in seqlens:
                trunc_len = min(trunc_len, _sum_seqlens(seqlens[key]))

    # Determine packed_input_ids target length separately.
    packed_target_len = None
    if "packed_input_ids" in sequence_sample.data and torch.is_tensor(sequence_sample.data["packed_input_ids"]):
        packed_len = sequence_sample.data["packed_input_ids"].shape[0]
        packed_target_len = min(packed_len, trunc_len + 1)
        print(f"huyiming : packed_target_len={packed_target_len}")
        if isinstance(seqlens, dict) and "packed_input_ids" in seqlens:
            packed_target_len = min(packed_target_len, _sum_seqlens(seqlens["packed_input_ids"]))

    for k, v in sequence_sample.data.items():
        if not torch.is_tensor(v):
            continue
        if k == "packed_input_ids":
            if packed_target_len is not None:
                sequence_sample.data[k] = v[:packed_target_len]
        else:
            sequence_sample.data[k] = v[:trunc_len]
    if "logprobs" in data and torch.is_tensor(data["logprobs"]):
        data["logprobs"] = data["logprobs"][:trunc_len]
    if isinstance(seqlens, dict):
        # Keep seqlens consistent with truncation for each key.
        for key, v in seqlens.items():
            target = trunc_len
            if key == "packed_input_ids" and packed_target_len is not None:
                target = packed_target_len
            remain = target
            new_seqlens = []
            for row_in in v:
                row_out = []
                for l in row_in:
                    if remain <= 0:
                        row_out.append(0)
                    else:
                        use = min(l, remain)
                        row_out.append(use)
                        remain -= use
                new_seqlens.append(row_out)
            seqlens[key] = new_seqlens
    device = torch.cuda.current_device()
    for k, v in sequence_sample.data.items():
        sequence_sample.data[k] = v.to(device)

    #engine_forward
    print(f">>>>>>>JY engine_forward")
    from asystem_runtime.rl_function.actor_function import logp_compute
    engine_forward = partial(megatron_backend.forward, use_ref_model=False)
    from realhf.api.cli_args import MicroBatchSpec
    mb_spec = MicroBatchSpec(1, 16000)

    print(f">>>>>>>JY logprobs")
    #logprobs = logp_compute(sequence_sample, mb_spec, engine_forward)
    #print(logprobs)
    if torch.distributed.get_rank() == 0:
        print("huyiming : in hook")
        # hook_for_model(model)
    from asystem_runtime.rl_function.actor_function import prepare_data_and_fwd_function
    print(f"enable_padding_data={config.get('enable_padding_data')}")
    # if  config.get('enable_padding_data', False):
    #     sequence_sample = None

    print(f">>>>>>>JY prepare_data_and_fwd_function")
    mb_inputs, forward_step,  total_loss_weight  = prepare_data_and_fwd_function(sequence_sample, mb_spec,
                                                            megatron_backend.model_optimizer, None)
    megatron_backend.set_global_step(1)
    # self.engine.save_hf_checkpoint("/tmp")
    debugger.start(model)  #开始

    outputs, stats = megatron_backend.train(mb_inputs, forward_step,
                                        lambda seq_sample: sum([sum(seqlens) for seqlens in seq_sample.seqlens[seq_sample._get_split_key()]]))
    debugger.stop() #结束
    debugger.step() #采集



    train_loss_configs =  global_config['loss_configs']
    assert train_loss_configs.keys() == loss_configs.keys(), "user setting loss configs keys are different from loss configs in training"
    for k in train_loss_configs.keys():
        assert train_loss_configs[k] == loss_configs[k],  "user setting loss configs value is different from loss configs in training"

    # test save_model
    print(f">>>>>>>JY save model")
    tmp_path = "/sfs_turbo/hw/jingyu_P/qwen35_train/0310/save_model"
    megatron_backend.save_hf_checkpoint(
        tmp_path
    )



@unittest.skipIf(torch.cuda.device_count() < 16, "Skip on gpu as cpu test covers it.")
class CheckpointUtilsTest(unittest.TestCase):
    def test_save_safe_tensors(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_file = f"/sfs_turbo/hw/jingyu_P/msprobe/qwen35_35b_skip.yaml"
        with open(yaml_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(config)
        config['enable_padding_data'] = False

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        world_size = 16
        mp.spawn(
            _run,
            nprocs=16,
            args=(world_size,config, ),
            join=True,
            daemon=False,
            start_method="spawn",
        )


 
 
if __name__ == "__main__":
    # export PYTHONPATH=/dnn_training_sys/users/xuantai/rl/antllm:/dnn_training_sys/users/xuantai/rl/Megatron-LM:/dnn_training_sys/users/hanxudong.hxd/Asystem-HybridEngine/
    # python tests/unit_tests/test_save_huggingface_checkpoint.py
    unittest.main()

```

/sfs_turbo/hw/jingyu_P/msprobe/dump_weights/config.json
```json
{
    "task": "statistics",
    "dump_path": "/sfs_turbo/hw/jingyu_P/msprobe/dump_weights/dump",
    "rank": [0],
    "step": [],
    "level": "L0",
    "async_dump": false,
    "extra_info": true,

    "statistics": {
        "scope": [], 
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

yaml：

```yaml
use_megatron_bridge: true 
adam_beta1: 0.9
adam_beta2: 0.999
adam_eps: 1.0e-08
tensor_model_parallel_size: 1
context_parallel_size: 1
expert_model_parallel_size: 2
pipeline_model_parallel_size: 1 #4
expert_tensor_parallel_size: 1
tokenizer_model: /sfs_turbo/models/Qwen3_5_35B_A3B
# /sfs_turbo/models/Qwen3.5-35B-A3B_layer20  # /.aistudio/aistudio-modelhub/nas/modelhub_15500012_88100072  #/dnn_training_sys/model/nlp/Qwen3-Next-80B-A3B-Instruct-hgf
load:  /sfs_turbo/models/Qwen3_5_35B_A3B
# /sfs_turbo/models/Qwen3.5-35B-A3B_layer20 # /.aistudio/aistudio-modelhub/nas/modelhub_15500012_88100072 # /dnn_training_sys/model/nlp/Qwen3-Next-80B-A3B-Instruct-hgf
tokenizer_type: HuggingFaceTokenizer
sequence_parallel: false
lr: 2.0e-06
# lr_decay_style: constant
lr_warmup_iters: 40
seq_length: 1024
max_position_embeddings: 1024
micro_batch_size: 1
norm_epsilon: 1e-06
normalization: RMSNorm
layernorm_zero_centered_gamma: true
rmsnorm_zerocenter: true
num_attention_heads: 16
num_layers: 8
num_hidden_layers: 8
num_experts: 256
moe_router_topk: 8
optimizer_cpu_offload: false
use_distributed_optimizer: true
use_precision_aware_optimizer: false
main_grads_dtype: "fp32"
main_params_dtype: "fp32"
exp_avg_dtype: "fp32"
exp_avg_sq_dtype: "fp32"
num_query_groups: 2
attention_softmax_in_fp32: true
auto_detect_ckpt_format: true
bf16: true
ckpt_format: torch_dist
clip_grad: 1
disable_bias_linear: true
eps_clip: 0.2
expert-ffn-hidden-size: 2048
# ffn_hidden_size: 18944
global_batch_size: 256
group_query_attention: true
hidden_dropout: 0
hidden_size: 2048
rotary_base: 10000000
train_iters: 5000000
padded_vocab_size: 1000000
rerun_mode: disabled
attention_backend: "flash"
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
deallocate_pipeline_outputs: false
moe_token_dispatcher_type: alltoall
use_precision_aware_optimizer_no_fp8_or_ds_fp8: true
layernorm_zero_centered_gamma: true
rmsnorm_zerocenter: true
apply-layernorm-1p: true
remote_megatron_config:
  # 配置使用 megatron bridge 
  use_megatron_bridge: true
  # 优化器配置
  use_distributed_optimizer: true
  optimizer: adam  # 添加这一行，指定优化器类型
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-08
  lr: 1.0e-6
  lr_warmup_iters: 40
  clip_grad: 1.0
override_transformer_config:
  layernorm_zero_centered_gamma: true
  rmsnorm_zerocenter: true
  apply-layernorm-1p: true
  use_flash_attn: true
  recompute_method: uniform
  recompute_granularity: full
  recompute_num_layers: 1
  multi_latent_attention: true
  attention_mask_type: causal
  use_fused_rotary_pos_emb: true
  context_parallel_size: 1
```

HF模型结构

```plain
DistributedDataParallel(
  (module): Qwen3_5MoeModel(
    (visual): Qwen3_5MoeVisionModel(
      (patch_embed): Qwen3_5MoeVisionPatchEmbed(
        (proj): Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))
      )
      (pos_embed): Embedding(2304, 1152)
      (rotary_pos_emb): Qwen3_5MoeVisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-26): 27 x Qwen3_5MoeVisionBlock(
          (norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (attn): Qwen3_5MoeVisionAttention(
            (qkv): Linear(in_features=1152, out_features=3456, bias=True)
            (proj): Linear(in_features=1152, out_features=1152, bias=True)
          )
          (mlp): Qwen3_5MoeVisionMLP(
            (linear_fc1): Linear(in_features=1152, out_features=4304, bias=True)
            (linear_fc2): Linear(in_features=4304, out_features=1152, bias=True)
            (act_fn): GELUTanh()
          )
        )
      )
      (merger): Qwen3_5MoeVisionPatchMerger(
        (norm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
        (linear_fc1): Linear(in_features=4608, out_features=4608, bias=True)
        (act_fn): GELU(approximate='none')
        (linear_fc2): Linear(in_features=4608, out_features=2048, bias=True)
      )
    )
    (language_model): Qwen3_5MoeTextModel(
      (embed_tokens): Embedding(248320, 2048)
      (layers): ModuleList(
        (0-2): 3 x Qwen3_5MoeDecoderLayer(
          (linear_attn): Qwen3_5MoeGatedDeltaNet(
            (act): SiLUActivation()
            (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
            (norm): Qwen3_5MoeRMSNormGated()
            (out_proj): Linear(in_features=4096, out_features=2048, bias=False)
            (in_proj_qkv): Linear(in_features=2048, out_features=8192, bias=False)
            (in_proj_z): Linear(in_features=2048, out_features=4096, bias=False)
            (in_proj_b): Linear(in_features=2048, out_features=32, bias=False)
            (in_proj_a): Linear(in_features=2048, out_features=32, bias=False)
          )
          (mlp): Qwen3_5MoeSparseMoeBlock(
            (gate): Qwen3_5MoeTopKRouter()
            (experts): Qwen3_5MoeExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen3_5MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
              (up_proj): Linear(in_features=2048, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2048, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
          )
          (input_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
        )
        (3): Qwen3_5MoeDecoderLayer(
          (self_attn): Qwen3_5MoeAttention(
            (q_proj): Linear(in_features=2048, out_features=8192, bias=False)
            (k_proj): Linear(in_features=2048, out_features=512, bias=False)
            (v_proj): Linear(in_features=2048, out_features=512, bias=False)
            (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
            (q_norm): Qwen3_5MoeRMSNorm((256,), eps=1e-06)
            (k_norm): Qwen3_5MoeRMSNorm((256,), eps=1e-06)
          )
          (mlp): Qwen3_5MoeSparseMoeBlock(
            (gate): Qwen3_5MoeTopKRouter()
            (experts): Qwen3_5MoeExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen3_5MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
              (up_proj): Linear(in_features=2048, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2048, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
          )
          (input_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
        )
        (4-6): 3 x Qwen3_5MoeDecoderLayer(
          (linear_attn): Qwen3_5MoeGatedDeltaNet(
            (act): SiLUActivation()
            (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
            (norm): Qwen3_5MoeRMSNormGated()
            (out_proj): Linear(in_features=4096, out_features=2048, bias=False)
            (in_proj_qkv): Linear(in_features=2048, out_features=8192, bias=False)
            (in_proj_z): Linear(in_features=2048, out_features=4096, bias=False)
            (in_proj_b): Linear(in_features=2048, out_features=32, bias=False)
            (in_proj_a): Linear(in_features=2048, out_features=32, bias=False)
          )
          (mlp): Qwen3_5MoeSparseMoeBlock(
            (gate): Qwen3_5MoeTopKRouter()
            (experts): Qwen3_5MoeExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen3_5MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
              (up_proj): Linear(in_features=2048, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2048, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
          )
          (input_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
        )
        (7): Qwen3_5MoeDecoderLayer(
          (self_attn): Qwen3_5MoeAttention(
            (q_proj): Linear(in_features=2048, out_features=8192, bias=False)
            (k_proj): Linear(in_features=2048, out_features=512, bias=False)
            (v_proj): Linear(in_features=2048, out_features=512, bias=False)
            (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
            (q_norm): Qwen3_5MoeRMSNorm((256,), eps=1e-06)
            (k_norm): Qwen3_5MoeRMSNorm((256,), eps=1e-06)
          )
          (mlp): Qwen3_5MoeSparseMoeBlock(
            (gate): Qwen3_5MoeTopKRouter()
            (experts): Qwen3_5MoeExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen3_5MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
              (up_proj): Linear(in_features=2048, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2048, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
          )
          (input_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen3_5MoeRMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen3_5MoeTextRotaryEmbedding()
    )
  )
)

```

megatron模型结构：

```plain
 Float16Module(
  (module): Qwen3VLModel(
    (vision_model): Qwen3VLVisionModel(
      (patch_embed): Qwen3VLVisionPatchEmbed(
        (proj): Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))
      )
      (pos_embed): Embedding(2304, 1152)
      (rotary_pos_emb): Qwen3VLVisionRotaryEmbedding()
      (decoder): Qwen3VLVisionTransformerBlock(
        (layers): ModuleList(
          (0-26): 27 x TransformerLayer(
            (input_layernorm): IdentityOp()
            (self_attention): Qwen3VLSelfAttention(
              (core_attention): MindSpeedTEDotProductAttention(
                (core_attention): FlashAttention()
              )
              (linear_proj): RowParallelLinear(in_features=1152, out_features=1152, bias=False, TP=1)
              (linear_qkv): MindSpeedTELayerNormColumnParallelLinear()
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): IdentityOp()
            (mlp): MLP(
              (linear_fc1): MindSpeedTELayerNormColumnParallelLinear()
              (linear_fc2): RowParallelLinear(in_features=4304, out_features=1152, bias=False, TP=1)
            )
          )
        )
        (deepstack_merger_list): ModuleList()
      )
      (merger): Qwen3VLVisionPatchMerger(
        (patch_norm): FusedLayerNorm()
        (linear_fc1): ColumnParallelLinear(in_features=4608, out_features=4608, bias=False, TP=1)
        (linear_fc2): RowParallelLinear(in_features=4608, out_features=2048, bias=False, TP=1)
      )
    )
    (language_model): Qwen3VLGPTModel(
      (embedding): LanguageModelEmbedding(
        (word_embeddings): VocabParallelEmbedding()
        (embedding_dropout): Dropout(p=0.0, inplace=False)
      )
      (rotary_pos_emb): Qwen3VLMultimodalRotaryEmbedding()
      (decoder): Qwen3VLTransformerBlock(
        (layers): ModuleList(
          (0-2): 3 x TransformerLayer(
            (input_layernorm): IdentityOp()
            (self_attention): GatedDeltaNet(
              (in_proj): MindSpeedTELayerNormColumnParallelLinear()
              (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
              (out_norm): RMSNorm()
              (out_proj): RowParallelLinear(in_features=4096, out_features=2048, bias=False, TP=1)
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): RMSNorm()
            (mlp): MoELayer(
              (router): TopKRouter()
              (experts): TEGroupedMLP(
                (linear_fc1): MindSpeedTEColumnParallelGroupedLinear()
                (linear_fc2): MindSpeedTERowParallelGroupedLinear()
              )
              (shared_experts): SharedExpertMLP(
                (linear_fc1): ColumnParallelLinear(in_features=2048, out_features=1024, bias=False, TP=1)
                (linear_fc2): RowParallelLinear(in_features=512, out_features=2048, bias=False, TP=1)
              )
            )
          )
          (3): TransformerLayer(
            (input_layernorm): IdentityOp()
            (self_attention): Qwen3VLSelfAttention(
              (core_attention): MindSpeedTEDotProductAttention(
                (core_attention): FlashAttention()
              )
              (linear_proj): RowParallelLinear(in_features=4096, out_features=2048, bias=False, TP=1)
              (linear_qkv): MindSpeedTELayerNormColumnParallelLinear()
              (q_layernorm): RMSNorm()
              (k_layernorm): RMSNorm()
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): RMSNorm()
            (mlp): MoELayer(
              (router): TopKRouter()
              (experts): TEGroupedMLP(
                (linear_fc1): MindSpeedTEColumnParallelGroupedLinear()
                (linear_fc2): MindSpeedTERowParallelGroupedLinear()
              )
              (shared_experts): SharedExpertMLP(
                (linear_fc1): ColumnParallelLinear(in_features=2048, out_features=1024, bias=False, TP=1)
                (linear_fc2): RowParallelLinear(in_features=512, out_features=2048, bias=False, TP=1)
              )
            )
          )
          (4-6): 3 x TransformerLayer(
            (input_layernorm): IdentityOp()
            (self_attention): GatedDeltaNet(
              (in_proj): MindSpeedTELayerNormColumnParallelLinear()
              (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
              (out_norm): RMSNorm()
              (out_proj): RowParallelLinear(in_features=4096, out_features=2048, bias=False, TP=1)
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): RMSNorm()
            (mlp): MoELayer(
              (router): TopKRouter()
              (experts): TEGroupedMLP(
                (linear_fc1): MindSpeedTEColumnParallelGroupedLinear()
                (linear_fc2): MindSpeedTERowParallelGroupedLinear()
              )
              (shared_experts): SharedExpertMLP(
                (linear_fc1): ColumnParallelLinear(in_features=2048, out_features=1024, bias=False, TP=1)
                (linear_fc2): RowParallelLinear(in_features=512, out_features=2048, bias=False, TP=1)
              )
            )
          )
          (7): TransformerLayer(
            (input_layernorm): IdentityOp()
            (self_attention): Qwen3VLSelfAttention(
              (core_attention): MindSpeedTEDotProductAttention(
                (core_attention): FlashAttention()
              )
              (linear_proj): RowParallelLinear(in_features=4096, out_features=2048, bias=False, TP=1)
              (linear_qkv): MindSpeedTELayerNormColumnParallelLinear()
              (q_layernorm): RMSNorm()
              (k_layernorm): RMSNorm()
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): RMSNorm()
            (mlp): MoELayer(
              (router): TopKRouter()
              (experts): TEGroupedMLP(
                (linear_fc1): MindSpeedTEColumnParallelGroupedLinear()
                (linear_fc2): MindSpeedTERowParallelGroupedLinear()
              )
              (shared_experts): SharedExpertMLP(
                (linear_fc1): ColumnParallelLinear(in_features=2048, out_features=1024, bias=False, TP=1)
                (linear_fc2): RowParallelLinear(in_features=512, out_features=2048, bias=False, TP=1)
              )
            )
          )
        )
        (final_layernorm): RMSNorm()
      )
      (output_layer): ColumnParallelLinear(in_features=2048, out_features=248320, bias=False, TP=1)
    )
  )
)

```



## 定位记录
2026/03/21:

通过统计量发现embading层输入输出一致，但是到layernorm层输入一致输出不一致

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774233813572-d3a512e2-05f6-4fc9-96e0-10e5a6e353d4.png)

首先确认norm_epsilon参数是否传入，根据挂hook等方式确认参数生效。hf和megatron两边对等都是1e-6。

megatron的norm具体实现是被Mindspeed的MindSpeedTELayerNormColumnParallelLinear()实现给包住的，所以在mindspeed打印LayerNorm。<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774234512547-45b90163-605a-4197-9d1f-68e6f98ba3a8.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774234550913-65f97f55-1028-4168-9ad0-9580b896e811.png)

mindspeed内部input/output。input一致，output不一致。证明问题在layernorm实现

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774238413348-826aa780-f518-46f0-9656-a4868156df41.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774234585630-843d9524-c4d8-4a82-83eb-9214c7ccfeb0.png)

发现输出不一致，后续怀疑权重不一致。发现打印出来的的权重确实在每一层的layernorm侧对不上。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774234815945-cea93635-9790-48e9-be09-9aa1a093591e.png)

定位到权重转换后发现megatron bridge会有对权重+1 -1的操作，后经过茶盏发现是rmsnorm权重转换时走了rmsnorm->zero centered rmsnorm分支

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774234881529-0848d326-27d6-4aac-b0a8-c202b4e79df7.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774234893299-d47ad630-80df-43cc-836e-27bf7b779890.png)

qwen3 next中映入了zero centered rmsnorm归一化方式，mindspeed内部暫未適配。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774235023809-66803105-da6c-4439-bbaa-134d67f7e21b.png)

解決方案：嘗試將+1 -1注釋掉過後發現權重以對齊。但是仍然有精度問題。正在下一步定位解決中

我们直接打印mindspeed里的权重shape和权重内容后发现此处shape为2048.

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774250096605-5fb196a1-b684-4d12-84bc-4639509f53fc.png)

hugging face处权重维度为128

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774250239881-20f59d36-0ade-43ed-b694-bacd974e5c17.png)

经查证hf会根据head num去单独做归一化<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774252574543-8a0080cc-fe50-4b39-aaec-680336c94ccf.png)

mindspeed可能需要单独做适配。

根据进一步定位发现上面shape对不齐为定位失误。是input_layernorm权重和RMSnorm打印顺序错误导致。但是根因仍然是ZeroCenter RMSnorm没有实现导致。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774264776215-ff7ab0fa-ed91-48fe-ba70-a8c79ae6bdd9.png)

翻阅PR后发现qwen团队曾经在Megatron仓内有尝试合入PR：

[https://github.com/NVIDIA/Megatron-LM/pull/1958](https://github.com/NVIDIA/Megatron-LM/pull/1958)

后因在Megatron仓内有<font style="color:rgb(31, 35, 40);">layernorm_zero_centered_gamma 特性并在megatron-bridge内部在加载layernorm这部分权重时进行-1操作就可以等价。在NPU上需要讨论是否适配了这部分内容。</font>

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774266098075-920dc50b-be25-4498-bdd0-448d98aaaf85.png)

2026/03/24：

在MindSpeedTELayerNormColumnParallelLinear内部的norm实现已适配完成并对齐。下一场卷积层的输入输出已对齐

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774356719178-dea84fb0-1b65-4709-9ca9-3b10592e6638.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774356779581-c4c0687e-c33f-4f2e-b388-a4a1f544dca5.png)

直到后续的RMSnorm操作为止，我们发现进过RMSnorm算子后有精度对不齐的问题。<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774356892499-4662a905-e18e-4ad1-b25e-0127ca5921d5.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774356918759-6af08906-eb54-4378-afcb-6c3cc9114fff.png)

到了pre_mlp_layernorm层应该调用的是小算子的RMSNorm。根据模型实现图所有的RMSnorm都因修改为zero center RMSNorm。

2026/03/25：

目前在第四层的input已对齐。因工具原因。采集失败，正在解决中。同时进行tensor级别对比。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774420359435-322bbe4a-283b-41a6-97eb-1d88115128c1.png)

目前因为一些原因所以无法做到完全一致的截断在embadding的最后一个字符会有差异。所以只能先做到大概得比较在最后一个embadding会有差别。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774442943567-5a35bf7c-1782-4cdb-bc4a-cb6d3f57817d.png)

第0层的decode layer的输出

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774442472657-e238749f-6ac5-42ee-b7d2-d0d32e64de66.png)

第4层的decode layer的输出

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774442806245-20157151-e46c-424c-b8f2-7e158b249f0c.png)



2026/03/26：

重新对齐输出后的前向完美对齐。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186756527/1774517768723-3ef7c6fd-70de-43a2-b6ae-e97041138b9d.png)

