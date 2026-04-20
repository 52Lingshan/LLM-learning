
  device_map 的几种用法

  # 1. 字符串快捷方式
  device_map="auto"          # 自动分配到所有可用 GPU，放不下的溢出到 CPU/磁盘
  device_map="sequential"    # 按顺序填满一张卡再用下一张
  device_map="balanced"      # 尽量均匀分配

  # 2. 单卡指定
  device_map="cuda:0"        # 整个模型放一张卡
  device_map="npu:0"         # 整个模型放 NPU

  # 3. 字典手动指定（你代码中的方式）
  device_map = {
      "model.embed_tokens": 0,
      "model.layers.0": 0,
      "model.layers.1": 1,
      ...
      "lm_head": 3,
  }

```Python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/sfs_turbo/hw/lifeng/kimi/Kimi-K2.5-bf16-new"


num_layers = 27  # Kimi-K2.5
num_gpus = 4
layers_per_gpu = (num_layers + num_gpus - 1) // num_gpus
device_map = {}
device_map["language_model.model.embed_tokens"] = 0
device_map["vision_tower"] = 0
device_map["mm_projector"] = 0
device_map["image_newline"] = 0
for i in range(num_layers):
    gpu_idx = min(i // layers_per_gpu, num_gpus - 1)
    device_map[f"language_model.model.layers.{i}"] = gpu_idx
device_map["language_model.model.norm"] = num_gpus - 1
device_map["language_model.lm_head"] = num_gpus - 1

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map=device_map,
    # device_map="npu:0",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)



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
    """递归打印张量信息到文件，同时保存 .pt 文件"""
    prefix = "  " * indent
    if not os.path.exists('./dump_data/hf/'):
        os.makedirs('./dump_data/hf/')

    if isinstance(tensors, torch.Tensor):
        safe_name = name.replace("/", "_").replace(" ", "_")
        torch.save(tensors, './dump_data/hf/' + safe_name + '.pt')
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
        f.flush()

    return hook_function

def hook_for_model(model):
    """为模型所有模块注册 forward hook"""
    for name, module in model.named_modules():
        module.register_forward_hook(hook_func(f'[forward]: {name}', module))

    import atexit
    atexit.register(close_log_file)
#--------------hook end

hook_for_model(model)

forward_args = torch.load('/sfs_turbo/hw/lifeng/kimi/Megatron-Bridge/forward_args.pt')

output = model(forward_args["input_ids"],forward_args["attention_mask"])

print(output)
```