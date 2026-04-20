```python
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
```

```Python
# ========== Hook 工具 ==========
import torch
import os
from functools import wraps

_log_file = None

def get_log_file():
    global _log_file
    if _log_file is None:
        rank = os.environ.get("RANK", "0")
        log_path = f"hook_log_rank{rank}.txt"
        _log_file = open(log_path, "w", encoding="utf-8")
        print(f"[Hook] Log: {log_path}")
    return _log_file

def close_log_file():
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None

def tensor_stats(tensor, f):
    """写入 tensor 的 sum/mean/max/min 统计量"""
    ab = torch.abs(tensor)
    for name, fn in [("sum", torch.sum), ("mean", torch.mean), ("max", torch.max), ("min", torch.min)]:
        try:
            f.write(f"  >{name}: {fn(ab).item():.6e}\n")
        except Exception:
            f.write(f"  >{name}: unsupported\n")

def dump_tensor(name, tensors, f, indent=0):
    """递归打印 tensor 信息到文件"""
    prefix = "  " * indent
    if isinstance(tensors, torch.Tensor):
        f.write(f"{prefix}{name} {list(tensors.shape)} {tensors.dtype}\n")
        f.write(f"{prefix}{tensors}\n")
        if tensors.dtype != torch.bool:
            tensor_stats(tensors, f)
    elif isinstance(tensors, (tuple, list)):
        f.write(f"{prefix}{name} ({type(tensors).__name__}, len={len(tensors)}):\n")
        for i, t in enumerate(tensors):
            dump_tensor(f"{name}[{i}]", t, f, indent + 1)
    else:
        f.write(f"{prefix}{name} type={type(tensors).__name__}\n")

def make_hook(name):
    def hook_fn(module, inputs, outputs):
        f = get_log_file()
        f.write("=" * 70 + "\n")
        f.write(f"[{name}] {module.__class__.__name__}\n")
        f.write("-" * 70 + "\n")
        f.write("INPUTS:\n")
        dump_tensor("inputs", inputs, f)
        f.write("-" * 70 + "\n")
        f.write("OUTPUTS:\n")
        dump_tensor("outputs", outputs, f)
        f.write("=" * 70 + "\n\n")
        f.flush()
    return hook_fn

def register_hooks(model):
    for name, module in model.named_modules():
        module.register_forward_hook(make_hook(f"[forward]: {name}"))
    import atexit
    atexit.register(close_log_file)
```