# PyTorch CUDA 虚拟内存（Caching Allocator）与 vLLM Sleep Mode

## 一、PyTorch CUDA 虚拟内存机制

### 本质

PyTorch 的"虚拟内存"是 **CUDA Caching Allocator**（缓存分配器），不是 OS 意义上的虚拟内存。PyTorch 不直接调用 `cudaMalloc`/`cudaFree`，而是自己做一层内存池管理。

### 工作流程

```
分配 (malloc):
  ├─ 在缓存池中找 ≥ 所需大小的空闲 block
  │   ├─ 找到 → 直接返回（O(1)，不调 cudaMalloc）
  │   └─ 没找到 → 调 cudaMalloc 向 GPU 申请新显存
  └─ 大 block 可被拆分（split）

释放 (free):
  ├─ 不调 cudaFree，而是标记为"可复用"
  ├─ 相邻空闲 block 自动合并（merge）
  └─ 显存留在进程手中，等待下次复用
```

### 作用

- **减少 cudaMalloc 开销**：cudaMalloc 是同步操作，开销大；缓存池直接复用，近乎零开销
- **减少内存碎片**：block 拆分+合并策略，降低外部碎片
- **加速训练循环**：前向→反向→更新，每轮释放的 tensor 下轮直接复用
- **显存峰值控制**：`max_split_size_mb` 等参数可调优分配策略

### 关键配置

```bash
# 禁用缓存分配器（调试用）
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# 限制单个 block 最大拆分大小
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 可扩展分配策略（PyTorch 2.x，减少碎片）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 监控 API

```python
torch.cuda.memory_allocated()    # tensor 实际占用的显存
torch.cuda.memory_reserved()     # 缓存池占用的总显存（含空闲）
torch.cuda.memory_summary()      # 详细报告
torch.cuda.empty_cache()         # 强制清空缓存池（牺牲复用性能）
```

---

## 二、vLLM Sleep Mode

### 是什么

Sleep mode 用于**多模型共享 GPU** 场景（如 disaggregated prefill/decode）：

```
正常运行:  [Model A 占用 GPU 显存]
sleep A:   [A 的权重卸载到 CPU, GPU 显存释放]
唤醒 A:    [A 的权重从 CPU 加载回 GPU]
```

---

## 三、核心冲突：为什么 Sleep Mode 不支持 Caching Allocator

### 语义矛盾

| | Caching Allocator | Sleep Mode 需求 |
|---|---|---|
| **释放语义** | `free` → 标记可复用，显存不归还 GPU | `sleep` → 显存必须真正归还 GPU |
| **显存归属** | 进程持续持有（reserved） | 需要让其他模型/进程可用 |

### 冲突过程

```
Sleep Mode 期望:
  1. 模型权重 tensor 从 GPU 移到 CPU
  2. GPU 显存真正释放（cudaFree）
  3. 另一个模型可以使用这些显存 ✓

有 Caching Allocator 时的实际:
  1. 模型权重 tensor 移到 CPU
  2. Caching Allocator 把 block 标记为"空闲可复用"
  3. 显存仍被 PyTorch 进程持有（reserved）
  4. 其他模型/进程无法使用这些显存 ✗
```

**本质：caching allocator 的"释放=缓存复用"语义，与 sleep mode 的"释放=真正归还"语义矛盾。**

---

## 四、vLLM 的解决方案

Sleep mode 下绕过 caching allocator，直接使用 `cudaMalloc`/`cudaFree`：

```python
# 方式1：禁用 caching allocator
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# 方式2：使用自研 Custom Caching Allocator，支持"真正释放"语义
```

### 权衡

| | Caching Allocator 开启 | Caching Allocator 关闭 |
|---|---|---|
| tensor 分配速度 | 快（缓存池复用） | 慢（每次调 cudaMalloc） |
| 显存碎片 | 少（合并策略） | 多（频繁 malloc/free） |
| 显存真正释放 | 不释放 | 释放 |
| Sleep Mode 兼容 | 不兼容 | 兼容 |

---

## 五、与 vLLM BrokenPipeError 的关联

当 vLLM 在多机场景下使用 sleep mode 时，如果 caching allocator 未正确禁用，sleep 后显存未真正释放，可能导致：
- 新模型启动时 GPU OOM
- Worker 进程被 kill
- Coordinator 的管道对端消失 → BrokenPipeError

排查建议：确认 sleep mode 下 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 或使用了 vLLM 的 custom allocator。