# vLLM Sleep 多轮内存泄漏分析

- **日志来源**: `log_aistudio-1br3ujqt_2026-04-29_08-00-04.txt`
- **实验配置**: kimi_k25_16b_nccl_base_4k, 2 nodes, DP8×TP2 (16 Workers/node)
- **分析日期**: 2026-04-29

---

## 一、背景

在 AReaL RL 训练中，每轮 rollout 结束后 vLLM 推理引擎调用 `CaMemAllocator.sleep()` 释放 HBM 给训练使用。本次分析聚焦 sleep 后残留显存的增长趋势。

### 1.1 sleep() 流程

```
rollout_end 事件
  └─ vllm worker.sleep()
      └─ CaMemAllocator.sleep(offload_tags)
          ├─ allocator_and_pools.clear()
          ├─ 遍历 pointer_to_
          │   ├─ 若 tag 在 offload_tags → memcpy D2H (pin_memory) → offload 到 CPU
          │   └─ unmap_and_release(handle)  # ACL 层释放物理显存
          ├─ gc.collect()
          └─ torch.npu.empty_cache()
```

### 1.2 加的调试日志位置

在 `camem.py` 的 `sleep()` 方法中，分别在 offload 操作**前**和 `empty_cache()` **后**打印：

```python
# BEFORE offload
print(f"pytorch缓存池大小torch.npu.memory_reserved: {torch.npu.memory_reserved()}")
free_mem, total_mem = torch_npu.npu.mem_get_info()
print(f"phbm占用 / 总可用: {free_mem} / {total_mem}")  # 注意：变量名是 free_mem，实际是"空闲"量

# ... unmap_and_release + gc.collect + empty_cache ...

# AFTER empty_cache
print(f"pytorch缓存池大小torch.npu.memory_reserved: {torch.npu.memory_reserved()}")
free_mem_1, total_mem_1 = torch_npu.npu.mem_get_info()
print(f"phbm占用 / 总可用: {free_mem_1} / {total_mem_1}")
```

> **日志标签有误**：`mem_get_info()` 返回 `(free, total)`，但打印了"phbm占用"，实际值是**空闲量**。

---

## 二、两个节点的 sleep 时序

| 节点 | RayEngine PID | Worker PIDs | IP |
|------|--------------|-------------|-----|
| Node 0 | 10295 | 15103-15118 | (local) |
| Node 1 | 6649 | 10201-10224 | 10.238.250.173 |

共发生 **3 轮 sleep**：

| 轮次 | 时间戳 | 触发事件 | global_step |
|------|--------|---------|-------------|
| 第 1 轮 | 08:06:58 | 初始化后首次 release_memory_occupation | 0 |
| 第 2 轮 | 08:15:55 | 第 1 轮 rollout_end | 0 |
| 第 3 轮 | 08:28:38 | 第 2 轮 rollout_end | 1 |

- Node 0 所有 Worker 的 sleep 在同一秒内完成（~200ms 内）
- Node 1 Worker 之间 sleep 耗时差异更大（第 1 轮跨度 ~16s：08:08:19 ~ 08:08:35）

---

## 三、核心发现：sleep 后残留显存持续增长

### 3.1 单 Worker 三轮对比（Worker_DP7_TP1_EP15, pid=15115, Node 0）

| 指标 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|------|--------|--------|--------|
| sleep 前 pytorch_reserved | 13.66 GiB | 14.36 GiB | 14.43 GiB |
| sleep 前 HBM 已用 | 17.99 GiB | 20.05 GiB | 23.21 GiB |
| sleep 后 pytorch_reserved | **13.66 GiB** | **13.67 GiB** | **13.67 GiB** |
| sleep 后 HBM 已用 | **4.42 GiB** | **5.69 GiB** | **8.79 GiB** |
| CaMemAllocator 释放量 | 13.57 GiB | 14.36 GiB | 14.41 GiB |

### 3.2 全 Worker 残留显存统计

| 轮次 | sleep 后残留（"still in use"） | 范围 |
|------|-------------------------------|------|
| 第 1 轮 | **~4.2 - 4.7 GiB** | TP0 略高于 TP1（~0.25 GiB） |
| 第 2 轮 | **~5.4 - 6.0 GiB** | TP0 略高于 TP1（~0.25 GiB） |
| 第 3 轮 | **~8.2 - 9.3 GiB** | TP0 明显高于 TP1（~0.5 GiB） |

### 3.3 增长趋势

```
残留显存 (GiB)
  9 ┤                                          ██
    │                                          ██
  8 ┤                                          ██
    │
  7 ┤
    │
  6 ┤                      ██
    │                      ██
  5 ┤                      ██
    │
  4 ┤     ██
    │     ██
  3 ┤
    └────────────────────────────────────────────
        第1轮(08:06)    第2轮(08:15)    第3轮(08:28)
```

**每轮增长量**：+1.3 GiB（1→2轮） → +3.1 GiB（2→3轮），增速在加快。

---

## 四、`torch.npu.empty_cache()` 效果分析

| 轮次 | reserved 变化 | 说明 |
|------|--------------|------|
| 第 1 轮 | 13.66 → 13.66 GiB（**无变化**） | empty_cache 完全无效 |
| 第 2 轮 | 14.36 → 13.67 GiB（**-0.69 GiB**） | 仅释放少量 |
| 第 3 轮 | 14.43 → 13.67 GiB（**-0.76 GiB**） | 仅释放少量 |

`pytorch_reserved` sleep 后稳定在 ~13.67 GiB，说明 PyTorch 自身的缓存池几乎不受 sleep 影响。实际释放的 HBM 主要来自 `CaMemAllocator` 的 `unmap_and_release()`（ACL 层）。

> **为什么 HBM 已用 (4.42 GiB) < pytorch_reserved (13.66 GiB)？**
> `torch.npu.memory_reserved()` 跟踪的是 PyTorch 缓存分配器的**虚拟预留量**，不等于实际物理占用。NPU 驱动的 `mem_get_info()` 反映真实物理 HBM 使用。

---

## 五、泄漏源分析

sleep 后的残留显存 = 不在 `CaMemAllocator.pointer_to_data` 管理范围内的分配。

### 5.1 可能的泄漏来源

| 来源 | 可能性 | 理由 |
|------|--------|------|
| vLLM EngineCore 内部状态（scheduler, block manager metadata） | 高 | 每轮推理后 block_pool reset 但可能有 metadata 残留 |
| KV-cache block 的 PyTorch tensor | 高 | CaMemAllocator 只管理自己分配的内存，KV-cache 如果由 PyTorch 分配则不会被 sleep 释放 |
| 推理过程中的中间 tensor 引用泄漏 | 中 | Python 循环引用可能导致 gc.collect() 不完全 |
| NCCL/HCCL 通信 buffer | 低 | 通常大小固定，不会逐轮增长 |
| FSDP 训练侧残留（参数 buffer、gradient） | 低 | 训练和推理使用不同进程 |

### 5.2 TP0 vs TP1 差异

TP0 的残留始终略高于 TP1（约 0.25-0.5 GiB），且差距在逐轮扩大。TP0 通常承担额外的 embedding/output head 参数或 all-reduce 的发起角色，可能有额外的 buffer 不在 CaMemAllocator 管理范围内。

---

## 六、排查建议

### 6.1 定位泄漏源

在 sleep 后添加以下诊断代码：

```python
# 在 empty_cache() 之后
import torch
print(torch.npu.memory_summary())  # PyTorch 分配器详情

# 检查存活的 NPU tensor
import gc
gc.collect()
npu_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_npu]
total_bytes = sum(t.nelement() * t.element_size() for t in npu_tensors)
print(f"存活 NPU tensor 数量: {len(npu_tensors)}, 总大小: {total_bytes / 1024**3:.2f} GiB")

# 按大小排序打印 top 10
npu_tensors.sort(key=lambda t: t.nelement() * t.element_size(), reverse=True)
for t in npu_tensors[:10]:
    print(f"  shape={t.shape}, dtype={t.dtype}, size={t.nelement() * t.element_size() / 1024**2:.1f} MiB")
```

### 6.2 修复日志标签

```python
free_mem, total_mem = torch_npu.npu.mem_get_info()
used_mem = total_mem - free_mem
print(f"HBM 已用/总量: {used_mem / 1024**3:.2f} GiB / {total_mem / 1024**3:.2f} GiB")
```

### 6.3 对比实验

- 跑 5+ 轮 sleep/wakeup 确认残留是否线性增长
- 在 sleep 前手动 `del` vLLM 的 `kv_cache` 相关对象，观察残留是否减少

---

## 七、与 Gloo 初始化崩溃的关联

同一日志中还出现了 `dist.all_gather_object()` Gloo TCP 崩溃（详见 `online_vllm_backend.py:1097`），涉及 3 个推理节点间的分布式初始化。这与 sleep 泄漏是**独立问题**：

- sleep 泄漏：单节点内显存管理问题
- Gloo 崩溃：跨节点网络通信问题（节点 `10.238.251.46` TCP 连接断开引发级联失败）
