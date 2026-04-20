# execute_dummy_batch 作用解析

> 来源：Asystem-HybridEngine NCCL Hang 问题定位过程
> 日期：2026-04-09

---

## 一句话概括

**用一个假输入跑一遍模型，让 NPU 提前"热身"并把计算图固化下来。**

---

## 背景：昇腾 NPU 的 ACL Graph 机制

昇腾 NPU（华为 Ascend）有一个类似 NVIDIA CUDA Graph 的机制，叫 **ACL Graph**：

| 阶段 | 发生了什么 |
|------|-----------|
| **首次运行（捕获期）** | NPU 把整个计算过程录制成一张"图"，记录所有算子和数据流 |
| **后续运行（重放期）** | 直接重放这张图，省去每次的内核调度开销，推理速度显著提升 |

**限制：** 输入的 shape 必须固定。如果模型权重发生变化，旧的 ACL Graph 失效，必须重新捕获。

---

## execute_dummy_batch 的执行流程

```
execute_dummy_batch (vllm_ascend/worker/worker.py:533)
│
│  构造"假 batch"：形状与真实请求相同，内容为随机数/全零
│
└── _dummy_run (model_runner_v1.py:2338)
      │
      │  用假 batch 完整执行一次模型前向传播
      │
      └── _model_forward (model_runner_v1.py:1717)
            │
            │  包含完整的 Transformer 计算（含 MoE AlltoAll）
            │
            └── acl_graph.__call__ (acl_graph.py:199)
                  │
                  │  触发 ACL Graph 录制或重捕获
                  │
                  └── torch_npu.synchronize (streams.py:85)
                        │
                        └── 等待 NPU 硬件真正完成计算
```

---

## 在 RL 训练中的角色

RL 训练中，模型权重会在每个训练步结束后更新，推理侧需要加载最新权重来生成 rollout 数据。权重更新后必须重新执行 `execute_dummy_batch`：

```
训练侧（Megatron / FSDP）
  └── 完成梯度更新，得到新权重
        │
        ▼
权重传输（asystem_runtime/weights_exchange/nccl_comm.py）
  └── P2P 传输：训练 Worker → 推理 Worker（vLLM）
        │
        ▼
推理 Worker 加载新权重
        │
        ▼
execute_dummy_batch   ← 必须执行！
  原因：旧 ACL Graph 是用旧权重录制的，
        权重改变后图中的算子参数已失效，
        必须重新捕获一张带有新权重的计算图
        │
        ▼
ACL Graph 重捕获完成
        │
        ▼
正常推理 / Rollout 生成
```

---

## 为什么 execute_dummy_batch 会导致 Hang

### 根本原因：MoE EP AlltoAll 是集合通信

MoE（混合专家）模型在 `_model_forward` 中包含 **Expert Parallel AlltoAll**：

```
作用：把 token 按照路由结果（哪个 token 去哪个 Expert）
      分发到 72 个 EP Worker 上的不同 Expert，再收集计算结果

特点：这是一个集合通信操作，类似 dist.all_to_all
      要求参与的所有 EP Worker 必须同时到达这个操作点
```

### Hang 的触发场景（本次问题）

```
正常流程（所有 Worker 同时到达）：
  EP0 进入 dummy_run ──┐
  EP1 进入 dummy_run ──┤
  ...                  ├── AlltoAll 顺利完成 ✓
  EP71 进入 dummy_run ─┘

本次异常（不同 Worker 到达时间不一致）：

  原因：nccl_comm.py 的权重 P2P 传输临时 Hang（08:57:06）
        不同 Worker 完成权重加载的时间出现偏差

  EP0~EP71：权重加载快 → 率先进入 dummy_run → 发起 AlltoAll → 等待 EP72
  EP72：    权重加载慢 → 迟迟未进入 dummy_run

  结果：
  EP0~EP71 卡在 AlltoAll，永久等待 EP72
  EP72 还没进来
  → 全局 Hang，无法自动恢复
```

---

## 与 CUDA Graph（GPU）的类比

| 维度 | NVIDIA GPU | 昇腾 NPU |
|------|-----------|---------|
| 机制名称 | CUDA Graph | ACL Graph |
| 触发捕获 | `torch.cuda.CUDAGraph.capture_begin/end` | `acl_graph.__call__` |
| 重放 | `graph.replay()` | ACL Graph 内部重放 |
| warmup 函数 | `model(dummy_input)` | `execute_dummy_batch` |
| 权重更新后需重捕获 | ✅ 是 | ✅ 是 |

---

## 涉及的代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `vllm_ascend/worker/worker.py` | 533 | `execute_dummy_batch` 入口 |
| `vllm_ascend/worker/model_runner_v1.py` | 2338 | `_dummy_run`：构造假 batch |
| `vllm_ascend/worker/model_runner_v1.py` | 1717 | `_model_forward`：完整前向 |
| `vllm_ascend/compilation/acl_graph.py` | 199 | `__call__`：ACL Graph 捕获/重放 |
| `torch_npu/npu/streams.py` | 85 | `synchronize`：等待 NPU 完成 |

---

## 相关问题记录

- 完整 Hang 定位过程：`D:\knowledge\大模型\日记\NCCL_Hang_Debug_2026-04-08.md`
