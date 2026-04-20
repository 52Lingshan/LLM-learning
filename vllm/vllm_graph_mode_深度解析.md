# vLLM 图模式（CUDA Graph）深度解析

> 整理日期：2026-04-09
> 来源：vLLM 官方文档、知乎、GitHub Issues、LinkedIn 技术博客、vllm-ascend 官方文档、华为昇腾开发者文档

---

## 一、背景：为什么需要 CUDA Graph？

在标准的 PyTorch **Eager 模式**下，每次执行模型的 forward pass，CPU 都需要逐一向 GPU 提交 CUDA kernel。这个过程涉及大量开销：

- **Python 层逻辑**：if/else 判断、循环等每轮都重新执行
- **PyTorch dispatch 开销**：算子调度、类型检查、形状推断等
- **内存分配开销**：中间张量的动态申请与释放
- **GPU 驱动/内核启动开销**：每个 kernel 单独提交到 CUDA stream

对于 LLM **decode 阶段**，每次只生成 1 个 token，计算量极小，但上述 CPU-side overhead 占比极高，GPU 利用率低下，成为严重瓶颈。

**CUDA Graph** 技术的核心思想是：**先录制（capture），再回放（replay）**，将整段 GPU 执行序列压缩为一次调用。

---

## 二、CUDA Graph 基本原理：Capture & Replay

### 2.1 工作机制

```
第一阶段：Capture（录制）
  CPU 执行一遍 forward pass
  → 所有 CUDA kernel 调用被记录进一个 Graph 对象
  → 不实际执行，只记录执行序列和参数地址

第二阶段：Replay（回放）
  后续推理时，直接 replay 这个 Graph
  → GPU 直接执行预录制的 kernel 序列
  → CPU 只需发出一条 "replay" 指令
  → 完全绕过 Python/PyTorch dispatch
```

**关键约束**：CUDA Graph 要求执行路径和张量形状**完全静态**：
- 输入/输出张量的**内存地址**必须固定（Graph 录制时绑定的是指针地址）
- 批次大小（batch size）必须固定
- 不能有动态分支（依赖运行时数据的 if/else）

### 2.2 对 LLM 推理的适用性

| 阶段 | 特点 | 适合 CUDA Graph？ |
|------|------|-----------------|
| **Prefill 阶段** | 输入序列长度动态变化，形状不固定 | **不适合** |
| **Decode 阶段** | 每步只处理 1 个 token，batch size 相对固定 | **非常适合** |

这是 vLLM 主要对 **decode forward pass** 启用 CUDA Graph 的根本原因。

---

## 三、vLLM 的实现方案：多 batch size 捕获 + Padding

### 3.1 问题：batch size 动态变化

实际推理时，每个 decode step 的并发请求数是不断变化的。而 CUDA Graph 要求固定形状，怎么办？

**vLLM 的解法**：预先捕获多个不同 batch size 的 CUDA Graph，运行时选择最接近的那个，并对输入做 **padding**。

### 3.2 默认捕获的 batch size

vLLM 默认捕获约 **67 个不同 batch size** 的 CUDA Graph：

```
1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, ...（按步长递增至 512+）
```

可通过参数 `cudagraph_capture_sizes` 自定义捕获哪些 batch size。

### 3.3 Padding 机制

运行时实际 batch size 为 N，选择最小的已捕获 size M（M ≥ N），将输入 padding 到 M：

```
实际 batch_size = 35
选择已捕获 size  = 40（最接近且 ≥ 35 的已捕获 size）
padding 5 个 dummy token → replay size=40 的 CUDA Graph
```

多余的 padding token 不影响有效输出（通过 mask 处理）。

### 3.4 内存共享（Graph Pools）

为减少显存占用，多个 CUDA Graph 之间**共享内存池**（memory pool / graph pool）：
- 所有捕获的 graph 使用同一块预分配的内存区域
- 不同 graph 之间通过 pool 机制复用内存，避免每个 graph 单独占用一份完整显存

---

## 四、vLLM CUDA Graph 模式演进

### 4.1 V0 时代：全图模式（Full CUDA Graph）

对整个 decode forward pass 录制一个大的 CUDA Graph，包括：
- Token embeddings
- 所有 Transformer 层（attention + MLP）
- 输出层（logits）

**问题**：FlashAttention 等自定义算子内部存在**动态控制流**，无法直接被整图捕获。解决方法是在 graph capture 时将 attention 替换为专门适配固定形状的实现（如 `flash_attn_with_kvcache`）。

### 4.2 V1 时代：Piecewise CUDA Graph（分段图模式）

随着 `torch.compile` 的引入，vLLM v1 架构采用了更精细的**分段（piecewise）CUDA Graph** 设计。

#### 核心思想

不对整个 forward pass 录制一个 graph，而是**将计算图分割为多个片段**：

```
forward pass:
[QKV proj] → [FlashAttention] → [Output proj] → [MLP gate] → [MLP up/down]
    ↑               ↑                ↑                ↑
  可捕获          不可捕获           可捕获            可捕获
(piecewise graph) (eager执行)  (piecewise graph) (piecewise graph)
```

- **可捕获片段**（compilable regions）：纯计算部分（QKV 投影、MLP 等）
- **不可捕获片段**（non-compilable ops）：自定义算子（FlashAttention kernel 本身）、动态操作，以 `splitting_ops` 标记为分割点

#### 技术实现

1. **CompilationConfig**（2024 年 12 月引入）：统一配置 `torch.compile` 和 CUDA Graph 捕获
2. **`splitting_ops`**：标记哪些算子作为图分割点，不被 compile 捕获
3. **`custom_ops`**：控制哪些自定义 Triton/CUDA ops 参与编译
4. 对每个 `cudagraph_capture_sizes` 指定的 batch size，torch.compile 该具体形状后，再捕获 CUDA Graph

> **注意**：Piecewise 模式要求 `allreduce` 必须是 **out-of-place**（非原地操作），因为 PyTorch custom ops 在编译图中不支持 in-place allreduce。

---

## 五、Runtime 模式详解

vLLM v1 定义了以下 CUDA Graph runtime 模式（通过 `cudagraph_mode` 参数配置）：

| 模式 | 描述 |
|------|------|
| `NONE` | 纯 Eager 模式，不使用任何 CUDA Graph，等价于 `enforce_eager=True` |
| `PIECEWISE` | 分段图模式，仅对可编译片段捕获 graph，自定义算子 eager 执行（**v1 默认**） |
| `FULL` | 全图模式，对整个 decode forward pass 捕获一个完整 CUDA Graph |
| `FULL_AND_PIECEWISE` | 双模式，同时支持 Full 和 Piecewise，根据批次类型自动分发 |

### 模式降级（Downgrade）机制

不是所有 attention backend 都兼容每种 CUDA Graph 模式。vLLM 会自动将不兼容的模式降级：
- 使用不支持 Full Graph 的 attention backend → 自动降级为 `PIECEWISE`
- 使用不支持任何 graph 的 backend → 降级为 `NONE`

### Prefill / Decode 批次的差异化处理

Full CUDA Graph 捕获时，对不同批次类型分别处理：
- **`prefill` / `mixed_batch`**（prefill 和 decode 混合）：单独捕获 graph
- **`uniform_decode`**（纯 decode 批次）：单独捕获 graph

---

## 六、与 torch.compile 的深度集成

vLLM v1 的 CUDA Graph 不是孤立的，而是深度集成在 `torch.compile` 工作流中：

```
源码 (Python/模型)
    ↓ torch.compile (Dynamo trace)
IR 图 (FX Graph)
    ↓ 图分割 (splitting_ops 标记分割点)
多个编译片段
    ↓ Inductor / 自定义后端编译
优化后的 kernel
    ↓ CUDA Graph capture (对每个 cudagraph_capture_size)
CUDA Graph 对象（固定形状）
    ↓ 推理时 replay
```

**关键流程**：
1. 对 `cudagraph_capture_sizes` 中每个 batch size，torch.compile 该具体形状（concrete shape specialization）
2. 编译后为该形状捕获 CUDA Graph
3. 推理时，根据实际 batch size 选择最近的已捕获 graph，padding 后 replay

---

## 七、配置参数

### 7.1 主要配置

```python
# 最简单：完全禁用 CUDA Graph
LLM(model="...", enforce_eager=True)

# 使用 CompilationConfig 精细控制（vLLM v1）
from vllm.config import CompilationConfig

LLM(
    model="...",
    compilation_config=CompilationConfig(
        cudagraph_mode="PIECEWISE",           # 或 "FULL", "NONE", "FULL_AND_PIECEWISE"
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],  # 自定义 batch size
    )
)
```

### 7.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enforce_eager` | `False` | 为 True 时禁用所有 CUDA Graph，强制 eager 执行 |
| `cudagraph_mode` | `PIECEWISE`（v1） | CUDA Graph 运行时模式 |
| `cudagraph_capture_sizes` | 约 67 个预设值 | 预捕获的 batch size 列表 |
| `splitting_ops` | 内置默认列表 | 作为分割点的算子，不被 compile 捕获 |
| `custom_ops` | `"all"` | 参与编译的自定义算子范围 |

### 7.3 Optimization Level（优化等级）

vLLM 提供 4 个优化等级，可用于权衡启动时间与性能：

| 等级 | 描述 |
|------|------|
| `-O0` | 禁用所有优化，等价于 eager 模式 |
| `-O1` | 基础优化 |
| `-O2` | 启用 torch.compile |
| `-O3` | 启用 torch.compile + CUDA Graph（最高性能） |

### 7.4 Lazy CUDA Graph Capture（懒捕获，RFC 阶段）

默认情况下，vLLM 在启动时**预捕获所有** 67 个 graph，这会导致：
- 启动时间延长 10 秒以上
- 显存占用增加

社区提出了 **Lazy Capture** 方案（[RFC #20098](https://github.com/vllm-project/vllm/issues/20098)）：按需在第一次遇到某个 batch size 时才捕获对应 graph。

---

## 八、性能收益与代价

### 8.1 性能收益

**Decode 阶段（最大受益）**：
- 消除所有 CPU-side overhead：Python 逻辑、PyTorch dispatch、内存分配、驱动开销
- decode latency 可降低 **20%~40%**（依模型和 GPU 而定）
- 小 batch size（1~16）时收益最显著
- 整体吞吐量提升，GPU 利用率提高

**torch.compile + piecewise graph 额外收益**：
- 算子融合（kernel fusion）：相邻可编译片段内的算子可被融合为单一 kernel
- 消除中间 tensor 分配
- 针对具体 batch size 的形状特化优化

### 8.2 代价与限制

| 代价 | 说明 |
|------|------|
| **启动时间增加** | warm-up + capture 所有预设 batch size，可能增加 10s+ 启动时间 |
| **显存占用增加** | 每个 graph 需要固定内存；graph pool 本身也占用显存 |
| **Padding 浪费** | 实际 batch size 与捕获 size 不完全匹配时，有少量计算浪费 |
| **灵活性降低** | 不支持动态形状、动态控制流 |
| **调试困难** | Graph replay 下 profiler 只能看到单条 graph launch，看不到内部 kernel |
| **FP8 GEMM 问题** | CUDA Graph padding 不保证 M%4==0，可能影响 FP8 GEMM 性能（[RFC #30717](https://github.com/vllm-project/vllm/issues/30717)） |

---

## 九、各模式综合对比

| 维度 | Eager 模式 | Piecewise Graph | Full Graph |
|------|-----------|----------------|------------|
| CPU overhead | 高 | 低（编译片段） | 极低 |
| GPU 利用率 | 低（decode 时） | 高 | 最高 |
| decode latency | 高 | 低 | 最低 |
| 内存开销 | 无额外 | 中 | 高 |
| 启动时间 | 快 | 慢（编译） | 最慢（capture） |
| 动态形状支持 | 完全支持 | 部分（分割点外） | 不支持 |
| 自定义算子兼容 | 完全兼容 | 大部分兼容 | 受限 |
| 调试便利性 | 最好 | 较好 | 最差 |
| 适用场景 | 开发/调试 | **生产（推荐默认）** | 极致性能优化 |

---

## 十、何时启用 / 禁用图模式

### 推荐启用（默认行为）

- **生产部署**：追求最低 decode latency 和最高吞吐量
- **batch size 稳定**：请求并发量相对稳定，padding 浪费少
- **标准模型**：使用 vLLM 原生支持的模型和 attention backend
- **decode-heavy 负载**：长输出序列，decode 步骤远多于 prefill

### 推荐禁用（`enforce_eager=True`）

- **调试/开发阶段**：需要 profiler 看到完整 kernel 执行细节
- **自定义算子不兼容**：添加了不支持 CUDA Graph 的自定义 op
- **内存非常紧张**：显存不足以承担 graph 捕获的额外开销
- **动态形状需求**：输入形状高度动态，padding 浪费严重
- **非 GPU 推理**：CPU、NPU 等非 NVIDIA GPU 后端

---

## 十一、扩展：多模态 & 投机解码的 CUDA Graph 支持

### Vision Encoder CUDA Graphs

vLLM 为视觉编码器（ViT）引入了独立的 **Encoder CUDA Graphs**：
- 与 decoder CUDA Graph **正交**（两者可同时启用）
- **EncoderCudaGraphManager**：基于 budget 的捕获/回放管理，按 token budget 分档（而非 batch size）
- 针对 prefill 阶段的视觉 token 处理进行加速

### Eagle 投机解码

vLLM 的 Eagle（speculative decoding）模块也有专门的 CUDA Graph 支持：
- **仅支持 FULL 模式**
- 初始化时为 Eagle draft model 单独捕获 CUDA Graph

---

## 十二、底层实现关键类

| 类/文件 | 作用 |
|---------|------|
| `vllm/compilation/cuda_graph.py` | CUDA Graph wrapper，封装 capture/replay，提供属性透传 |
| `vllm/config.py` → `CompilationConfig` | 统一配置入口，定义 `cudagraph_mode`、`cudagraph_capture_sizes` 等 |
| `vllm/v1/worker/gpu/model_runner.py` | GPU model runner，控制 warm-up 流程和 graph 分发 |
| `vllm/v1/worker/encoder_cudagraph.py` | 视觉编码器 CUDA Graph 管理器 |
| `CUDAGraphRunner` | 接收 `runtime_mode` 和 `batch_descriptor`，决定 replay 哪个 graph |

### `CUDAGraphRunner` 工作流程

```
初始化：分配 runtime mode（FULL 或 PIECEWISE）
       ↓
运行时：接收 runtime_mode + batch_descriptor
       ↓
判断：是否命中已捕获的 graph？
  ├── 命中 → 更新输入 buffer → replay graph
  └── 未命中 → eager 执行（fallback）
```

---

---

## 十三、昇腾 ACL Graph 机制详解

### 13.1 ACL Graph 是什么？

ACL Graph（aclgraph）是华为昇腾 NPU 对应 CUDA Graph 的图捕获与回放加速机制，用于解决 PyTorch Eager 模式下的 **Host 侧开销**问题。

在 Eager 模式下，每个算子的执行路径为：

```
Host Python API → Host C++ 算子 dispatch → Device NPU kernel 执行
```

每个算子都需要经过 Host 侧的 Python 和 C++ 开销。对于 LLM decode 阶段（计算量轻但算子数量多），这种 Host-bound 开销成为显著瓶颈。

ACL Graph 通过**一次 Host 调用回放预录制的 NPU 任务序列**，完全绕过逐算子的 dispatch 开销。

### 13.2 技术栈：TorchAir（torchair）

ACL Graph 构建在 **TorchAir（torchair）** 之上，torchair 是 PyTorch 和昇腾 GE（Graph Engine）之间的图模式桥接层：

```
PyTorch 模型
    ↓  torch.compile(backend="npu_backend")
TorchAir / torchair（FX Graph 追踪 + GE 图构建）
    ↓
昇腾 GE（Graph Engine）—— 算子融合、内存规划、硬件级优化
    ↓
昇腾 NPU Kernel 执行
```

### 13.3 两大加速收益

与 CUDA Graph 类似，ACL Graph 提供两类加速：

1. **框架开销消除**：Host 侧 Python/C++ 逐算子 dispatch 开销归零，整图作为单一 NPU 任务提交
2. **算子融合（Kernel Fusion）**：GE 引擎在图编译阶段将多个算子融合为更少的 kernel，减少 kernel 启动次数和内存带宽消耗——这是 CUDA Graph 本身**不具备**的额外收益

### 13.4 TorchAir 的多种图模式

TorchAir 通过 `torch.compile(backend="npu_backend")` 暴露多种图构建模式：

| 模式 | 配置键 | 描述 |
|------|--------|------|
| `reduce-overhead` | aclgraph | 捕获-回放模式，类比 CUDA Graph。目前为实验特性 |
| `max-autotune` | GeConcreteGraph | 完全静态 shape，最大优化，使用 GE Concrete Graph 路径 |
| 默认 | eager-like torchair | 图被编译但每次都重新提交，无完整捕获 |

---

## 十四、ACL Graph 与 CUDA Graph 的核心区别

### 14.1 最关键差异：动态 Shape 支持

这是两者最重要的架构差异：

**CUDA Graph（NVIDIA）**：
- 严格静态：每个捕获的图绑定到**精确的 tensor shape 和内存地址**
- 任何 batch size 或序列长度变化都需要重新捕获
- vLLM 通过预捕获多个 batch size（约 67 个）并做 padding 来应对

**ACL Graph（昇腾）**：
- **原生支持在声明范围内的动态 shape**
- 图捕获的是**算子拓扑序列**，允许特定维度（batch size、序列长度）在声明的 min/max 范围内自由变化
- 一张图可覆盖一定范围的 batch size，无需为每个 size 单独捕获

```
CUDA Graph：需要 N 张图覆盖 N 个 batch size
ACL Graph：通常 1 张图可覆盖一个 batch size 范围（bucket group）
```

### 14.2 综合对比表

| 维度 | CUDA Graph（NVIDIA） | ACL Graph（昇腾） |
|------|---------------------|-----------------|
| **实现机制** | CUDA stream capture，录制 GPU 命令序列 | GE 图引擎，捕获 NPU 任务拓扑 |
| **Shape 灵活性** | 每个捕获图要求完全静态 shape | 支持在声明范围内动态变化 |
| **Batch size 处理** | 每个 size 单独捕获，运行时 padding 对齐 | 可在一张图内处理一定范围的 batch size |
| **序列长度处理** | 需要分档捕获 | 在 min/max 范围内原生动态，无需分档 |
| **接入方式** | `torch.cuda.CUDAGraph()` / `torch.compile(mode="reduce-overhead")` | `torch.compile(backend="npu_backend")` via torchair；或直接 aclgraph API |
| **算子融合** | 有限（由 CUDA 编译器单独处理） | 是——GE 引擎在图编译阶段融合算子 |
| **Host 开销消除** | 是 | 是 |
| **内存** | 预分配固定 buffer | 预分配；buffer 溢出是已知问题 |
| **成熟度** | 生产级，广泛部署 | 实验性到生产级（因后端版本而异） |
| **vLLM 支持** | 默认且稳定 | 支持，仅限 V1 引擎，持续稳定化 |
| **动态 shape 代价** | 需要重新捕获（开销大） | 范围内变化免费；超出范围才触发重新编译 |

---

## 十五、AscendTurboGraph 与 aclgraph 的区别

华为有两种关联但不同的图模式技术：

### aclgraph（基于 TorchAir）
- 通过 `torch.compile(backend="npu_backend")` 实现
- FX 图追踪 → GE 图编译 → NPU 执行
- vllm-ascend 中大多数模型使用的主要机制

### AscendTurboGraph
- 华为云文档描述为**基于 Capture-Replay 架构的 Host 图**
- 专门设计用于消除 Host 瓶颈，原生支持动态 shape
- 独立于 PyTorch `torch.compile` 路径——在更底层捕获完整 Host 图提交序列
- 用于 ModelArts/ModelArts Pro 推理优化后端

---

## 十六、vllm-ascend 图模式的三代演进

vllm-ascend 经历了三个图模式后端的演进，是理解当前状态的关键背景：

### 第一代：Torchair 图模式（**已废弃**）
- 通过 `torch.compile(backend="npu_backend")` + `torchair_graph_config` 配置
- 为 DeepSeek 等模型提供了初始图模式支持
- **官方废弃**（计划 2026 年 Q1 移除）

```python
# 旧配置方式（不推荐）
{
    "torchair_graph_config": {
        "enabled": True,
        "mode": "reduce-overhead"
    }
}
```

### 第二代：ACL Graph（**当前主力**）
- 更底层的捕获-回放机制，不经过 `torch.compile` / FX 图追踪
- 直接捕获 NPU 任务序列
- 现为 vllm-ascend V1 引擎的主要图模式
- 通过 `enforce_eager=False` 启用（不设置 `enforce_eager=True` 即可）

### 第三代：npugraph_ex（**新兴默认**）
- 基于 ACL Graph 的**上层优化**，在 FX 图层面增加 NPU 友好的算子融合
- 融合顺序：FX 图级别融合 → 降级到 ACL Graph 捕获-回放
- 正在成为默认编译后端（RFC #6214）

```
演进路线：
torchair（已废弃）→ aclgraph（当前主力）→ npugraph_ex（新兴默认）
```

最终架构：
```
FX Graph（torch.compile 追踪）
    ↓ npugraph_ex（NPU 特定 FX 层融合）
    ↓ ACL Graph（捕获-回放，消除 Host 开销）
    ↓ 昇腾 NPU 硬件
```

---

## 十七、vllm-ascend 配置与已知问题

### 17.1 启用图模式

图模式**仅在 V1 引擎**下可用：

```python
# 启用图模式（默认）
llm = LLM(model="...", enforce_eager=False)

# 禁用图模式（退回 eager）
llm = LLM(model="...", enforce_eager=True)
```

### 17.2 Batch Size 处理

vllm-ascend 使用 **token 维度 padding** + **图尺寸分桶**：
- 捕获阶段：对一组预设的 token 数量（而非 batch size，因为 V1 使用 token packing）捕获图
- 推理时：实际 token 数 padding 到最近的已捕获图尺寸

对于 batch size 维度：使用**分桶（bucketing）**方式，为 1、2、4、8、16 等分别捕获图，运行时 padding 到最近的桶。
对于序列长度维度：ACL Graph 原生支持动态范围，**无需分桶**。

### 17.3 已知问题分类（RFC #7599）

ACL Graph 存在 5 类失败模式：

| 类别 | 描述 |
|------|------|
| **编译失败** | torch.compile / FX 追踪对某些模型架构失败 |
| **捕获失败** | 图捕获阶段失败（如 buffer 溢出、不支持的算子） |
| **回放失败** | 捕获的图在推理时失败（shape 不匹配、NPU 运行时错误） |
| **精度下降** | 图模式与 Eager 模式产生不同数值结果 |
| **性能回退** | 图模式比 Eager 模式慢（分桶选择不合理时可能发生） |

### 17.4 典型 Bug 与解决方案

| 问题 | 修复 |
|------|------|
| MTP + 大 EP 配置导致图捕获 buffer 溢出 | 显式设置 `enforce_eager=True` 或限制捕获尺寸 |
| DP 场景全图模式精度问题 | 已在后续版本修复 |
| NPU 流（stream）耗尽 | 更新图捕获尺寸计算逻辑，减少捕获数量 |
| 多卡（TP）token-wise padding 不正确 | 修复多卡图模式的 padding 逻辑 |
| 上游 vLLM 变更破坏 padding 逻辑 | 每次 vLLM 升级后需验证图模式兼容性 |

---

## 参考资料

**CUDA Graph（NVIDIA/vLLM）**
- [vLLM 官方文档：CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [vLLM 官方文档：Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM 官方文档：Vision Encoder CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/)
- [How CUDA Graphs make vLLM think faster - LinkedIn](https://www.linkedin.com/pulse/efficiently-serving-llms-part-4-how-cuda-graphs-make-vllm-thomas-4ofuc)
- [vLLM CUDA 图和 eager 模式 - 知乎](https://zhuanlan.zhihu.com/p/1989734188635689446)
- [vllm v1 中的几种优化模式 - 知乎](https://zhuanlan.zhihu.com/p/1960475172206281384)
- [51CTO：揭开 vLLM 推理系统实现高效吞吐的秘籍](https://www.51cto.com/article/828072.html)
- [RFC: Lazy CUDA Graph Capture #20098](https://github.com/vllm-project/vllm/issues/20098)
- [RFC: Token Padding Strategy for FP8 GEMM #30717](https://github.com/vllm-project/vllm/issues/30717)

**ACL Graph（昇腾）**
- [vllm-ascend：如何使用 ACL Graph](https://docs.vllm.ai/projects/ascend/zh-cn/v0.11.0-dev/developer_guide/feature_guide/ACL_Graph.html)
- [vllm-ascend：npugraph_ex 开发者指南](https://docs.vllm.ai/projects/ascend/en/v0.18.0/developer_guide/feature_guide/npugraph_ex.html)
- [vllm-ascend Release Notes](https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/release_notes.html)
- [华为云：LLM 推理图模式（AscendTurboGraph）](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5910020.html)
- [昇腾文档：TorchAir 图模式使用（torchair）](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/torchairuseguide/torchair_0008.html)
- [vLLM 中 ACL Graph 的应用解析 - 知乎](https://zhuanlan.zhihu.com/p/1986741243808616589)
- [深入解析 Ascend 的 aclgraph - CSDN](https://blog.csdn.net/wasm7browser/article/details/154629625)
- [RFC: Modular Unit Testing for ACL Graph #7599](https://github.com/vllm-project/vllm-ascend/issues/7599)
- [RFC: npugraph_ex backend #4715](https://github.com/vllm-project/vllm-ascend/issues/4715)
