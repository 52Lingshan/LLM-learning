# MLP / FFN（Feed-Forward Network）

Transformer 中每一层包含两个核心子模块：Attention 和 FFN。  
Attention 负责"词与词之间交换信息"，FFN 负责"对每个词独立做非线性变换，增加表达力"。

---

## 一、标准 FFN（原始 Transformer）

```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
```

只有两个线性层：
- W₁：升维（d_model → d_ff，通常 4 倍，如 4096 → 16384）
- W₂：降维（d_ff → d_model）

中间用 ReLU 激活，引入非线性。

---

## 二、SwiGLU FFN（LLaMA / GLM-2+ 使用）

### 公式

```
FFN(x) = W_down · ( SiLU(W_gate · x) ⊙ W_up · x )
```

- `⊙` 表示逐元素相乘（element-wise multiplication）
- `SiLU(x) = x · sigmoid(x)`，是 ReLU 的平滑版本

### 三个矩阵的角色

| 名称 | 形状（以 LLaMA-7B 为例） | 作用 | 类比 |
|------|--------------------------|------|------|
| **W_gate** | (4096 → 11008) | 产生门控信号，经 SiLU 后决定每个特征放行多少 | 质检员：给每个特征打分 0~1 |
| **W_up** | (4096 → 11008) | 升维，产生实际的特征内容 | 生产线：加工原材料为半成品 |
| **W_down** | (11008 → 4096) | 降维回原始维度 | 打包员：压缩回标准尺寸 |

### 命名含义

- **gate**（门）：控制信息通过量，0 = 关门，1 = 全开
- **up**（上投影）：维度从小变大（4096 → 11008）
- **down**（下投影）：维度从大变小（11008 → 4096）

### 数据流

```
输入 x（4096维）
    │
    ├──→ W_gate · x ──→ SiLU ──→ "门控信号"（哪些特征该放行）
    │                               │
    ├──→ W_up · x ─────────────→ "原始内容"（升维后的特征）
    │                               │
    │                               ⊙ 逐元素相乘
    │                               │
    │                               ▼
    └──────────────── W_down ← "筛选后的结果"（降维回原尺寸）
                        │
                        ▼
                      输出（4096维）
```

### 对应 LLaMA 源码

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # W_gate
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)  # W_up
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # W_down

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x))   # 门控信号
            *                            # ⊙ 逐元素相乘
            self.up_proj(x)              # 原始内容
        )
```

### 具体数值例子

假设输入 2 维，中间 3 维：

```
x = [0.5, -0.3]

W_gate · x = [1.2, -0.8, 0.3]
SiLU(...)  = [0.88, -0.27, 0.16]   ← 第1个特征放行88%，第2个抑制

W_up · x   = [0.6, 1.5, -0.4]     ← 实际内容

逐元素相乘  = [0.88×0.6, -0.27×1.5, 0.16×(-0.4)]
            = [0.53, -0.40, -0.06]  ← 被门控筛选后的结果

W_down · [0.53, -0.40, -0.06] → [最终输出，降回2维]
```

---

## 三、nn.Linear 的本质

```python
nn.Linear(4096, 11008, bias=False)
```

内部就是一个形状为 (11008 × 4096) 的权重矩阵 W，前向计算：

```
输出 = x · Wᵀ

x:    (seq_len, 4096)     ← 输入
Wᵀ:   (4096, 11008)       ← 权重转置
输出:  (seq_len, 11008)    ← 升维结果
```

`bias=False` 表示没有偏置项，纯矩阵乘法。LLaMA 全模型无 bias。

---

## 四、工程优化：gate_proj 和 up_proj 可合并

gate_proj 和 up_proj 接收相同的输入 x，可以合并为一次矩阵乘法再拆分：

```python
# 分开投影（源码写法）
gate_out = F.silu(self.gate_proj(x))
up_out   = self.up_proj(x)
output   = self.down_proj(gate_out * up_out)

# 合并投影（等价优化）
linear_combine = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
combined = linear_combine(x)
gate_out, up_out = combined.chunk(2, dim=-1)
output = self.down_proj(F.silu(gate_out) * up_out)
```

合并的好处：减少一次 kernel launch，GPU 利用率更高。  
两者数学上完全等价（`torch.allclose` 验证为 True）。

---

## 五、为什么 intermediate_size 是 ~8/3 倍

标准 FFN 用 2 个矩阵，中间维度 = 4 × hidden_size，参数量：
```
2 × (d × 4d) = 8d²
```

SwiGLU 用 3 个矩阵，为保持总参数量相当：
```
3 × (d × intermediate) = 8d²
→ intermediate ≈ 8d/3 ≈ 2.67d
```

LLaMA-7B: hidden=4096, intermediate=11008 ≈ 2.69 × 4096，符合这个比例。

---

## 六、SwiGLU vs 标准 FFN 对比

| | 标准 FFN | SwiGLU FFN |
|---|---------|------------|
| 矩阵数量 | 2 个 | 3 个 |
| 激活函数 | ReLU / GELU | SiLU（在门控分支上） |
| 门控机制 | 无 | 有（gate 分支控制 up 分支的通过量） |
| 中间维度 | 4 × d_model | ~8/3 × d_model |
| 总参数量 | 8d² | 约 8d²（设计上对齐） |
| 效果 | 基线 | 更好（PaLM、LLaMA 论文验证） |

### 核心优势

gate 和 up 看的是同一个输入 x，但通过不同的权重矩阵学到不同的东西：
- **gate 学"什么重要"**（注意力/筛选）
- **up 学"内容是什么"**（特征提取）

这种分工让模型比单纯 ReLU 有更强的表达力。
