# Agent Teams 资料整理

> 整理时间：2026-04-08
> 目标读者：有 Python 基础、了解 LLM 基本概念，但刚接触 Multi-Agent 系统

---

## 什么是 Agent Teams？

**一句话解释：** Agent Teams 就是让多个 AI 一起分工合作完成任务，就像一个软件开发团队——有架构师、开发者、测试员，各司其职，而不是所有事都让同一个 AI 来做。

**为什么需要多个 Agent？**

| 问题 | 单 Agent 的局限 | Agent Teams 的解法 |
|------|----------------|-------------------|
| 任务太复杂 | 上下文窗口撑不住 | 每个 Agent 只关注自己那一块 |
| 需要并行 | 只能一步步来 | 多个 Agent 同时工作 |
| 需要互相检查 | 自己很难发现自己的错误 | Reviewer Agent 专门挑毛病 |
| 专业分工 | 一个 Agent 很难每方面都精通 | 每个 Agent 专注一个角色 |

---

## 一、四大主流框架横向对比

> 当前最流行的 4 个 Multi-Agent 框架：LangGraph、CrewAI、AutoGen、Swarm

### 1.1 框架定位一览

| 框架 | 开发者 | 核心理念 | 适合谁用 | 上手难度 |
|------|--------|----------|---------|---------|
| **LangGraph** | LangChain 团队 | 把工作流画成"流程图"，精确控制每一步 | 要上生产系统的工程师 | ★★★☆☆ 中等 |
| **CrewAI** | CrewAI Inc | 像组建公司团队一样，给每个 Agent 分配角色 | 快速验证想法的开发者 | ★★☆☆☆ 简单 |
| **AutoGen** | 微软 | 多个 AI 相互对话、讨论，类似群聊 | 研究人员、需要代码自动执行 | ★★☆☆☆ 简单 |
| **Swarm** | OpenAI | 极简设计，Agent 之间直接"交接"任务 | 学习原理、快速原型 | ★☆☆☆☆ 最简单 |

---

### 1.2 LangGraph：把工作流画成图

**核心思想**

把整个 AI 工作流程画成一张**有向图**：
- **节点（Node）** = 每个干活的 Agent 或函数
- **边（Edge）** = 任务流向，支持条件判断（类似 if/else）
- **状态（State）** = 所有节点共享的数据，类似"黑板"，大家都可以读写

**架构图**

```
用户输入
    │
    ▼
[START]
    │
    ▼
[架构师 Node] ──→ 生成架构设计，写入 State
    │
    ▼
[开发者 Node] ──→ 读取架构，写代码，写入 State
    │
    ▼
[测试员 Node] ──→ 读取代码，运行测试，写入 State
    │
    ▼
[审查者 Node]
    │
    ├── "代码有问题" ──→ 回到 [开发者 Node]（条件循环）
    │
    └── "没问题" ──→ [END]
```

**代码示例（项目开发工作流）**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# ① 定义"黑板"——所有 Agent 共享的状态
class DevState(TypedDict):
    requirements: str           # 用户需求
    architecture: str           # 架构设计结果
    code: dict[str, str]        # 文件名 → 代码内容
    test_results: list[dict]    # 测试结果
    review_comments: list[str]  # 审查意见
    iteration: int              # 当前迭代轮次（防止无限循环）

# ② 定义节点函数
def architect(state: DevState) -> dict:
    """架构师：分析需求，输出架构方案"""
    design = llm.invoke(f"你是架构师，请设计系统架构：{state['requirements']}")
    return {"architecture": design.content}

def coder(state: DevState) -> dict:
    """开发者：根据架构写代码"""
    code = llm.invoke(f"你是开发者，请根据以下架构编写代码：{state['architecture']}")
    return {"code": {"main.py": code.content}, "iteration": state["iteration"] + 1}

def tester(state: DevState) -> dict:
    """测试员：生成并运行测试"""
    # 真实场景中这里会实际运行代码
    return {"test_results": [{"test": "test_login", "passed": True}]}

def reviewer(state: DevState) -> dict:
    """审查者：检查代码质量"""
    review = llm.invoke(f"你是代码审查员，请审查代码：{state['code']}")
    return {"review_comments": [review.content]}

# ③ 条件路由：审查后决定是否继续迭代
def should_iterate(state: DevState) -> str:
    if state["iteration"] >= 3:             # 最多迭代 3 次，防止死循环
        return "end"
    if any("需要修改" in c for c in state["review_comments"]):
        return "coder"                      # 有问题 → 回去改代码
    return "end"                            # 没问题 → 结束

# ④ 构建工作流图
graph = StateGraph(DevState)
graph.add_node("architect", architect)
graph.add_node("coder", coder)
graph.add_node("tester", tester)
graph.add_node("reviewer", reviewer)

# 固定边：架构 → 开发 → 测试 → 审查
graph.set_entry_point("architect")
graph.add_edge("architect", "coder")
graph.add_edge("coder", "tester")
graph.add_edge("tester", "reviewer")

# 条件边：审查后根据结果决定下一步
graph.add_conditional_edges(
    "reviewer",
    should_iterate,
    {"coder": "coder", "end": END}  # 映射：返回值 → 目标节点
)

# ⑤ 运行
app = graph.compile()
result = app.invoke({
    "requirements": "实现用户登录系统",
    "iteration": 0
})
```

**适合场景**

```
✅ 推荐用 LangGraph:
├── 工作流有复杂的条件分支（比如"如果测试失败就重试"）
├── 需要循环迭代直到满足条件
├── 需要精确控制每一步的执行顺序
├── 生产环境，需要可观测性和错误追踪
└── 需要保存进度（断点续传）

❌ 不适合的场景:
├── 简单的一问一答任务
└── 快速原型验证（配置较繁琐）
```

---

### 1.3 CrewAI：像组建公司团队

**核心思想**

把 Agent 想象成"员工"：
- 每个 Agent 有**角色**（软件架构师）、**目标**（设计模块化架构）、**背景**（10年经验）
- 任务可以有**依赖关系**（开发必须等架构设计完）
- 支持**顺序执行**和**层级执行**（有 Manager 统筹）

**架构图**

```
Crew（团队）
│
├── Agent: 架构师 ──→ Task: 设计架构 ──→ 输出：架构文档
│
├── Agent: 开发者 ──→ Task: 写代码   ──→ 依赖: 架构文档
│                                       输出：代码
├── Agent: 测试员 ──→ Task: 测试代码 ──→ 依赖: 代码
│                                       输出：测试报告
└── Agent: 审查员 ──→ Task: 代码审查 ──→ 依赖: 代码 + 测试报告
                                        输出：审查报告
```

**代码示例**

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# ① 定义团队成员（给每人设定角色和背景，让 LLM 更好地扮演该角色）
architect = Agent(
    role="软件架构师",
    goal="设计模块化、可扩展的系统架构",
    backstory="10年经验的资深架构师，擅长大型分布式系统设计",
    llm=llm,
    verbose=True       # 打印执行过程，便于调试
)

coder = Agent(
    role="全栈开发者",
    goal="编写高质量、可维护的功能代码",
    backstory="5年经验的全栈工程师，熟悉 Python 和 TypeScript",
    llm=llm,
    verbose=True
)

tester = Agent(
    role="测试工程师",
    goal="确保代码质量，覆盖边界情况",
    backstory="专注于自动化测试，擅长发现隐藏 bug",
    llm=llm,
    verbose=True
)

reviewer = Agent(
    role="代码审查员",
    goal="确保代码符合最佳实践和安全规范",
    backstory="技术 Lead，注重代码可读性和安全性",
    llm=llm,
    verbose=True
)

# ② 定义任务（关键：context 参数指定依赖哪些前置任务）
design_task = Task(
    description="为用户登录系统设计完整架构，包括数据模型、API 接口和安全方案",
    agent=architect,
    expected_output="包含模块划分、数据流和接口定义的架构设计文档"
)

code_task = Task(
    description="根据架构文档实现用户登录功能，包括注册、登录、JWT Token 生成",
    agent=coder,
    context=[design_task],      # 告诉这个 task：需要等 design_task 完成后才能开始
    expected_output="完整的 Python 实现代码，包含注释"
)

test_task = Task(
    description="为登录功能编写单元测试和集成测试",
    agent=tester,
    context=[code_task],
    expected_output="测试文件及测试覆盖率报告"
)

review_task = Task(
    description="审查代码质量，检查安全漏洞（如 SQL 注入、密码明文存储等）",
    agent=reviewer,
    context=[code_task, test_task],    # 同时依赖代码和测试报告
    expected_output="代码审查报告，列出问题和改进建议"
)

# ③ 组建团队并执行
dev_crew = Crew(
    agents=[architect, coder, tester, reviewer],
    tasks=[design_task, code_task, test_task, review_task],
    process=Process.sequential,     # 顺序执行（也可以用 Process.hierarchical 让 Manager 统筹）
    verbose=True
)

result = dev_crew.kickoff()
print(result)
```

**与 LangGraph 的核心区别**

| 维度 | CrewAI | LangGraph |
|------|--------|-----------|
| 控制粒度 | 框架自动处理顺序 | 开发者精确控制每条边 |
| 循环迭代 | 有限支持 | 原生支持（条件边） |
| 上手速度 | 快，10 分钟能跑起来 | 慢，需要理解图的概念 |
| 适合阶段 | 原型验证、业务清晰的场景 | 生产系统 |

---

### 1.4 AutoGen：多 Agent 群聊对话

**核心思想**

模拟人类团队协作方式——通过**对话**来协作，而不是预定义流程：
- Agent 之间互相发消息，像群聊一样
- 有特殊的 `UserProxyAgent`，可以**自动执行代码**，并把结果反馈给其他 Agent
- 天然支持人工介入（Human-in-the-Loop）

**对话模式**

```
模式1：两个 Agent 对话
┌─────────────┐         ┌─────────────┐
│   AI 助手   │ ◀─────▶ │  UserProxy  │
│ (写代码)     │         │ (执行代码)   │
└─────────────┘         └─────────────┘
  ↑ AI 写代码 → UserProxy 执行 → 结果反馈 → AI 修改 → 循环

模式2：多 Agent 群聊
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Researcher│  │ Analyst  │  │  Writer  │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    ▼
             GroupChatManager
             (决定谁下一个发言)
```

**代码示例**

```python
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

config_list = [{"model": "gpt-4", "api_key": "your-key"}]

# ① 定义 Agent
assistant = AssistantAgent(
    name="Assistant",
    system_message="你是全能助手，可以设计架构、写代码和审查。",
    llm_config={"config_list": config_list}
)

# UserProxyAgent 是 AutoGen 的特色：它代表"用户"，但可以自动执行代码
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",   # "NEVER"=全自动, "ALWAYS"=每步都问人, "TERMINATE"=只在结束问
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",   # 代码保存和执行的目录
        "use_docker": False,    # 是否用 Docker 隔离执行（生产环境建议 True）
        "timeout": 60,
    }
)

# ② 两个 Agent 对话（自动调试流程）
user_proxy.initiate_chat(
    assistant,
    message="""
    这段代码有 bug，请帮我找出并修复：

    def calculate_average(numbers):
        total = 0
        for n in numbers:
            total += n
        return total / len(numbers)   # 当 numbers 为空时会除零错误！

    print(calculate_average([]))
    """
)
# AutoGen 会自动：
# 1. AI 分析 bug → 写修复代码
# 2. UserProxy 执行修复后的代码
# 3. 若执行失败 → AI 继续修改 → 再执行
# 4. 直到代码正确运行

# ③ 多 Agent 群聊示例
researcher = AssistantAgent(
    name="Researcher",
    system_message="你是研究员，负责收集和整理信息，完成后说'信息收集完毕'",
    llm_config={"config_list": config_list}
)

analyst = AssistantAgent(
    name="Analyst",
    system_message="你是分析师，基于研究员的信息做深入分析，完成后说'分析完毕'",
    llm_config={"config_list": config_list}
)

writer = AssistantAgent(
    name="Writer",
    system_message="你是技术作家，把分析师的结论写成清晰的报告",
    llm_config={"config_list": config_list}
)

groupchat = GroupChat(
    agents=[researcher, analyst, writer],
    messages=[],
    max_round=20,
    speaker_selection_method="round_robin"  # 轮流发言；也可用 "auto" 让 LLM 决定谁发言
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

researcher.initiate_chat(
    manager,
    message="研究 2025 年大语言模型在代码生成领域的最新进展，输出一份完整报告。"
)
```

**AutoGen 最大的独特优势：代码自动执行**

```python
# 这是其他框架很少原生支持的能力：
# AI 写了代码 → 自动执行 → 看到报错 → 自动修改 → 再执行 → 直到成功

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True,     # 生产推荐：用 Docker 防止代码乱搞系统
        "timeout": 60,
        "last_n_messages": 3    # 只看最近 3 条消息里的代码（节省上下文）
    }
)
```

---

### 1.5 Swarm：最轻量的任务交接

**核心思想**

超级简单：Agent 之间通过**函数返回**来"交接"任务，谁处理完了就把控制权传给下一个 Agent。

```
用户请求 → Agent A（处理完后调用 transfer_to_b）→ Agent B → Agent C → 结束

本质上就是：函数返回下一个 Agent 对象，框架就切换过去
```

**代码示例**

```python
from swarm import Swarm, Agent

client = Swarm()

# 交接函数：返回下一个要接手的 Agent
def transfer_to_coder():
    return coder

def transfer_to_reviewer():
    return reviewer

# Agent 通过 functions 参数声明"我可以把任务交给谁"
architect = Agent(
    name="Architect",
    instructions="你是架构师，设计完系统架构后，将任务交给开发者继续实现。",
    functions=[transfer_to_coder]  # 只能交给 coder
)

coder = Agent(
    name="Coder",
    instructions="你是开发者，实现代码后，将任务交给审查员做代码审查。",
    functions=[transfer_to_reviewer]  # 只能交给 reviewer
)

reviewer = Agent(
    name="Reviewer",
    instructions="你是审查员，审查代码质量并给出最终评价。"
    # 没有 functions，表示任务到此终止
)

# 运行（从 architect 开始）
response = client.run(
    agent=architect,
    messages=[{"role": "user", "content": "实现一个用户登录系统"}]
)

print(response.messages[-1]["content"])
```

**注意：Swarm 是 OpenAI 发布的教学/实验性项目，不推荐用于生产环境。**

---

### 1.6 框架能力全面对比

#### 功能评分

| 能力 | LangGraph | AutoGen | CrewAI | Swarm |
|------|-----------|---------|--------|-------|
| 状态管理 | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| 流程控制（分支/循环） | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |
| 并行执行 | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| 代码自动执行 | 需手动集成 | ★★★★★（内置）| 需手动集成 | 需手动集成 |
| 人机协作 | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| 可观测性/调试 | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★☆☆☆☆ |
| 生产就绪 | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| 上手难度（越低越好）| 中等 | 简单 | 简单 | 最简单 |

#### 性能参考数据

| 指标 | LangGraph | AutoGen | CrewAI | Swarm |
|------|-----------|---------|--------|-------|
| 冷启动时间 | ~500ms | ~1s | ~300ms | ~100ms |
| 内存占用 | 中等 | 较高 | 中等 | 很低 |
| Token 效率 | 高（状态精确控制）| 中（对话上下文累积）| 高 | 高 |

#### 选型决策树

```
你的任务是什么类型？
│
├── 需要复杂流程控制（条件、循环、回滚）
│   └── 选 LangGraph
│
├── 需要代码自动执行 + AI 自动修复
│   └── 选 AutoGen
│
├── 角色分工明确，流程相对固定
│   └── 选 CrewAI
│
└── 只是学习概念 / 快速原型
    └── 选 Swarm（但别用于生产）
```

---

## 二、Claude Code 原生 Agent Teams

> 这是 Claude Code（Anthropic 的 CLI 工具）内置的 Agent 协作功能，**不需要写代码**，直接通过自然语言指令触发。

### 2.1 是什么？怎么工作的？

Claude Code 支持在一次对话中自动 **spawn（生成）** 多个子 Agent，这些子 Agent 并行工作，最后由主会话（Team Lead）汇总结果。

```
你的指令："帮我开发一个用户管理系统"
              │
              ▼
┌─────────────────────────────────────────────────────┐
│                  Team Lead（主会话）                   │
│                                                     │
│  步骤1：分析任务，识别可并行的子任务                    │
│  步骤2：创建共享任务列表（所有 Agent 都能看到）          │
│  步骤3：spawn 子 Agent，分配任务                       │
│  步骤4：等所有子 Agent 完成后，汇总验证                 │
└──────────────────────┬──────────────────────────────┘
                       │ spawns（派生）
          ┌────────────┼────────────┐
          ▼            ▼            ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  Agent 1 │  │  Agent 2 │  │  Agent 3 │
  │  架构师  │  │  开发者  │  │  测试员  │
  │          │  │          │  │          │
  │独立上下文 │  │独立上下文 │  │独立上下文 │
  │（互不干扰）│  │          │  │          │
  └──────────┘  └──────────┘  └──────────┘
          │            │            │
          └────────────┼────────────┘
                       ▼
              结果聚合：Team Lead 检查一致性、整合输出
```

**"独立上下文"意味着什么？**

每个子 Agent 都有自己的对话窗口，它们不会看到彼此的对话内容。好处是可以并行、互不干扰；代价是 Token 消耗更高（多个窗口 × Token 数量）。

---

### 2.2 如何触发 Agent Teams

**方法一：直接明说**

```
"用 agent team 帮我完成这个项目开发任务，并行处理各个模块"

"spawn 多个 agents 来同时处理这些相互独立的任务"
```

**方法二：指定角色和分工**

```
"创建一个 agent team 来开发一个博客系统，需要包含：
- 架构师：设计整体架构和数据库 Schema
- 开发者A：实现用户认证模块（注册/登录/JWT）
- 开发者B：实现文章管理模块（CRUD）
- 测试员：为两个模块分别编写单元测试"
```

**方法三：多角度分析型**

```
"用多个 agent 从不同角度分析这段代码：
- 性能分析 Agent：找出性能瓶颈
- 安全分析 Agent：检查安全漏洞
- 可读性 Agent：评估代码可维护性
最后整合成一份综合评审报告"
```

---

### 2.3 实际使用示例

**场景：开发电商系统**

```
你的输入：
"帮我开发一个简单的电商系统，包含用户模块、商品模块、订单模块，三个模块相互独立，用 agent team 并行开发"

Claude 执行过程：
│
├── Team Lead 分析任务
│   ├── 识别出 3 个可独立并行的模块
│   ├── 创建任务列表（共享给所有 Agent）
│   └── 规划依赖关系（用户模块是基础，其他模块依赖它的 User 类）
│
├── spawn Agent 1：用户模块开发者
│   ├── 设计 User 数据模型（id, username, email, password_hash）
│   ├── 实现注册/登录 API
│   └── 输出：user_module.py + 使用说明
│
├── spawn Agent 2：商品模块开发者（同步进行）
│   ├── 设计 Product 数据模型
│   ├── 实现商品 CRUD API
│   └── 输出：product_module.py
│
├── spawn Agent 3：订单模块开发者（同步进行）
│   ├── 设计 Order 数据模型
│   ├── 实现下单/查询/取消 API
│   └── 输出：order_module.py
│
└── Team Lead 汇总
    ├── 检查三个模块的接口是否一致（如 User ID 格式是否统一）
    ├── 整合成完整项目结构
    └── 输出：完整项目代码 + README
```

---

### 2.4 与传统框架对比

| 维度 | Claude 原生 Agent Teams | LangGraph / AutoGen / CrewAI |
|------|------------------------|------------------------------|
| 配置成本 | 零配置，自然语言触发 | 需要写代码，定义节点/角色 |
| 灵活性 | 中等（Claude 自主决策）| 高（开发者完全控制） |
| 可控性 | 较低（黑盒）| 高（每步可审计） |
| 适合场景 | 探索性任务、快速开发 | 生产系统、需要精确控制 |
| Token 消耗 | 高（多 Agent 多窗口）| 可优化（精确控制）|

---

### 2.5 使用建议

**适合用 Agent Teams 的场景**

```
✅ 复杂多模块项目（模块之间相对独立）
✅ 需要并行处理的任务（各部分互不依赖）
✅ 多角度分析（性能审查 + 安全审查 + 代码审查同时进行）
✅ 大型代码重构（不同文件可以独立处理）
```

**不适合的场景**

```
❌ 简单单步任务（"帮我写个 hello world"，用 Agent Team 是浪费）
❌ 强依赖串行任务（B 必须等 A 全部完成才能开始，并行没意义）
❌ 对 Token 成本敏感的场景（多 Agent 消耗成倍增加）
```

**注意事项**

- **Token 成本更高**：每个子 Agent 有独立上下文，总消耗 ≈ 单 Agent 的 N 倍
- **任务边界要清晰**：如果任务描述模糊，Agent 之间容易产生"接口不一致"问题
- **关键决策建议人工确认**：Agent 自主决策可能出错，重要的架构决策最好人工介入

---

## 三、MCP 协议：Agent 间的"通用插座"

> MCP（Model Context Protocol）是 Anthropic 提出的开放标准，让 AI 能以统一方式访问各种工具和数据源。Multi-Agent 系统用它来共享上下文。

### 3.1 MCP 是什么？用一个比喻解释

想象一下：

- **没有 MCP**：每个 AI 工具都有自己的"私有接口"，Agent A 的工具不能被 Agent B 使用，就像电器插头各种形状，到国外还得买转换器
- **有了 MCP**：统一标准接口，任何 Agent 都能访问任何 MCP Server 提供的工具和数据，就像 USB-C 接口标准化

### 3.2 MCP 架构

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Host（宿主应用）                    │
│  Claude Desktop / VS Code / 你的自定义应用               │
└───────────────────────┬─────────────────────────────────┘
                        │ 通过 MCP 协议通信
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    MCP Server（服务端）                    │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Resources  │  │    Tools    │  │   Prompts   │     │
│  │（只读数据）  │  │  （可执行操作）│  │  （提示模板）│     │
│  │             │  │             │  │             │     │
│  │ 文件内容     │  │ 创建文件     │  │ 常用提示词   │     │
│  │ 数据库记录   │  │ 调用 API    │  │ 工作流模板   │     │
│  │ API 响应    │  │ 执行命令     │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    外部服务                               │
│    数据库  │  第三方 API  │  文件系统  │  云服务           │
└─────────────────────────────────────────────────────────┘
```

### 3.3 MCP 在 Multi-Agent 中的应用

**核心场景：Agent 间共享状态**

```python
# 实现一个 MCP Server 作为 Agent 之间的"共享黑板"
from mcp import MCPServer, Resource, Tool

class SharedMemoryServer(MCPServer):
    """提供 Agent 间共享上下文的 MCP 服务"""

    def __init__(self):
        self.shared_memory = {}  # 内存中的共享状态

    @Resource("memory://{key}")
    def get_memory(self, key: str):
        """读取共享数据（只读操作）"""
        return self.shared_memory.get(key, None)

    @Tool("set_memory")
    def set_memory(self, key: str, value: str):
        """写入共享数据"""
        self.shared_memory[key] = value
        return f"已保存：{key} = {value}"

# Agent A（架构师）完成后，把架构写入共享内存
mcp_client.call_tool("set_memory", {
    "key": "architecture_design",
    "value": "用户模块：使用 JWT；数据库：PostgreSQL；..."
})

# Agent B（开发者）读取架构，基于它写代码
architecture = mcp_client.get_resource("memory://architecture_design")
```

### 3.4 Agent 间通信方式对比

| 通信方式 | 特点 | 适合场景 |
|---------|------|---------|
| **MCP** | 标准化协议，跨框架、跨语言都能用 | Claude 生态、需要工具集成 |
| **LangGraph State** | 类型安全，与 LangGraph 深度绑定 | LangGraph 工作流内部 |
| **AutoGen Message** | 以对话消息传递信息，灵活但非结构化 | AutoGen 群聊 |
| **自定义协议** | 完全可控，但需自己实现 | 特殊需求、高性能要求 |

---

## 四、2025 年 Multi-Agent 技术趋势

### 趋势 1：标准化协议（MCP 成为事实标准）
- 越来越多的工具支持 MCP，Agent 可以"即插即用"各种能力
- 不同框架的 Agent 可以通过 MCP 互操作

### 趋势 2：持久化和断点续传
- 长达数小时的任务需要保存进度
- 任务失败可以从检查点恢复，而不是从头开始

### 趋势 3：可观测性成为标配
- 每一步 Agent 做了什么、花了多少 Token、耗时多少——都要可追踪
- LangSmith（LangChain 的）、AutoGen Studio、LangGraph Studio 等工具提供可视化

### 趋势 4：更自然的人机协作
- Agent 在关键决策点暂停，等人审批
- 支持异步协作：人可以"随时介入"而不是只能在固定节点

### 趋势 5：动态自适应编排
- Agent 数量根据任务动态调整（任务多 → 多派 Agent；任务少 → 减少）
- 智能错误恢复：某个 Agent 失败了，自动重试或换策略

---

## 五、企业级 Multi-Agent 架构参考

> 生产系统不是随便跑几个 Agent 就行，需要考虑安全、扩展、监控。

```
┌─────────────────────────────────────────────────────────────────┐
│                        网关层（入口）                              │
│   API Gateway（限流） │ Auth（鉴权） │ Load Balancer（负载均衡）    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                       编排层（大脑）                               │
│   选择合适的框架：LangGraph / AutoGen / CrewAI                     │
│   负责：任务分解、Agent 调度、结果聚合                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                       Agent 层（执行）                             │
│   Agent 1 │ Agent 2 │ Agent 3 │ Agent 4 │ Agent 5               │
│   （每个 Agent 负责特定角色，相互隔离）                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                       服务层（资源）                               │
│   Memory（记忆） │ Tools（工具） │ LLM API │ Vector DB │ 消息队列    │
└─────────────────────────────────────────────────────────────────┘
```

### 关键技术选型

| 决策点 | 选项 | 建议 |
|--------|------|------|
| **状态存储** | 内存 / Redis / 数据库 | 生产用 Redis（支持分布式）或数据库（持久化）|
| **任务队列** | 同步 / 异步（消息队列）| 长任务（>30秒）一定要用异步 |
| **LLM 调用** | 单一模型 / 多模型组合 | 简单 Agent 用便宜模型，关键 Agent 用强模型 |
| **错误处理** | 重试 / 回滚 / 人工介入 | 关键任务三层保障；最多重试 3 次，失败转人工 |
| **监控告警** | 日志 / 指标 / 链路追踪 | 三者都要，缺一不可 |

---

## 六、项目开发场景：Orchestrator-Workers 模式详解

### 6.1 模式图解

```
                ┌──────────────────────────────────────┐
                │            Orchestrator              │
                │        （任务分解 + 分配 + 汇总）        │
                └──────────────────┬───────────────────┘
                                   │ 分配任务
            ┌──────────┬───────────┼───────────┬──────────┐
            ▼          ▼           ▼           ▼          ▼
       ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
       │ 架构师  │ │ 开发者A │ │ 开发者B │ │  测试员 │ │ 审查员  │
       │Architect│ │ Coder-A │ │ Coder-B │ │ Tester  │ │Reviewer │
       └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
            │           │           │           │           │
            └───────────┴───────────┴───────────┴───────────┘
                                   │ 汇报结果
                                   ▼
                         Orchestrator 整合验证
```

### 6.2 各角色职责

| 角色 | 输入 | 职责 | 输出 |
|------|------|------|------|
| **Orchestrator** | 用户需求 | 分解任务、分配给合适的 Agent、处理依赖、验证整合结果 | 完整交付物 |
| **Architect** | 需求文档 | 模块划分、接口定义、数据模型设计 | 架构设计文档 |
| **Coder** | 任务描述 + 架构文档 | 实现具体功能代码 | 代码文件 |
| **Reviewer** | 代码 | 发现 Bug、安全漏洞、不符合规范的地方 | 审查报告 + 修改意见 |
| **Tester** | 代码 | 编写测试用例，验证功能正确性 | 测试文件 + 覆盖率报告 |
| **Integrator** | 多个模块代码 | 合并代码、解决接口冲突、处理依赖 | 集成后的代码 |

### 6.3 核心挑战与解决方案

**挑战 1：模块间依赖管理**

```
问题：商品模块需要用到用户模块的 User 类型，但两者在并行开发
解决：
- Architect 在分配任务前先定义好"接口契约"（Interface Contract）
- 比如：User = {"id": int, "username": str, "email": str}
- 各 Coder 都基于这个契约开发，互不等待
- Integrator 最后检查实现是否符合契约
```

**挑战 2：代码风格不一致**

```
问题：多个 Coder 同时写代码，命名风格、错误处理方式可能不同
解决：
- 在 Prompt 中明确规范（如：使用 snake_case，异常用 raise 而非 return None）
- Reviewer Agent 专门检查风格一致性
- 使用 linter/formatter 工具（black, eslint）做自动统一
```

**挑战 3：并行开发产生冲突**

```
问题：两个 Coder 都修改了 utils.py 中的同一个函数
解决：
- Architect 在任务分配时明确文件归属（谁负责哪些文件，不允许跨越）
- 共享的工具函数由 Integrator 统一处理
- 定期同步检查点（如每完成 1 个模块就做一次整合）
```

---

## 七、快速选型总结

```
你的需求是什么？
│
├── 想快速验证一个想法，不想写太多代码
│   └── 用 Claude Code 原生 Agent Teams（自然语言触发）
│       或 CrewAI（代码量少，容易上手）
│
├── 需要严格控制流程（条件判断、循环、回滚）
│   └── 用 LangGraph
│
├── 需要 AI 自动写代码、执行代码、修复 Bug（全自动）
│   └── 用 AutoGen
│
├── 只是学习 Multi-Agent 的基本概念
│   └── 用 Swarm（最简单，代码最少）
│
└── 生产系统，需要高可用、可观测、可审计
    └── 用 LangGraph + LangSmith（监控）+ Redis（状态持久化）
```

---

## 八、参考资料

### 官方文档
- [Claude Code Agent Teams 官方文档](https://code.claude.com/docs/en/agent-teams)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [AutoGen 文档](https://microsoft.github.io/autogen/)
- [CrewAI 文档](https://docs.crewai.com/)

### X (Twitter) 原始来源
- [四大框架对比 @goyalshaliniuk](https://x.com/goyalshaliniuk/status/2001540431335555351)
- [2025 Multi-Agent 深度分析 @zhuokaiz](https://x.com/zhuokaiz/status/2024881456422457690)
- [Claude Opus 4.6 Agent Teams @_devEmmy](https://x.com/_devEmmy/status/2037058209328828806)
- [AutoGen 最佳场景 @ingliguori](https://x.com/ingliguori/status/1973361513353134227)
- [Agentic AI 趋势 @techNmak](https://twitter.com/techNmak/status/1942494370051076556)

### 深度文章
- [Forbes: Agent 编排最佳实践](https://www.forbes.com/councils/forbestechcouncil/2025/12/16/agent-orchestration-best-practices-and-pitfalls/)
- [为什么 2025 是 Multi-Agent 元年](https://www.linkedin.com/pulse/rise-multi-agent-orchestration-why-2025-year-ai-agent-teams-llumoai-4stjf)
- [企业级 Multi-Agent 策略 2025-2026](https://www.onabout.ai/p/mastering-multi-agent-orchestration-architectures-patterns-roi-benchmarks-for-2025-2026)
- [Agent Teams 完整指南](https://amitray.com/claude-opus-4-6-agent-teams-complete-guide/)
- [如何用 Agent Teams 获得好结果](https://darasoba.medium.com/how-to-set-up-and-use-claude-code-agent-teams-and-actually-get-great-results-9a34f8648f6d)
- [Agent Teams 协作模式详解](https://heeki.medium.com/collaborating-with-agents-teams-in-claude-code-f64a465f3c11)
