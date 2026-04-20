# Claude Code Skills 使用指南

> 更新时间：2026-04-16

## 概述

Skills 是 Claude Code 的扩展能力插件，通过 `/skill-name` 或在对话中自动触发。Skills 分为以下几大类：

- **Superpowers** — 工作流/流程类技能，规范开发过程
- **Document Skills** — 文档/文件创建与处理
- **Example Skills** — Document Skills 的示例版（功能相同，可选用）
- **Frontend Design** — 前端设计专用
- **独立技能** — 配置、API、平台集成等

---

## 一、Superpowers（工作流技能）

这些技能规范开发流程，大部分在特定场景下自动触发。

| 技能名 | 触发时机 | 用途 |
|--------|---------|------|
| `superpowers:brainstorming` | 任何创造性工作之前 | 探索用户意图、需求和设计，在动手写代码前先头脑风暴 |
| `superpowers:writing-plans` | 有多步任务的需求/规格时 | 编写实现计划，在写代码前先规划 |
| `superpowers:executing-plans` | 有写好的实现计划时 | 按计划逐步执行，带审查检查点 |
| `superpowers:subagent-driven-development` | 计划中有多个独立任务时 | 使用子代理并行执行独立任务 |
| `superpowers:dispatching-parallel-agents` | 面对 2+ 个独立任务时 | 并行派发多个代理处理无依赖任务 |
| `superpowers:test-driven-development` | 实现任何功能或修 bug 前 | TDD 流程：先写测试，再写实现 |
| `superpowers:systematic-debugging` | 遇到 bug、测试失败时 | 系统化调试，先诊断再修复 |
| `superpowers:verification-before-completion` | 声称工作完成之前 | 运行验证命令确认结果，用证据说话 |
| `superpowers:requesting-code-review` | 完成任务/合并前 | 请求代码审查，验证是否符合要求 |
| `superpowers:receiving-code-review` | 收到代码审查反馈时 | 严谨处理审查意见，不盲目同意 |
| `superpowers:using-git-worktrees` | 需要隔离的功能开发时 | 创建 git worktree 隔离开发环境 |
| `superpowers:finishing-a-development-branch` | 实现完成、测试通过时 | 引导完成分支工作：合并/PR/清理 |
| `superpowers:writing-skills` | 创建或编辑 skill 时 | 编写、修改、验证 skill |
| `superpowers:using-superpowers` | 每次对话开始时 | 建立 skill 使用规范（自动加载） |

### 使用示例

```
# 在对话中直接描述需求，superpowers 会自动触发
用户：帮我给这个项目加一个用户认证功能
→ 自动触发 brainstorming → writing-plans → test-driven-development → ...

# 也可以明确要求
用户：用 TDD 方式实现这个功能
→ 触发 superpowers:test-driven-development
```

---

## 二、Document Skills（文档处理技能）

处理各类文档和文件的创建、编辑、转换。

### 文档文件类

| 技能名 | 触发关键词 | 用途 |
|--------|-----------|------|
| `document-skills:pdf` | PDF 相关操作 | 读取/合并/拆分/旋转/水印/创建 PDF |
| `document-skills:docx` | Word 文档、.docx | 创建/读取/编辑 Word 文档 |
| `document-skills:pptx` | PPT、演示文稿、.pptx | 创建/读取/编辑 PowerPoint 演示文稿 |
| `document-skills:xlsx` | 电子表格、.xlsx/.csv | 打开/编辑/创建电子表格，处理公式和图表 |

### 使用示例

```
用户：帮我把这个 PDF 的前5页提取出来
→ 触发 document-skills:pdf

用户：创建一个项目汇报的 PPT
→ 触发 document-skills:pptx

用户：读取这个 Excel 文件并添加一列汇总数据
→ 触发 document-skills:xlsx
```

### 设计与创意类

| 技能名 | 触发关键词 | 用途 |
|--------|-----------|------|
| `document-skills:frontend-design` | 网页、组件、仪表盘 | 创建高质量前端界面（React 组件、Landing Page 等） |
| `document-skills:canvas-design` | 海报、艺术设计 | 用代码生成 PNG/PDF 格式的视觉设计作品 |
| `document-skills:algorithmic-art` | 生成艺术、流场 | 使用 p5.js 创建算法艺术 |
| `document-skills:theme-factory` | 主题、样式 | 为文档/幻灯片/网页应用预设主题（10种预设） |
| `document-skills:brand-guidelines` | 品牌色、Anthropic 风格 | 应用 Anthropic 官方品牌配色和字体 |
| `document-skills:slack-gif-creator` | Slack GIF | 创建适配 Slack 的动画 GIF |
| `document-skills:web-artifacts-builder` | 复杂 HTML artifact | 用 React + Tailwind + shadcn/ui 构建复杂前端组件 |

### 写作与沟通类

| 技能名 | 触发关键词 | 用途 |
|--------|-----------|------|
| `document-skills:doc-coauthoring` | 写文档、提案、技术规格 | 引导式共同撰写文档，高效传递上下文 |
| `document-skills:internal-comms` | 内部汇报、状态更新 | 撰写各种内部沟通文档（周报、汇报等） |

### 开发工具类

| 技能名 | 触发关键词 | 用途 |
|--------|-----------|------|
| `document-skills:claude-api` | Claude API、Anthropic SDK | 使用 Claude API / Anthropic SDK 构建应用 |
| `document-skills:mcp-builder` | MCP 服务器 | 构建 MCP (Model Context Protocol) 服务器 |
| `document-skills:skill-creator` | 创建/编辑 skill | 创建新 skill、修改现有 skill、运行评估 |
| `document-skills:webapp-testing` | 网页测试、Playwright | 用 Playwright 测试本地 Web 应用 |
| `document-skills:reward_graph_tool` | reward 图、训练曲线 | 从训练日志提取 GRPO 指标并生成折线图 |

---

## 三、独立技能

| 技能名 | 触发关键词 | 用途 |
|--------|-----------|------|
| `update-config` | 配置 settings.json、hooks | 配置 Claude Code 的 settings.json，设置自动化 hooks |
| `simplify` | 简化代码、代码审查 | 审查已修改代码的复用性、质量和效率 |
| `loop` | 循环执行、定时检查 | 按时间间隔循环执行命令（如 `/loop 5m /foo`） |
| `claude-api` | Claude API、Anthropic SDK | 使用 Claude API 构建应用 |
| `antcode-skill` | AntCode、PR、代码评审 | AntCode 代码托管全流程：仓库/PR/评审/Issue/Pipeline |
| `skill-yuque-cli-guide` | 语雀 CLI、yuque 命令 | 语雀 CLI 的安装、认证、查询、写入操作指南 |

---

## 四、Frontend Design

| 技能名 | 触发关键词 | 用途 |
|--------|-----------|------|
| `frontend-design:frontend-design` | 前端界面、Web 组件 | 创建生产级高质量前端界面，避免泛 AI 风格 |

---

## 五、Example Skills（示例技能）

与 Document Skills 功能相同的示例版本，可互换使用。包含：

`algorithmic-art` · `brand-guidelines` · `canvas-design` · `claude-api` · `doc-coauthoring` · `docx` · `frontend-design` · `internal-comms` · `mcp-builder` · `pdf` · `pptx` · `skill-creator` · `slack-gif-creator` · `theme-factory` · `webapp-testing` · `web-artifacts-builder` · `xlsx`

---

## 六、使用方式

### 1. 自动触发

大多数 skill 会根据对话内容自动触发，无需手动调用：

```
用户：帮我创建一个 Word 文档
→ Claude 自动识别并触发 document-skills:docx
```

### 2. 斜杠命令调用

在 Claude Code 中使用 `/skill-name` 显式调用：

```
/simplify          # 审查代码质量
/loop 5m /simplify # 每5分钟执行一次代码审查
```

### 3. 对话中提及

直接在对话中描述需求，Claude 会匹配最相关的 skill：

```
用户：用 Playwright 帮我测试一下本地的 Web 应用
→ 触发 document-skills:webapp-testing
```

---

## 七、技能优先级

当多个 skill 可能适用时：

1. **流程技能优先**（brainstorming, debugging）— 决定如何做
2. **实现技能其次**（frontend-design, mcp-builder）— 指导怎么做

例如：
- "帮我做一个 X 功能" → 先 brainstorming，再 implementation skills
- "修复这个 bug" → 先 systematic-debugging，再领域特定 skills

---

## 八、用户指令优先级

```
用户明确指令（CLAUDE.md）> Superpowers Skills > 系统默认行为
```

如果 CLAUDE.md 中有 "不使用 TDD"，即使 skill 要求 TDD，也以用户指令为准。
