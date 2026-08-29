# Coding Agent

这是一个从零实现的命令行 Coding Agent。它通过 DeepSeek API 与大语言模型交互，并由本地 Harness 自己负责工具调用、权限检查、上下文管理和循环终止判断。

项目目标不是封装现成 Agent 产品，而是展示一个简化版 Claude Code / Codex 类工具的核心运行机制。

## 约束说明

本项目没有使用以下 Agent 框架或 SDK：

- LangChain
- LlamaIndex
- OpenAI Agents SDK
- Claude Agent SDK
- AutoGen
- CrewAI

项目仅使用模型 API 客户端库连接 DeepSeek 的 OpenAI 兼容接口。核心逻辑均在本地实现，包括：

- 对话历史维护
- system prompt 组装
- tool calling 参数解析
- 本地文件工具
- 本地命令执行
- 权限检查
- Hooks 生命周期
- 上下文压缩
- 长期记忆
- 任务系统
- 后台任务
- 定时调度
- 多 Agent 团队通信
- MCP-style 动态工具
- Workflow Runtime
- Goal Loop 完成条件判断

## 运行方式

```bash
git clone https://github.com/ruojx/coding-agent.git
cd coding-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

然后编辑 `.env`，填入你的 DeepSeek API Key：

```text
DEEPSEEK_API_KEY=你的 API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

启动：

```bash
python main.py
```

启动后会进入命令行交互：

```text
Coding Agent（输入 exit 退出）
>
```

## 示例任务

可以输入：

```text
/goal 创建 demo_math.py 和 demo_test.py，实现 add(a, b) 函数，并运行 python demo_test.py，直到测试通过且退出码为 0
```

这个示例会展示：

- Agent 根据目标制定计划；
- 调用文件工具创建或修改代码；
- 调用 bash 执行测试；
- 根据工具结果继续循环；
- Goal Loop 在停止前检查目标是否真的完成。

## 功能演进

项目采用逐步提交的方式开发，每个 commit 对应一个明确能力：

1. DeepSeek 模型调用和最小 Agent Loop
2. 本地文件与命令工具
3. 权限检查
4. Hooks 生命周期
5. TodoWrite 计划工具
6. Subagent 子任务
7. Skill Loading
8. Context Compaction
9. Persistent Memory
10. Task System
11. Background Tasks
12. Cron Scheduler
13. Agent Teams
14. MCP-style Dynamic Tools
15. Integrated Harness
16. Workflow Runtime
17. Goal Loop

## 主要设计

### Agent Loop

`agent_loop()` 是核心循环。它持续调用模型，解析模型返回的 tool calls，执行本地工具，将工具结果写回对话历史，直到模型不再请求工具。

### 工具系统

本地工具包括：

- `read_file`
- `write_file`
- `edit_file`
- `glob`
- `bash`
- `todo_write`
- `task`
- `load_skill`
- `remember`
- `create_task`
- `schedule_cron`
- `spawn_teammate`
- `connect_mcp`
- `run_workflow`

工具执行统一经过 `execute_tool()`，不会绕过权限和 Hooks。

### 权限与 Hooks

工具执行前会触发 `PreToolUse`，危险命令会被拒绝或要求确认。工具执行后触发 `PostToolUse`，Agent 准备结束时触发 `Stop`。

### Goal Loop

传统 Agent 往往以“模型不再调用工具”作为结束条件。本项目加入了 Goal Loop：当存在 `/goal` 时，模型想停止后还需要独立判断器检查目标是否被工具结果证明完成。若未满足，Harness 会把原因追加回上下文，让 Agent 继续执行。

## 凭据安全

API Key 通过 `.env` 提供，`.env` 不会提交到 Git。仓库只保留 `.env.example` 作为模板。

## 测试

本地测试位于 `test/` 目录，已在开发过程中用于验证 S03-S17 的功能。该目录不上传到 GitHub。
