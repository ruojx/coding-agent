Git 仓库地址：https://github.com/ruojx/coding-agent

本项目是一个从零实现的命令行 Coding Agent，调用 DeepSeek API。未使用 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 等 Agent 框架或 SDK。除模型 API 客户端外，对话历史、工具定义、本地文件读写、命令执行、模型输出解析、权限检查、上下文压缩、错误处理和终止判断均由本项目自行实现。

运行方式：
git clone https://github.com/ruojx/coding-agent.git
cd coding-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
在 .env 中填写 DeepSeek API Key 后运行：
python main.py
录制演示可运行：
python main.py --demo

示例任务：
/goal 我平时看程序日志时，经常要手动找 ERROR 和 WARNING，很麻烦。请帮我做一个本地可运行的小工具，能读取日志文件、统计日志等级并找出常见错误。目录结构、代码、示例数据和测试由 Agent 自己设计，最后运行验证直到确认可用。

特色功能：支持文件读写/编辑/搜索、bash 执行、权限 Hooks、TodoWrite、子 Agent、技能加载、上下文压缩、长期记忆、后台任务、Cron 调度、Agent Teams、MCP-style 动态工具、Workflow Runtime 和 Goal Loop。Goal Loop 会在模型想停止时检查文件、命令和测试结果，避免只口头声称完成。
