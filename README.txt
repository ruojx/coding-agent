Git 仓库地址：https://github.com/ruojx/coding-agent

本项目是一个从零实现的命令行 Coding Agent，运行时调用 DeepSeek API。项目没有使用 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 等 Agent 框架或 SDK；除模型 API 客户端外，Agent Loop、工具定义、本地执行、权限检查、模型 tool calling 解析、上下文压缩、任务系统、错误恢复和终止判断均由本项目自行实现。

运行方式：
1. 克隆仓库并进入目录：
   git clone https://github.com/ruojx/coding-agent.git
   cd coding-agent
2. 创建并启用虚拟环境：
   python3 -m venv .venv
   source .venv/bin/activate
3. 安装依赖：
   pip install -r requirements.txt
4. 复制环境变量模板并填写 DeepSeek API Key：
   cp .env.example .env
5. 启动：
   python main.py

启动后在命令行输入编程任务，例如：
/goal 创建 demo_math.py 和 demo_test.py，实现 add(a, b)，并运行 python demo_test.py，直到测试通过且退出码为 0

特色功能：
支持读写文件、编辑文件、搜索文件和执行命令；通过 Hooks 和权限管线拦截危险操作；支持 TodoWrite、子 Agent、技能加载、上下文压缩、长期记忆、持久化任务系统、后台命令、Cron 定时 prompt、Agent Teams、MCP-style 动态工具、Workflow Runtime 和 Goal Loop。Goal Loop 会在模型想停止时进行独立验收，避免模型只口头声称完成。
