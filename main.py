"""编程智能体的第二个版本：一个循环和五个本地工具。"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
WORKDIR = Path.cwd().resolve()

SYSTEM_PROMPT = (
    f"You are Coding Agent, a local coding agent powered by DeepSeek. "
    f"You are working in {WORKDIR}. "
    "Never claim to be Claude, Anthropic, OpenAI, or ChatGPT. "
    "Use the provided tools to inspect or change the local project. "
    "Prefer dedicated file tools over bash for file operations. "
    "When the task is complete, answer the user directly."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the current project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project directory.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional maximum number of lines to return.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text to a file inside the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace the first exact text match in a project file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project directory.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find project files that match a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern such as **/*.py.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
]


def safe_path(path: str) -> Path:
    """解析项目内路径，并阻止文件工具访问项目目录之外的位置。"""
    target = (WORKDIR / path).resolve()
    try:
        target.relative_to(WORKDIR)
    except ValueError as exc:
        raise ValueError(f"路径超出项目目录：{path}") from exc
    return target


def run_read(path: str, limit: Optional[int] = None) -> str:
    """读取项目内的 UTF-8 文本文件。"""
    text = safe_path(path).read_text(encoding="utf-8")
    if limit is None:
        return text
    return "\n".join(text.splitlines()[:limit])


def run_write(path: str, content: str) -> str:
    """在项目内写入 UTF-8 文本，并按需创建父目录。"""
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {len(content)} 个字符到 {path}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在项目文件中精确替换第一次出现的指定文本。"""
    if not old_text:
        return "错误：old_text 不能为空"

    target = safe_path(path)
    text = target.read_text(encoding="utf-8")
    if old_text not in text:
        return "错误：未找到要替换的文本"

    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"已编辑 {path}"


def run_glob(pattern: str) -> str:
    """查找项目内匹配通配模式的文件或目录。"""
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        return f"错误：通配模式超出项目目录：{pattern}"

    matches = []
    for match in WORKDIR.glob(pattern):
        resolved = match.resolve()
        try:
            relative = resolved.relative_to(WORKDIR)
        except ValueError:
            continue
        matches.append(str(relative))

    shown = sorted(set(matches))[:200]
    if not shown:
        return "（没有匹配项）"
    if len(set(matches)) > 200:
        shown.append("……其余匹配项已省略，请缩小搜索范围")
    return "\n".join(shown)


def run_bash(command: str) -> str:
    """执行一条 Shell 命令，并返回标准输出和错误输出。"""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds"
    except OSError as exc:
        return f"Error: {exc}"

    output = (completed.stdout + completed.stderr).strip()
    return output if output else "(no output)"


TOOL_HANDLERS: Dict[str, Callable[..., str]] = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """根据工具名称查找处理函数，并统一返回执行结果。"""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"错误：未知工具 {name}"

    try:
        return handler(**arguments)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        return f"错误：{exc}"


def agent_loop(client: OpenAI, messages: List[Dict[str, Any]]) -> None:
    """持续调用模型，直到模型不再请求使用工具。"""
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            print(assistant_message.content or "")
            return

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("工具参数必须是 JSON 对象")

                if tool_name == "bash":
                    command = arguments["command"]
                    print(f"$ {command}")
                else:
                    print(f"[{tool_name}]")

                result = execute_tool(tool_name, arguments)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                result = f"错误：工具参数无效：{exc}"

            print(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("Coding Agent（输入 exit 退出）")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        messages.append({"role": "user", "content": query})
        agent_loop(client, messages)


if __name__ == "__main__":
    main()
