"""编程智能体的第一个版本：一个循环和一个 Bash 工具。"""

import json
import os
import subprocess
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = (
    f"You are Coding Agent, a local coding agent powered by DeepSeek. "
    f"You are working in {os.getcwd()}. "
    "Never claim to be Claude, Anthropic, OpenAI, or ChatGPT. "
    "Use the bash tool when you need to inspect or change the local project. "
    "When the task is complete, answer the user directly."
)

BASH_TOOL = {
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
}


def run_bash(command: str) -> str:
    """执行一条 Shell 命令，并返回标准输出和错误输出。"""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
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


def agent_loop(client: OpenAI, messages: List[Dict[str, Any]]) -> None:
    """持续调用模型，直到模型不再请求使用工具。"""
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[BASH_TOOL],
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            print(assistant_message.content or "")
            return

        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name != "bash":
                result = f"Error: unknown tool {tool_call.function.name}"
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    command = arguments["command"]
                    print(f"$ {command}")
                    result = run_bash(command)
                    print(result)
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    result = f"Error: invalid bash arguments: {exc}"

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
