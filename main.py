"""编程智能体的第七个版本：增加按需加载 Skill 的能力。"""

import ast
import json
import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
WORKDIR = Path.cwd().resolve()
SKILLS_DIR = WORKDIR / "skills"

BASE_SYSTEM_PROMPT = (
    f"You are Coding Agent, a local coding agent powered by DeepSeek. "
    f"You are working in {WORKDIR}. "
    "Never claim to be Claude, Anthropic, OpenAI, or ChatGPT. "
    "Use the provided tools to inspect or change the local project. "
    "Prefer dedicated file tools over bash for file operations. "
    "For multi-step tasks, call todo_write first to make a plan, "
    "then update todo status as you work. "
    "Use the task tool for focused subtasks that can be handled in an "
    "isolated context, such as file inspection or localized analysis. "
    "Use load_skill to read full skill instructions when a listed skill "
    "applies to the current task. "
    "Tool calls may be denied by the local permission system. "
    "If a call is denied, do not claim it succeeded; choose a safer approach. "
    "When the task is complete, answer the user directly."
)

BASE_SUBAGENT_SYSTEM_PROMPT = (
    f"You are a focused subagent inside Coding Agent. "
    f"You are working in {WORKDIR}. "
    "Complete only the delegated subtask. "
    "Use tools when needed, but keep the final answer concise. "
    "Use load_skill when the delegated subtask matches a listed skill. "
    "Do not claim to be Claude, Anthropic, OpenAI, or ChatGPT. "
    "Your intermediate tool calls stay in your own context; "
    "only your final summary returns to the parent agent."
)


class SkillLoader:
    """扫描 skills 目录，只把技能目录放进系统提示。"""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills: Dict[str, Dict[str, str]] = {}

    def scan(self) -> None:
        """读取 skills/*/SKILL.md 的元信息，建立名称到正文的索引。"""
        self.skills.clear()
        if not self.skills_dir.exists():
            return

        root = self.skills_dir.resolve()
        for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
            resolved = manifest.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            if not resolved.is_file():
                continue

            content = resolved.read_text(encoding="utf-8")
            metadata, body = self.parse_frontmatter(content)
            name = self.clean_metadata(metadata.get("name"))
            if not name:
                name = manifest.parent.name

            description = self.clean_metadata(metadata.get("description"))
            if not description:
                description = self.first_body_line(body)
            try:
                display_path = str(resolved.relative_to(WORKDIR))
            except ValueError:
                display_path = str(resolved)

            self.skills[name] = {
                "name": name,
                "description": description,
                "content": content,
                "path": display_path,
            }

    def catalog(self) -> str:
        """返回只包含名称和描述的技能目录。"""
        if not self.skills:
            return "- 暂无可用技能"

        lines = []
        for name in sorted(self.skills):
            description = self.skills[name]["description"]
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def load(self, name: str) -> str:
        """按技能名称返回完整 SKILL.md，不把 name 当作文件路径。"""
        skill = self.skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self.skills)) or "none"
            return f"错误：未知技能 {name}。可用技能：{available}"

        return (
            f"已加载技能：{skill['name']}\n"
            f"来源：{skill['path']}\n\n"
            f"{skill['content']}"
        )

    def parse_frontmatter(
        self, content: str
    ) -> Tuple[Dict[str, str], str]:
        """解析最小 YAML frontmatter，只支持 key: value 形式。"""
        if not content.startswith("---\n"):
            return {}, content

        end_marker = content.find("\n---\n", 4)
        if end_marker == -1:
            return {}, content

        metadata_text = content[4:end_marker]
        body = content[end_marker + len("\n---\n"):]
        metadata: Dict[str, str] = {}
        for line in metadata_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                metadata[key] = value
        return metadata, body

    def clean_metadata(self, value: Optional[str]) -> str:
        """清理 frontmatter 字段，避免空值进入技能目录。"""
        if not isinstance(value, str):
            return ""
        return " ".join(value.strip().split())

    def first_body_line(self, body: str) -> str:
        """没有 description 时，用正文第一行作为兜底描述。"""
        for line in body.splitlines():
            text = line.strip().lstrip("#").strip()
            if text:
                return " ".join(text.split())
        return "无描述"


SKILL_LOADER = SkillLoader(SKILLS_DIR)


def build_system_prompt() -> str:
    """运行时拼接主 Agent 提示词和技能目录。"""
    SKILL_LOADER.scan()
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Available skills:\n{SKILL_LOADER.catalog()}\n\n"
        "Call load_skill with a skill name before following its full rules."
    )


def build_subagent_system_prompt() -> str:
    """运行时拼接子 Agent 提示词和同一份技能目录。"""
    SKILL_LOADER.scan()
    return (
        f"{BASE_SUBAGENT_SYSTEM_PROMPT}\n\n"
        f"Available skills:\n{SKILL_LOADER.catalog()}\n\n"
        "Call load_skill with a skill name before following its full rules."
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
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Create or update the current task plan before and during "
                "multi-step coding work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The full current task list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "A concrete task step.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                    ],
                                    "description": "Current status.",
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Run a focused subtask in a fresh subagent context. "
                "Use this for isolated inspection or analysis work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The delegated subtask prompt.",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load the full SKILL.md instructions for one available "
                "skill by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name from the available catalog.",
                    }
                },
                "required": ["name"],
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


class TodoManager:
    """保存当前会话中的任务计划，并校验模型传入的 todo 列表。"""

    VALID_STATUSES = {"pending", "in_progress", "completed"}

    def __init__(self) -> None:
        self.items: List[Dict[str, str]] = []

    def update(self, todos: Any) -> str:
        """用模型提交的新列表替换当前计划，并返回可读进度。"""
        if isinstance(todos, str):
            todos = self._parse_todos_text(todos)
        if not isinstance(todos, list):
            raise ValueError("todos 必须是列表")
        if len(todos) > 20:
            raise ValueError("todos 最多只能包含 20 项")

        validated: List[Dict[str, str]] = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] 必须是对象")

            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).strip()
            if not content:
                raise ValueError(f"todos[{index}] 缺少 content")
            if status not in self.VALID_STATUSES:
                raise ValueError(
                    f"todos[{index}] 的 status 不合法：{status}"
                )
            if status == "in_progress":
                in_progress_count += 1

            validated.append({"content": content, "status": status})

        if in_progress_count > 1:
            raise ValueError("同一时间最多只能有一个 in_progress 任务")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """把当前 todo 状态渲染成终端和模型都容易读懂的文本。"""
        if not self.items:
            return "当前没有任务计划"

        marker_by_status = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }
        lines = ["## 当前任务计划"]
        for todo in self.items:
            marker = marker_by_status[todo["status"]]
            lines.append(f"{marker} {todo['content']}")

        completed_count = sum(
            todo["status"] == "completed" for todo in self.items
        )
        lines.append(f"\n进度：{completed_count}/{len(self.items)} 已完成")
        return "\n".join(lines)

    def _parse_todos_text(self, todos: str) -> Any:
        """兼容 JSON 字符串和 Python 字面量字符串，但不使用 eval。"""
        try:
            return json.loads(todos)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(todos)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    "todos 字符串必须是 JSON 数组或安全字面量列表"
                ) from exc


TODO = TodoManager()
ACTIVE_CLIENT: Optional[OpenAI] = None


def run_todo_write(todos: Any) -> str:
    """更新当前任务计划；这个工具只负责规划，不直接修改文件。"""
    output = TODO.update(todos)
    print(output)
    return output


def run_task(prompt: str) -> str:
    """启动一个同步子 Agent，并只把最终总结返回给主 Agent。"""
    if ACTIVE_CLIENT is None:
        return "错误：模型客户端尚未初始化，无法启动子 Agent"
    if not isinstance(prompt, str) or not prompt.strip():
        return "错误：task prompt 不能为空"

    return run_subagent(ACTIVE_CLIENT, prompt.strip())


def run_load_skill(name: str) -> str:
    """按名称加载完整技能说明，返回给模型继续使用。"""
    if not isinstance(name, str) or not name.strip():
        return "错误：技能名称不能为空"

    SKILL_LOADER.scan()
    return SKILL_LOADER.load(name.strip())


TOOL_HANDLERS: Dict[str, Callable[..., str]] = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
    "task": run_task,
    "load_skill": run_load_skill,
}

SUBAGENT_TOOLS = [
    tool
    for tool in TOOLS
    if tool["function"]["name"] != "task"
]

SUBAGENT_TOOL_HANDLERS = {
    name: handler
    for name, handler in TOOL_HANDLERS.items()
    if name != "task"
}


class PermissionDecision(str, Enum):
    """权限管线可能产生的三种决定。"""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


# 第一关：这些 Bash 片段风险过高，即使用户确认也不执行。
HARD_DENY_PATTERNS = (
    "rm -rf /",
    "sudo ",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
)

# 第二关：这些命令可能有破坏性，但具体文件可能确实需要用户删除。
DESTRUCTIVE_BASH_PATTERN = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|rmdir)(?=\s|$|[;&|()])"
)


def path_is_outside_workspace(path: str) -> bool:
    """判断工具参数中的路径是否越过当前工作目录。"""
    try:
        (WORKDIR / path).resolve().relative_to(WORKDIR)
    except (OSError, ValueError):
        return True
    return False


def check_hard_deny(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """第一关：匹配无条件禁止的高危 Bash 命令。"""
    if tool_name != "bash":
        return None

    command = str(arguments.get("command", "")).lower()
    for pattern in HARD_DENY_PATTERNS:
        if pattern in command:
            return f"命令包含永久禁止的高危片段：{pattern}"
    return None


def check_permission_rules(
    tool_name: str, arguments: Dict[str, Any]
) -> Optional[Tuple[PermissionDecision, str]]:
    """第二关：根据工具名称和结构化参数匹配权限规则。"""
    if tool_name in {"read_file", "write_file", "edit_file"}:
        path = arguments.get("path")
        if not isinstance(path, str) or path_is_outside_workspace(path):
            return PermissionDecision.DENY, "文件路径超出当前工作目录"

    if tool_name == "glob":
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str):
            return PermissionDecision.DENY, "glob 模式必须是字符串"
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            return PermissionDecision.DENY, "glob 模式超出当前工作目录"

    if tool_name == "bash":
        command = arguments.get("command")
        if not isinstance(command, str):
            return PermissionDecision.DENY, "Bash 命令必须是字符串"
        if DESTRUCTIVE_BASH_PATTERN.search(command):
            return PermissionDecision.ASK, "命令可能删除文件或目录"
        if "> /etc/" in command or "chmod 777" in command:
            return PermissionDecision.ASK, "命令可能修改敏感位置或权限"

    return None


def evaluate_permission(
    tool_name: str, arguments: Dict[str, Any]
) -> Tuple[PermissionDecision, str]:
    """依次执行硬拒绝和规则匹配，返回权限决定及原因。"""
    hard_deny_reason = check_hard_deny(tool_name, arguments)
    if hard_deny_reason:
        return PermissionDecision.DENY, hard_deny_reason

    matched_rule = check_permission_rules(tool_name, arguments)
    if matched_rule:
        return matched_rule

    return PermissionDecision.ALLOW, "常规工作区操作"


def request_user_approval(
    tool_name: str, arguments: Dict[str, Any], reason: str
) -> bool:
    """第三关：展示风险操作，并让用户决定是否继续。"""
    arguments_text = json.dumps(arguments, ensure_ascii=False)
    if len(arguments_text) > 500:
        arguments_text = arguments_text[:500] + "……"

    print(f"[需要确认] {reason}")
    print(f"工具：{tool_name}")
    print(f"参数：{arguments_text}")
    choice = input("允许执行吗？[y/N] ").strip().lower()
    return choice in {"y", "yes"}


def check_permission(
    tool_name: str, arguments: Dict[str, Any]
) -> Tuple[bool, str]:
    """串联三道权限门，并返回是否允许以及可反馈给模型的结果。"""
    decision, reason = evaluate_permission(tool_name, arguments)

    if decision == PermissionDecision.DENY:
        message = f"权限拒绝：{reason}"
        print(f"[已阻止] {reason}")
        return False, message

    if decision == PermissionDecision.ASK:
        if not request_user_approval(tool_name, arguments, reason):
            message = f"用户拒绝执行：{reason}"
            print("[未执行] 用户没有批准本次工具调用")
            return False, message
        print("[已允许] 用户批准本次工具调用")

    return True, ""


HookCallback = Callable[..., Optional[str]]

HOOKS: Dict[str, List[HookCallback]] = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback: HookCallback) -> None:
    """把回调函数注册到指定的 Hook 事件。"""
    if event not in HOOKS:
        raise ValueError(f"未知 Hook 事件：{event}")
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args: Any) -> Optional[str]:
    """依次触发事件回调；非空返回值表示当前流程需要被拦截。"""
    if event not in HOOKS:
        raise ValueError(f"未知 Hook 事件：{event}")

    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


def prompt_context_hook(query: str) -> None:
    """UserPromptSubmit：显示本次请求使用的工作目录。"""
    print(f"[Hook:UserPromptSubmit] 工作目录：{WORKDIR}")
    return None


def permission_hook(
    tool_name: str, arguments: Dict[str, Any]
) -> Optional[str]:
    """PreToolUse：复用 S03 权限管线，拒绝时返回原因。"""
    allowed, result = check_permission(tool_name, arguments)
    return None if allowed else result


def tool_log_hook(tool_name: str, arguments: Dict[str, Any]) -> None:
    """PreToolUse：记录即将执行的工具及参数摘要。"""
    arguments_text = json.dumps(arguments, ensure_ascii=False)
    if len(arguments_text) > 120:
        arguments_text = arguments_text[:120] + "……"
    print(f"[Hook:PreToolUse] {tool_name} {arguments_text}")
    return None


def large_output_hook(
    tool_name: str, arguments: Dict[str, Any], output: str
) -> None:
    """PostToolUse：工具输出过大时给出提醒，但暂不截断。"""
    del arguments
    if len(output) > 100_000:
        print(
            f"[Hook:PostToolUse] 警告：{tool_name} 输出了 "
            f"{len(output)} 个字符"
        )
    return None


def stop_summary_hook(messages: List[Dict[str, Any]]) -> None:
    """Stop：在 Agent 即将结束时统计当前会话的工具结果数。"""
    tool_count = sum(
        1 for message in messages if message.get("role") == "tool"
    )
    print(f"[Hook:Stop] 当前会话共返回 {tool_count} 个工具结果")
    return None


register_hook("UserPromptSubmit", prompt_context_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", tool_log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", stop_summary_hook)


def execute_tool(
    name: str,
    arguments: Dict[str, Any],
    handlers: Dict[str, Callable[..., str]],
) -> str:
    """根据工具名称查找处理函数，并统一返回执行结果。"""
    handler = handlers.get(name)
    if handler is None:
        return f"错误：未知工具 {name}"

    try:
        return handler(**arguments)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        return f"错误：{exc}"


def agent_loop(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    handlers: Optional[Dict[str, Callable[..., str]]] = None,
    max_steps: int = 30,
    agent_name: str = "Agent",
    print_final: bool = True,
) -> str:
    """持续调用模型，直到模型不再请求使用工具。"""
    tools = TOOLS if tools is None else tools
    handlers = TOOL_HANDLERS if handlers is None else handlers
    rounds_since_todo = 0

    for _step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            continuation = trigger_hooks("Stop", messages)
            if continuation is not None:
                messages.append({"role": "user", "content": continuation})
                continue
            final_text = assistant_message.content or ""
            if print_final:
                print(final_text)
            return final_text

        used_todo = False
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("工具参数必须是 JSON 对象")

                if tool_name == "bash":
                    command = arguments["command"]
                    print(f"[{agent_name} 请求 bash] $ {command}")
                else:
                    print(f"[{agent_name} 请求 {tool_name}]")
                if tool_name == "todo_write":
                    used_todo = True

                blocked = trigger_hooks(
                    "PreToolUse", tool_name, arguments
                )
                if blocked is None:
                    result = execute_tool(tool_name, arguments, handlers)
                    trigger_hooks(
                        "PostToolUse", tool_name, arguments, result
                    )
                else:
                    result = blocked
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

        if used_todo:
            rounds_since_todo = 0
        else:
            rounds_since_todo += 1

        if rounds_since_todo >= 3:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "<reminder>请使用 todo_write 更新当前任务计划，"
                        "再继续执行后续步骤。</reminder>"
                    ),
                }
            )
            rounds_since_todo = 0

    final_text = f"错误：{agent_name} 达到最大循环步数 {max_steps}"
    if print_final:
        print(final_text)
    return final_text


def run_subagent(client: OpenAI, prompt: str) -> str:
    """用全新的 messages 列表执行子任务，避免污染主上下文。"""
    sub_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_subagent_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    final_text = agent_loop(
        client=client,
        messages=sub_messages,
        tools=SUBAGENT_TOOLS,
        handlers=SUBAGENT_TOOL_HANDLERS,
        max_steps=10,
        agent_name="Subagent",
        print_final=False,
    )
    return final_text or "子 Agent 没有返回总结"


def main() -> None:
    global ACTIVE_CLIENT

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    ACTIVE_CLIENT = client
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()}
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

        trigger_hooks("UserPromptSubmit", query)
        messages.append({"role": "user", "content": query})
        agent_loop(client, messages)


if __name__ == "__main__":
    main()
