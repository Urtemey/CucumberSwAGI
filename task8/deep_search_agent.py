"""
Task 8. Deep search agent from scratch.

The agent follows the main ideas from "deep agents from scratch":
- an explicit agent loop around a chat model with tool calls;
- a todo list for multi-step work;
- web search as an external information tool;
- a virtual file system stored in agent state;
- final export of virtual files into the real file system.

Run:
    python task8/deep_search_agent.py

Environment defaults target LM Studio:
    OPENAI_BASE_URL=http://localhost:1234/v1
    OPENAI_MODEL=qwen/qwen3-vl-4b
    OPENAI_API_KEY=fake
"""

from __future__ import annotations

import html
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = "agent_output"


def configure_console_encoding() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


configure_console_encoding()


@dataclass
class TodoItem:
    task: str
    status: Literal["todo", "doing", "done"] = "todo"


@dataclass
class DeepAgentState:
    virtual_files: dict[str, str] = field(default_factory=dict)
    todos: list[TodoItem] = field(default_factory=list)
    exported_to: str | None = None


ACTIVE_STATE: DeepAgentState | None = None


def current_state() -> DeepAgentState:
    if ACTIVE_STATE is None:
        raise RuntimeError("Agent state is not initialized")
    return ACTIVE_STATE


def normalize_virtual_path(path: str) -> str:
    """Keep virtual file names portable and inside the virtual root."""
    cleaned = path.replace("\\", "/").strip()
    pure_path = PurePosixPath(cleaned)

    if pure_path.is_absolute() or ".." in pure_path.parts or not cleaned:
        raise ValueError("Use a relative virtual path without '..'")

    return str(pure_path)


def safe_export_root(output_dir: str) -> Path:
    root = (BASE_DIR / output_dir).resolve()
    base = BASE_DIR.resolve()
    if base != root and base not in root.parents:
        raise ValueError("Export directory must stay inside task8")
    return root


def export_virtual_files(state: DeepAgentState, output_dir: str = DEFAULT_EXPORT_DIR) -> str:
    export_root = safe_export_root(output_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    for virtual_path, content in state.virtual_files.items():
        relative_path = normalize_virtual_path(virtual_path)
        real_path = (export_root / relative_path).resolve()
        if export_root != real_path.parent and export_root not in real_path.parents:
            raise ValueError(f"Unsafe export path: {virtual_path}")

        real_path.parent.mkdir(parents=True, exist_ok=True)
        real_path.write_text(content, encoding="utf-8")

    state.exported_to = str(export_root)
    return str(export_root)


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the internet and return compact search results with titles and URLs."""
    max_results = max(1, min(max_results, 8))
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded_query}"

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            )
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw_html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return (
            "Search failed. Check internet access or replace search_web with an API "
            f"search provider. Error: {exc}"
        )

    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    results: list[str] = []
    for match in pattern.finditer(raw_html):
        title = re.sub(r"<.*?>", "", match.group("title"))
        title = html.unescape(title).strip()
        href = html.unescape(match.group("href")).strip()

        parsed = urllib.parse.urlparse(href)
        query_params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in query_params:
            href = query_params["uddg"][0]

        if title and href:
            results.append(f"{len(results) + 1}. {title}\n   URL: {href}")
        if len(results) >= max_results:
            break

    if not results:
        return "No search results parsed from DuckDuckGo HTML."

    return "\n".join(results)


@tool
def update_todo(task: str, status: Literal["todo", "doing", "done"] = "todo") -> str:
    """Add or update one todo item in the agent plan."""
    state = current_state()

    for item in state.todos:
        if item.task == task:
            item.status = status
            return f"Updated todo: [{status}] {task}"

    state.todos.append(TodoItem(task=task, status=status))
    return f"Added todo: [{status}] {task}"


@tool
def list_todos() -> str:
    """List current todo items."""
    state = current_state()
    if not state.todos:
        return "Todo list is empty."

    return "\n".join(f"- [{item.status}] {item.task}" for item in state.todos)


@tool
def write_file(path: str, content: str) -> str:
    """Write or overwrite a virtual file."""
    state = current_state()
    virtual_path = normalize_virtual_path(path)
    state.virtual_files[virtual_path] = content
    return f"Virtual file written: {virtual_path} ({len(content)} characters)"


@tool
def read_file(path: str) -> str:
    """Read a virtual file."""
    state = current_state()
    virtual_path = normalize_virtual_path(path)

    if virtual_path not in state.virtual_files:
        return f"Virtual file not found: {virtual_path}"

    return state.virtual_files[virtual_path]


@tool
def edit_file(path: str, old: str, new: str) -> str:
    """Replace text inside a virtual file."""
    state = current_state()
    virtual_path = normalize_virtual_path(path)

    if virtual_path not in state.virtual_files:
        return f"Virtual file not found: {virtual_path}"

    content = state.virtual_files[virtual_path]
    if old not in content:
        return f"Text to replace was not found in {virtual_path}"

    state.virtual_files[virtual_path] = content.replace(old, new, 1)
    return f"Edited virtual file: {virtual_path}"


@tool
def list_files() -> str:
    """List virtual files."""
    state = current_state()
    if not state.virtual_files:
        return "Virtual file system is empty."

    return "\n".join(
        f"- {path} ({len(content)} characters)"
        for path, content in sorted(state.virtual_files.items())
    )


@tool
def export_files(output_dir: str = DEFAULT_EXPORT_DIR) -> str:
    """Export all virtual files into a real directory inside task8."""
    state = current_state()
    if not state.virtual_files:
        return "No virtual files to export."

    export_root = export_virtual_files(state, output_dir)
    return f"Exported {len(state.virtual_files)} virtual file(s) to: {export_root}"


TOOLS = [
    search_web,
    update_todo,
    list_todos,
    write_file,
    read_file,
    edit_file,
    list_files,
    export_files,
]
TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}


SYSTEM_PROMPT = f"""
You are a deep research agent implemented from scratch.

Your workflow:
1. Maintain a todo list with update_todo and list_todos.
2. Search the internet with search_web when the task needs external facts.
3. Store durable intermediate and final work in virtual files using write_file.
4. Inspect and revise virtual files with read_file, edit_file, and list_files.
5. Before the final answer, call export_files with output_dir="{DEFAULT_EXPORT_DIR}".
6. Final answer must mention exported file paths and briefly summarize the result.

Rules:
- Do not pretend to have searched if search_web failed.
- Keep virtual file paths relative, for example notes.md or reports/summary.md.
- Create at least one final virtual file for the user.
- If web search is unavailable, still write a file explaining the limitation and
  what information would be needed.
""".strip()


def build_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "qwen/qwen3-vl-4b")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("OPENAI_API_KEY", "fake")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        temperature=0.2,
    )


def tool_result_for_call(tool_call: dict[str, Any]) -> str:
    tool_name = tool_call["name"]
    selected_tool = TOOLS_BY_NAME.get(tool_name)

    if selected_tool is None:
        return f"Unknown tool: {tool_name}"

    try:
        return str(selected_tool.invoke(tool_call["args"]))
    except Exception as exc:
        return f"Tool {tool_name} failed: {exc}"


def run_agent(user_task: str, max_steps: int = 30) -> tuple[str, DeepAgentState]:
    """Run the hand-written tool-calling loop."""
    global ACTIVE_STATE

    state = DeepAgentState()
    ACTIVE_STATE = state

    llm = build_llm().bind_tools(TOOLS)
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_task),
    ]

    final_answer = ""
    try:
        for step_number in range(1, max_steps + 1):
            print(f"\n--- agent step {step_number} ---")
            ai_message = llm.invoke(messages)
            messages.append(ai_message)

            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                final_answer = str(ai_message.content)
                break

            for tool_call in tool_calls:
                print(f"tool: {tool_call['name']}({tool_call['args']})")
                result = tool_result_for_call(tool_call)
                print(result)
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
        else:
            final_answer = "Stopped because max_steps was reached."

        if state.virtual_files and state.exported_to is None:
            export_root = export_virtual_files(state, DEFAULT_EXPORT_DIR)
            final_answer += (
                f"\n\nSafety export: virtual files were exported to {export_root}."
            )

        return final_answer, state
    finally:
        ACTIVE_STATE = None


def read_user_task() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    print("Введите исследовательскую задачу. Для примера нажмите Enter:")
    user_task = sys.stdin.readline().strip()
    if user_task:
        return user_task

    return (
        "Найди в интернете краткую информацию о LangGraph и создай виртуальный "
        "файл notes.md с основными выводами и ссылками."
    )


def main() -> None:
    user_task = read_user_task()
    final_answer, state = run_agent(user_task)

    print("\n=== Final answer ===")
    print(final_answer)

    print("\n=== Virtual files ===")
    if state.virtual_files:
        for path in sorted(state.virtual_files):
            print(f"- {path}")
    else:
        print("No virtual files were created.")

    if state.exported_to:
        print(f"\nExported to: {state.exported_to}")


if __name__ == "__main__":
    main()
