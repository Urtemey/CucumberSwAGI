"""
Практическое задание N3: память и подтверждение действий.

Скрипт является доработкой shopping-agent из предыдущих заданий:
- вывод идет через rich.Console;
- история разговора хранится через MemorySaver и общий thread_id;
- агент останавливается перед узлом tools через interrupt_before=["tools"];
- каждый запрошенный tool call показывается пользователю и требует подтверждения;
- при подтверждении выполнение возобновляется через ask_and_run(None, config);
- есть интерактивный чат-цикл.
"""

from __future__ import annotations

import sys
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr
from rich.console import Console
from rich.panel import Panel


def configure_console_encoding() -> None:
    """Windows-safe UTF-8 output without requiring PYTHONIOENCODING."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


configure_console_encoding()
console = Console()


# 1. Подключение к локальной LLM через LM Studio.
llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)


# 2. Инструмент с отдельным субагентом.
@tool
def get_price(product: str, city: str) -> str:
    """Возвращает примерную цену продукта в указанном городе."""
    subagent_prompt = (
        "Ты эксперт по розничным ценам в российских городах. "
        "Оцени реалистичную стоимость продукта в указанном городе. "
        "Ответь строго одной строкой markdown-таблицы формата:\n"
        "| Продукт | Цена (руб.) | Магазин |\n"
        "Без пояснений и без итогов."
    )

    price_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=subagent_prompt,
    )

    result = price_agent.invoke(
        {
            "messages": [
                {
                    "role": "human",
                    "content": f"Сколько стоит {product} в городе {city}?",
                }
            ]
        }
    )
    return result["messages"][-1].content


# 3. Главный агент с памятью и паузой перед каждым инструментом.
memory = MemorySaver()

main_agent = create_agent(
    model=llm,
    tools=[get_price],
    system_prompt=(
        "Ты помощник по планированию покупок. "
        "Когда пользователь дает список продуктов и город, вызывай get_price. "
        "Важно: делай только один вызов get_price за один шаг. "
        "После результата инструмента продолжай работу и, если нужны еще цены, "
        "снова вызови get_price только для одного следующего продукта. "
        "В конце собери результаты в markdown-таблицу и посчитай итоговую стоимость."
    ),
    checkpointer=memory,
    interrupt_before=["tools"],
)


step = 1


def format_message(message: Any) -> str:
    """Formats model messages and tool-call messages from stream updates."""
    if getattr(message, "content", None):
        return str(message.content)

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        return "\n".join(format_tool_call(call) for call in tool_calls)

    return repr(message)


def format_tool_call(tool_call: dict[str, Any]) -> str:
    return f"{tool_call['name']}({tool_call['args']})"


def format_chunk_message(chunk: tuple[Any, dict[str, Any]]) -> None:
    """Prints token chunks and a separator when the graph step changes."""
    global step

    message, meta = chunk
    current_step = meta.get("langgraph_step")
    if current_step != step:
        step = current_step
        console.print("\n --- --- --- \n")

    if getattr(message, "content", None):
        console.print(message.content, end="")


def get_pending_tool_calls(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Reads tool calls from the paused agent state."""
    state = main_agent.get_state(config)
    if state.next != ("tools",):
        return []

    last_message = state.values["messages"][-1]
    return list(getattr(last_message, "tool_calls", None) or [])


def confirm_pending_tool_calls(config: dict[str, Any]) -> bool:
    """Asks the user to approve every pending tool call before resuming."""
    tool_calls = get_pending_tool_calls(config)

    if not tool_calls:
        console.print("[red]Пауза обнаружена, но вызовы инструментов не найдены.[/red]")
        return False

    console.print()
    console.rule("[bold yellow]Подтверждение действий агента[/bold yellow]")

    for index, tool_call in enumerate(tool_calls, start=1):
        console.print(
            Panel.fit(
                format_tool_call(tool_call),
                title=f"Вызов инструмента {index}/{len(tool_calls)}",
                border_style="yellow",
            )
        )

        answer = input("Разрешить этот вызов? (Y/n): ").strip().lower()
        if answer not in ("", "y", "yes", "д", "да"):
            console.print("[red]Отменено пользователем. Инструменты не выполнялись.[/red]")
            return False

    return True


def ask_and_run(user_input: dict[str, Any] | None, config: dict[str, Any]) -> None:
    """Runs the agent stream or resumes execution after an interrupt.

    user_input is {"messages": [...]} for a new user message or None when the
    graph should continue from the saved interrupt point.
    """
    for chunk in main_agent.stream(
        user_input,
        config=config,
        stream_mode=["messages", "updates"],
    ):
        chunk_type, chunk_data = chunk
        state = main_agent.get_state(config)

        if chunk_type == "messages":
            format_chunk_message(chunk_data)

        if chunk_type == "updates" and isinstance(chunk_data, dict):
            if chunk_data.get("model"):
                last_message = chunk_data["model"]["messages"][-1]
                console.print(format_message(last_message))

            if "__interrupt__" in chunk_data and state.next == ("tools",):
                if confirm_pending_tool_calls(config):
                    ask_and_run(None, config)
                return


def main() -> None:
    # Один thread_id = один разговор с общей историей.
    config = {"configurable": {"thread_id": "shopping-dialog-1"}}

    console.print(
        "[bold green]Чат-агент по покупкам.[/bold green] "
        "Введите [bold]exit[/bold] для выхода."
    )

    while True:
        user_text = input("\nВы: ").strip()
        if user_text.lower() == "exit":
            break

        ask_and_run(
            {"messages": [{"role": "human", "content": user_text}]},
            config,
        )


if __name__ == "__main__":
    main()
