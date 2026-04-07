"""
Задание 3. Память разговора + подтверждение каждого вызова инструмента.

Доработка агента из задания 2:
- MemorySaver + thread_id для общей истории внутри одного разговора;
- interrupt_before=['tools'] — пауза перед каждым вызовом инструмента;
- rich.Console для красивого вывода;
- Чат-цикл с возможностью отменить вызов инструмента.
"""

import sys

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr
from rich.console import Console

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

console = Console()


# 1. LLM через LM Studio
llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)


# 2. Инструмент с субагентом
@tool
def get_price(product: str, city: str) -> str:
    """Возвращает примерную цену продукта в указанном городе.

    Используй этот инструмент, когда нужно узнать стоимость конкретного
    продукта в конкретном городе России. Возвращает строку markdown-таблицы.
    """
    sub_system_prompt = (
        "Ты эксперт по розничным ценам в российских городах. "
        "Оцени реалистичную стоимость продукта в городе. "
        "Ответь СТРОГО одной строкой markdown-таблицы формата:\n"
        "| Продукт | Цена (руб.) | Магазин |\n"
        "Без пояснений."
    )

    price_subagent = create_agent(
        model=llm,
        tools=[],
        system_prompt=sub_system_prompt,
    )

    result = price_subagent.invoke({
        "messages": [
            {"role": "human", "content": f"Сколько стоит {product} в городе {city}?"}
        ]
    })

    return result["messages"][-1].content


# 3. Главный агент с памятью и паузой перед инструментами
memory = MemorySaver()

main_agent = create_agent(
    model=llm,
    tools=[get_price],
    system_prompt=(
        "Ты помощник по планированию покупок. "
        "Когда пользователь даёт список продуктов и город, "
        "вызови инструмент get_price для каждого продукта по очереди, "
        "затем собери результаты в одну markdown-таблицу и подведи итог."
    ),
    checkpointer=memory,
    interrupt_before=["tools"],
)


# 4. Форматирование
step = 1


def format_message(message) -> str:
    if getattr(message, "content", None):
        return message.content
    if getattr(message, "tool_calls", None):
        call = message.tool_calls[0]
        return f"{call['name']}({call['args']})"
    return repr(message)


def format_chunk_message(chunk) -> None:
    global step
    message, meta = chunk
    if meta.get("langgraph_step") != step:
        step = meta["langgraph_step"]
        console.print("\n --- --- --- \n")
    if message.content:
        console.print(message.content, end="")


# 5. Запуск с обработкой паузы
def ask_and_run(user_input, config):
    """Запускает агента и обрабатывает паузу перед инструментом.

    user_input — словарь {"messages": [...]} для нового сообщения
                  или None для возобновления после паузы.
    """
    for chunk in main_agent.stream(
        user_input,
        config=config,
        stream_mode=["messages", "updates"],
    ):
        state = main_agent.get_state(config)
        chunk_type, chunk_data = chunk

        if chunk_type == "messages":
            format_chunk_message(chunk_data)

        if chunk_type == "updates":
            if isinstance(chunk_data, dict) and chunk_data.get("model"):
                last_message = chunk_data["model"]["messages"][-1]
                console.print(format_message(last_message))

            if isinstance(chunk_data, dict) and "__interrupt__" in chunk_data and state.next == ("tools",):
                _handle_interrupt(state, config)
                return

        if isinstance(chunk_data, tuple):
            continue


def _handle_interrupt(state, config):
    tool_call = state.values["messages"][-1].tool_calls[0]
    console.print(
        f"\n[yellow]Агент хочет вызвать утилиту "
        f"{tool_call['name']}({tool_call['args']})[/yellow]"
    )
    answer = input("Разрешить? (Y/n): ")
    if answer.lower().strip() in ("", "y"):
        ask_and_run(None, config)
    else:
        console.print("[red]Отменено[/red]")


def main() -> None:
    config = {"configurable": {"thread_id": "разговор-1"}}

    console.print(
        "[bold green]Чат-агент по покупкам.[/bold green] "
        "Введите 'exit' для выхода."
    )

    while True:
        user_input = input("\nВы: ")
        if user_input.strip() == "exit":
            break
        ask_and_run(
            {"messages": [{"role": "human", "content": user_input}]},
            config,
        )


if __name__ == "__main__":
    main()
