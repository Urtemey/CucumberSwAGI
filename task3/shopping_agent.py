# Задание №3: доработка shopping-agent — память и подтверждение действий.
# Поверх предыдущего задания добавляю:
#   - MemorySaver + общий thread_id => агент помнит контекст;
#   - interrupt_before=["tools"] => перед каждым tool-call стоп;
#   - rich.Console для вывода;
#   - ловлю __interrupt__ в стриме и спрашиваю подтверждение у юзера.

import sys

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr
from rich.console import Console

# чтобы кириллица не ломалась в винде
sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

console = Console()


llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)


# инструмент с субагентом — тот же что в заданиях 1-2
@tool
def get_price(product: str, city: str) -> str:
    """Возвращает примерную цену продукта в указанном городе."""
    sub_prompt = (
        "Ты эксперт по розничным ценам в РФ. Оцени реалистичную стоимость "
        "продукта в указанном городе. Ответ строго одной строкой markdown-таблицы:\n"
        "| Продукт | Цена (руб.) | Магазин |"
    )
    sub = create_agent(model=llm, tools=[], system_prompt=sub_prompt)
    res = sub.invoke({"messages": [
        {"role": "human", "content": f"Сколько стоит {product} в городе {city}?"},
    ]})
    return res["messages"][-1].content


# главный агент: память + пауза перед инструментами
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_price],
    system_prompt=(
        "Ты помощник по планированию покупок. "
        "Когда пользователь даёт список продуктов и город, вызывай get_price. "
        "За один шаг — только один вызов get_price на один продукт. "
        "В конце собери цены в markdown-таблицу и подведи итоговую стоимость."
    ),
    checkpointer=memory,        # без чекпоинтера пауза не сохранится
    interrupt_before=["tools"], # стоп перед узлом tools
)


# для красивого разделителя между шагами LangGraph
_step = 1


def print_stream_tokens(message_chunk):
    """Печатает токены потокового ответа модели."""
    global _step
    msg, meta = message_chunk
    cur = meta.get("langgraph_step")
    if cur != _step:
        _step = cur
        console.print("\n --- --- --- \n")
    if getattr(msg, "content", None):
        console.print(msg.content, end="")


def print_model_update(msg):
    # если модель вместо текста сгенерила tool_calls — печатаем их
    calls = getattr(msg, "tool_calls", None) or []
    for c in calls:
        console.print(f"{c['name']}({c['args']})")
    # либо обычный текстовый ответ
    if getattr(msg, "content", None):
        # содержимое уже шло токенами в print_stream_tokens — не дублируем
        pass


def ask_confirmation(config):
    """Смотрим, что агент хочет вызвать, спрашиваем подтверждение."""
    state = agent.get_state(config)
    last = state.values["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        # на всякий случай: пауза есть, а tool_calls нет — выходим
        console.print("[yellow]Пауза без tool_calls, продолжать нечего.[/yellow]")
        return False

    for call in calls:
        console.print(f"Агент хочет вызвать утилиту {call['name']}({call['args']})")
        ans = input("Разрешить? (Y/n): ").strip().lower()
        if ans not in ("", "y", "yes", "д", "да"):
            console.print("Отменено")
            return False
    return True


def ask_and_run(user_input, config):
    """user_input: {'messages': [...]} либо None — продолжить из паузы."""
    for chunk in agent.stream(user_input, config=config,
                              stream_mode=["messages", "updates"]):
        chunk_type, chunk_data = chunk
        state = agent.get_state(config)

        if chunk_type == "messages":
            print_stream_tokens(chunk_data)

        if chunk_type == "updates" and isinstance(chunk_data, dict):
            if chunk_data.get("model"):
                print_model_update(chunk_data["model"]["messages"][-1])

            # сама обработка паузы
            if "__interrupt__" in chunk_data and state.next == ("tools",):
                if ask_confirmation(config):
                    # None = продолжить с того места где остановились
                    ask_and_run(None, config)
                return


def main():
    # один thread_id => одна общая история разговора (память)
    config = {"configurable": {"thread_id": "shopping-1"}}

    console.print("[bold green]Чат-агент по покупкам.[/bold green] "
                  "Введите [bold]exit[/bold] для выхода.")

    while True:
        text = input("\nВы: ")
        if text.strip().lower() == "exit":
            break
        ask_and_run({"messages": [{"role": "human", "content": text}]}, config)


if __name__ == "__main__":
    main()
