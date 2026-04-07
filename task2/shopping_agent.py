import sys

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import SecretStr

# Принудительный UTF-8 для stdout/stderr, чтобы кириллица корректно
# отображалась в Windows-консоли без установки PYTHONIOENCODING.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# 1. Подключение к локальной LLM через LM Studio (OpenAI-совместимый API)
llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",  # имя модели, загруженной в LM Studio
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)


# 2. Инструмент с субагентом
@tool
def get_price(product: str, city: str) -> str:
    """Возвращает примерную цену продукта в указанном городе.

    Используй этот инструмент, когда нужно узнать стоимость конкретного
    продукта в конкретном городе России. Возвращает строку-таблицу
    с продуктом, ценой в рублях и магазином.
    """
    sub_system_prompt = (
        "Ты эксперт по розничным ценам в российских городах. "
        "На основе исторических данных о ценах оцени реалистичную "
        "стоимость указанного продукта в указанном городе. "
        "Ответь СТРОГО в виде одной строки markdown-таблицы формата:\n"
        "| Продукт | Цена (руб.) | Магазин |\n"
        "Без пояснений и комментариев."
    )

    price_subagent = create_agent(
        model=llm,
        tools=[],
        system_prompt=sub_system_prompt,
    )

    result = price_subagent.invoke({
        "messages": [
            {
                "role": "human",
                "content": f"Сколько стоит {product} в городе {city}?",
            }
        ]
    })

    return result["messages"][-1].content


# 3. Главный агент
main_agent = create_agent(
    model=llm,
    tools=[get_price],
    system_prompt=(
        "Ты помощник по планированию покупок. "
        "Когда пользователь даёт список продуктов и город, "
        "вызови инструмент get_price для каждого продукта по очереди, "
        "затем собери результаты в одну markdown-таблицу с заголовком "
        "'| Продукт | Цена (руб.) | Магазин |' и подведи итог в рублях."
    ),
)


# 4. Вспомогательная функция вывода сообщений
def format_message(message) -> str:
    if getattr(message, "content", None):
        return message.content
    if getattr(message, "tool_calls", None):
        call = message.tool_calls[0]
        return f"{call['name']}({call['args']})"
    return repr(message)


# 5. Stream-режим: вывод токенов по мере генерации
step = 1


def format_chunk_message(chunk) -> None:
    """Печатает фрагмент текста из потока 'messages' и разделитель между шагами."""
    global step
    message, meta = chunk

    if meta.get("langgraph_step") != step:
        step = meta["langgraph_step"]
        print("\n --- --- --- \n", flush=True)

    if message.content:
        print(message.content, end="", flush=True)


def main() -> None:
    question = (
        "Помоги составить список покупок: молоко, хлеб, яблоки. "
        "Я нахожусь в Казани."
    )

    stream = main_agent.stream(
        {"messages": [{"role": "human", "content": question}]},
        stream_mode=["messages", "updates"],
    )

    for chunk in stream:
        chunk_type, chunk_data = chunk

        if chunk_type == "messages":
            format_chunk_message(chunk_data)

        if chunk_type == "updates":
            if chunk_data.get("model"):
                last_message = chunk_data["model"]["messages"][-1]
                print(format_message(last_message))

    print()


if __name__ == "__main__":
    main()
