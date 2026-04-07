"""
Задание 1. Иерархический AI-агент на LangChain — помощник по покупкам.

Главный агент принимает список продуктов и город, для каждого продукта
вызывает инструмент get_price, внутри которого работает субагент,
оценивающий реалистичную цену. Результат возвращается через .invoke().
"""

import sys

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import SecretStr

# UTF-8 для корректной кириллицы в Windows-консоли
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# 1. Подключение к локальной LLM через LM Studio (OpenAI-совместимый API)
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
    продукта в конкретном городе России. Возвращает строку markdown-таблицы
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


# 4. Форматирование сообщения цепочки
def format_message(message) -> str:
    if getattr(message, "content", None):
        return message.content
    if getattr(message, "tool_calls", None):
        call = message.tool_calls[0]
        return f"{call['name']}({call['args']})"
    return repr(message)


def main() -> None:
    question = (
        "Помоги составить список покупок: молоко, хлеб, яблоки. "
        "Я нахожусь в Казани."
    )

    answer = main_agent.invoke({
        "messages": [
            {"role": "human", "content": question}
        ]
    })

    print("=" * 60)
    print("Цепочка сообщений агента:")
    print("=" * 60)
    for msg in answer["messages"]:
        print(format_message(msg))
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Финальный ответ:")
    print("=" * 60)
    print(answer["messages"][-1].content)


if __name__ == "__main__":
    main()
