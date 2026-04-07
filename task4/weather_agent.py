"""
Задание 4. Human-in-the-Loop через HumanInTheLoopMiddleware.

При каждом вызове инструмента get_weather агент останавливается и
показывает пользователю запрос на подтверждение. Пользователь вводит
'a' (approve) или 'r' (reject), после чего выполнение возобновляется
через Command(resume={"decisions": [...]}).
"""

import sys

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import SecretStr

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")


# 1. LLM
llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)


# 2. Инструмент-заглушка погоды
@tool
def get_weather(city: str, date: str) -> str:
    """Возвращает прогноз погоды в указанном городе на указанную дату."""
    # Простейшая заглушка — в реальном проекте тут был бы API.
    return f"В городе {city} на {date}: облачно, +5°C, без осадков."


# 3. Агент с HumanInTheLoopMiddleware
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="Ты полезный ассистент, помогаешь узнать погоду.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_weather": {"allowed_decisions": ["approve", "reject"]},
            },
            description_prefix="Подтвердите вызов инструмента",
        ),
    ],
    checkpointer=memory,
)


def _ask_decisions(action_requests):
    """Показывает запросы и собирает решения approve/reject."""
    decisions = []
    for req in action_requests:
        print("\n--- Подтверждение ---")
        print(f"Инструмент: {req['name']}")
        print(f"Аргументы:  {req['args']}")
        if req.get("description"):
            print(f"Описание:   {req['description']}")

        while True:
            answer = input("a = approve, r = reject: ").strip().lower()
            if answer == "a":
                decisions.append({"type": "approve"})
                break
            if answer == "r":
                reason = input("Сообщение для агента (причина отказа): ")
                decisions.append({"type": "reject", "message": reason})
                break
            print("Введите 'a' или 'r'.")

    return decisions


def run_with_confirmations(user_message: str, config: dict) -> None:
    """Запускает агента и обрабатывает все паузы middleware."""
    result = agent.invoke(
        {"messages": [{"role": "human", "content": user_message}]},
        config=config,
    )

    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        action_requests = interrupt_value["action_requests"]

        decisions = _ask_decisions(action_requests)

        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
        )

    print("\nАгент:", result["messages"][-1].content)


def main() -> None:
    config = {"configurable": {"thread_id": "сессия-1"}}

    print("Чат с агентом-метеорологом. Введите 'exit' для выхода.")
    while True:
        user_input = input("\nВы: ")
        if user_input.strip() == "exit":
            break
        run_with_confirmations(user_input, config)


if __name__ == "__main__":
    main()
