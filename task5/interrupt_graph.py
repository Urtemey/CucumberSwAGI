"""
Задание 5. Кастомное прерывание в графе LangGraph (Human-in-the-Loop).

Граф из одного узла, который вызывает interrupt() со структурированным
объектом (тип, вопрос, варианты ответа). Цикл запуска ловит прерывание,
показывает вопрос через questionary.select и возобновляет граф через
Command(resume=...).
"""

import sys
from typing import TypedDict

import questionary
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt


def select(question: str, choices: list[str]) -> str:
    """questionary.select при наличии TTY, иначе обычный input()."""
    if sys.stdin.isatty():
        return questionary.select(question, choices=choices).ask()
    print(question)
    for i, choice in enumerate(choices, 1):
        print(f"  {i}) {choice}")
    raw = input("Ваш выбор: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1]
    return raw

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")


# 1. Состояние графа
class State(TypedDict, total=False):
    foo: str
    human_value: str


# 2. Узел с прерыванием
def confirm_node(state: State) -> State:
    payload = interrupt({
        "type": "alert",
        "question": "Уверены, что хотите продолжить?",
        "allow_responds": ["approve", "reject"],
    })

    # После resume payload — это словарь с добавленным полем 'answer'.
    return {"human_value": payload["answer"]}


# 3. Сборка графа
builder = StateGraph(State)
builder.add_node("node", confirm_node)
builder.add_edge(START, "node")

graph = builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    config = {"configurable": {"thread_id": "session-1"}}
    initial_state: State = {"foo": "начальное значение"}

    # Первый запуск — до прерывания
    final_chunk = None
    for chunk in graph.stream(initial_state, config=config):
        if "__interrupt__" in chunk:
            interrupt_obj = chunk["__interrupt__"][0]
            payload = interrupt_obj.value

            print("Произошла остановка")
            print(payload)
            print(f"!!! {payload['type']} !!!")

            answer = select(payload["question"], payload["allow_responds"])

            print(f"> Received an input from the interrupt: {answer}")

            # Дополняем payload ответом и возобновляем граф
            payload["answer"] = answer
            for resumed in graph.stream(
                Command(resume=payload),
                config=config,
            ):
                final_chunk = resumed
                print(resumed)
        else:
            final_chunk = chunk
            print(chunk)

    # Финальное состояние
    final_state = graph.get_state(config).values
    print("\nИтоговое состояние:", final_state)


if __name__ == "__main__":
    main()
