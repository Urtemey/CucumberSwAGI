# Задание 5 — кастомное прерывание через interrupt() в LangGraph.
# Делаю граф из одного узла, который "спрашивает" пользователя и
# возобновляется после Command(resume=...).

import sys
from typing import TypedDict

import questionary
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt

# винда любит cp1251, поэтому форсим utf-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")


class State(TypedDict, total=False):
    foo: str           # просто что-то начальное, чтобы было видно изменение
    human_value: str   # сюда положим ответ юзера


def node(state):
    # передаю в interrupt структурированный объект — потом он же
    # прилетит обратно (после resume), но уже с полем answer
    payload = interrupt({
        "type": "alert",
        "question": "Уверены что хотите продолжить?",
        "allow_responds": ["approve", "reject"],
    })

    print(f"!!! {payload['type']} !!!")
    print(f"> Received an input from the interrupt: {payload['answer']}")
    return {"human_value": payload["answer"]}


# граф из одного узла
g = StateGraph(State)
g.add_node("node", node)
g.add_edge(START, "node")
graph = g.compile(checkpointer=InMemorySaver())


def show_menu(question, options):
    # questionary удобный, но в неинтерактивной консоли падает — делаю фоллбэк
    if sys.stdin.isatty():
        return questionary.select(question, choices=options).ask()
    print(question)
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    raw = input("> ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw


def run():
    config = {"configurable": {"thread_id": "session-1"}}
    initial = {"foo": "начальное значение"}

    # первый прогон — до прерывания
    for chunk in graph.stream(initial, config=config):
        if "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value

            print("Произошла остановка")
            print(payload)

            answer = show_menu(payload["question"], payload["allow_responds"])

            # докидываю поле в тот же объект и возобновляю граф
            payload["answer"] = answer
            for resumed in graph.stream(Command(resume=payload), config=config):
                print(resumed)
        else:
            print(chunk)

    final = graph.get_state(config).values
    print("\nИтоговое состояние:", final)


if __name__ == "__main__":
    run()
