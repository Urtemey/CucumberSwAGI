# Задание 6. Текстовая игра "выбери свою историю".
# Идея простая: 1) LLM пишет завязку и три варианта, 2) граф ставится
# на паузу через interrupt, 3) после выбора пользователя LLM пишет концовку.

import re
import sys
from typing import TypedDict

import questionary
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from pydantic import SecretStr

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")


# подключение к локальной LM Studio
llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.9,  # высокая температура — больше выдумки
)


class StoryState(TypedDict, total=False):
    theme: str
    intro: str
    options: list[str]
    choice: str
    ending: str


# модель иногда забивает на формат, поэтому разбираем "по-простому":
# что выглядит как пункт списка — то вариант, остальное — завязка
def parse_intro_and_options(text):
    options = []
    intro_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s*(.+)$", line)
        if m:
            options.append(m.group(1).strip())
        elif not options:
            intro_lines.append(line)

    intro = " ".join(intro_lines).strip()

    # запасной вариант: если разметки нет — возьмём 3 последние строки
    if len(options) < 3:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 4:
            options = lines[-3:]
            intro = " ".join(lines[:-3])

    return intro, options[:3]


def story_node(state):
    theme = state["theme"]

    # шаг 1: попросить у LLM завязку и варианты
    intro_prompt = (
        f"Тема: {theme}. Придумай короткую завязку (2-3 предложения) "
        "и ровно 3 варианта поступка героя. Отвечай в формате:\n"
        "<текст завязки>\n1) <вариант 1>\n2) <вариант 2>\n3) <вариант 3>"
    )
    raw = llm.invoke(intro_prompt).content
    intro, options = parse_intro_and_options(raw)

    print(f"\n[LLM] {intro}\n")

    # шаг 2: остановка с вопросом и вариантами
    payload = interrupt({
        "type": "choice",
        "question": "Что делаем?",
        "intro": intro,
        "options": options,
    })

    choice = payload["answer"]

    # шаг 3: концовка с учётом выбора
    ending_prompt = (
        f"Завязка: {intro}\n"
        f"Выбор героя: {choice}\n"
        "Допиши короткую концовку (2-3 предложения)."
    )
    ending = llm.invoke(ending_prompt).content

    return {
        "intro": intro,
        "options": options,
        "choice": choice,
        "ending": ending,
    }


# собираем граф (один узел, как в задании)
b = StateGraph(StoryState)
b.add_node("story", story_node)
b.add_edge(START, "story")
graph = b.compile(checkpointer=InMemorySaver())


def pick(question, options):
    if sys.stdin.isatty():
        return questionary.select(question, choices=options).ask()
    # фоллбэк для тестов / неинтерактивного запуска
    print(question)
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    raw = input("> ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw


def run_story(theme):
    cfg = {"configurable": {"thread_id": f"story-{theme}"}}
    print(f"\nТема: {theme}")

    for chunk in graph.stream({"theme": theme}, config=cfg):
        if "__interrupt__" not in chunk:
            continue

        payload = chunk["__interrupt__"][0].value
        answer = pick(payload["question"], payload["options"])
        payload["answer"] = answer

        # дочитываем оставшийся стрим (там сгенерится концовка)
        for _ in graph.stream(Command(resume=payload), config=cfg):
            pass

    final = graph.get_state(cfg).values
    print(f"\n[LLM] {final['ending']}")
    print("\n--- Итог ---")
    print(f"Тема:     {final['theme']}")
    print(f"Завязка:  {final['intro']}")
    print(f"Выбор:    {final['choice']}")
    print(f"Концовка: {final['ending']}")


def main():
    theme = input("Тема истории (например 'космический кот'): ").strip()
    if not theme:
        theme = "космический кот"
    run_story(theme)


if __name__ == "__main__":
    main()
