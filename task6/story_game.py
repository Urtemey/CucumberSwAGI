"""
Задание 6. Интерактивная история «выбери свою историю» с LLM и interrupt.

Граф из одного узла:
1) LLM придумывает завязку и 3 варианта поступка героя;
2) interrupt() ставит граф на паузу — пользователь выбирает вариант
   через questionary.select;
3) LLM дописывает короткую концовку с учётом выбора;
4) Возвращается итоговое состояние (завязка, выбор, концовка).
"""

import re
import sys
from typing import TypedDict

import questionary
from langchain_openai import ChatOpenAI


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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from pydantic import SecretStr

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")


# 1. LLM
llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.9,
)


# 2. Состояние графа
class StoryState(TypedDict, total=False):
    theme: str
    intro: str
    options: list[str]
    choice: str
    ending: str


# 3. Парсер ответа LLM с завязкой и пронумерованными вариантами
def parse_intro_and_options(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    intro_lines: list[str] = []
    options: list[str] = []

    option_pattern = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+)$")

    for line in lines:
        match = option_pattern.match(line)
        if match:
            options.append(match.group(1).strip())
        elif not options:
            intro_lines.append(line)

    intro = " ".join(intro_lines).strip()
    # На случай если модель не пронумеровала варианты — берём последние 3 строки
    if len(options) < 3 and len(lines) >= 3:
        options = [lines[-3], lines[-2], lines[-1]]
        intro = " ".join(lines[:-3]).strip() or intro

    return intro, options[:3]


# 4. Узел графа: завязка → прерывание → концовка
def story_node(state: StoryState) -> StoryState:
    theme = state["theme"]

    intro_prompt = (
        f"Тема: {theme}. Придумай короткую завязку (2-3 предложения) "
        "для интерактивной истории и ровно 3 варианта поступка героя. "
        "Ответь строго в формате:\n"
        "<текст завязки>\n"
        "1) <вариант 1>\n"
        "2) <вариант 2>\n"
        "3) <вариант 3>"
    )
    intro_response = llm.invoke(intro_prompt).content
    intro, options = parse_intro_and_options(intro_response)

    print("\n[LLM] " + intro + "\n")

    payload = interrupt({
        "type": "choice",
        "question": "Что делаем?",
        "intro": intro,
        "options": options,
    })

    choice = payload["answer"]

    ending_prompt = (
        f"Завязка: {intro}\n"
        f"Выбор героя: {choice}\n"
        "Допиши короткую концовку истории в 2-3 предложениях."
    )
    ending = llm.invoke(ending_prompt).content

    return {
        "intro": intro,
        "options": options,
        "choice": choice,
        "ending": ending,
    }


# 5. Сборка графа
builder = StateGraph(StoryState)
builder.add_node("story", story_node)
builder.add_edge(START, "story")

graph = builder.compile(checkpointer=InMemorySaver())


def run_story(theme: str) -> None:
    config = {"configurable": {"thread_id": f"story-{theme}"}}
    initial: StoryState = {"theme": theme}

    print(f"\nТема: {theme}")

    for chunk in graph.stream(initial, config=config):
        if "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value

            answer = select(payload["question"], payload["options"])

            payload["answer"] = answer

            for resumed in graph.stream(
                Command(resume=payload),
                config=config,
            ):
                pass

    final_state = graph.get_state(config).values
    print("\n[LLM] " + final_state["ending"])
    print("\n--- Итоговое состояние ---")
    print(f"Тема:    {final_state['theme']}")
    print(f"Завязка: {final_state['intro']}")
    print(f"Выбор:   {final_state['choice']}")
    print(f"Концовка:{final_state['ending']}")


def main() -> None:
    theme = input("Введите тему истории (например, 'космический кот'): ").strip()
    if not theme:
        theme = "космический кот"
    run_story(theme)


if __name__ == "__main__":
    main()
