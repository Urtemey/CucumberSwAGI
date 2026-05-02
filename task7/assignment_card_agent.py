"""
Task 7. Raw assignment text -> flat validated assignment card.

The script takes one informal assignment description, sends it through a
LangChain chain, validates the result with a PydanticOutputParser, then prints
both the structured object and a short human-readable summary.
"""

from __future__ import annotations

import json
import os
import sys
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr


def configure_console_encoding() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


configure_console_encoding()


class AssignmentCard(BaseModel):
    """Flat card extracted from a single assignment text."""

    title: str = Field(
        description="Short title of the assignment, 3-10 words."
    )
    subject: str = Field(
        description="Course topic or subject area mentioned or inferred from the text."
    )
    deadline_hint: str = Field(
        description="Deadline in Russian free form. Use 'не указано' if the text has no deadline."
    )
    deliverable_type: str = Field(
        description=(
            "What the student must submit, in Russian: отчёт, код, презентация, "
            "эссе, ноутбук, смешанный формат, другое."
        )
    )
    grading_hints: list[str] = Field(
        default_factory=list,
        description="Flat list of grading criteria explicitly mentioned in the text.",
    )


def build_llm() -> ChatOpenAI:
    """Create a model client; defaults target the local LM Studio server."""
    model = os.getenv("OPENAI_MODEL", "qwen/qwen3-vl-4b")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("OPENAI_API_KEY", "fake")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        temperature=0,
    )


parser = PydanticOutputParser(pydantic_object=AssignmentCard)

prompt = PromptTemplate(
    template=(
        "Ты преобразуешь один сырой русский текст задания в плоскую карточку данных.\n"
        "Не задавай уточняющих вопросов. Не добавляй вложенные структуры.\n"
        "Верни только один объект в формате парсера.\n"
        "Все текстовые значения заполняй на русском языке.\n"
        "deadline_hint сохраняй в естественной форме из текста, например 'к пятнице'.\n"
        "deliverable_type пиши на русском, например 'отчёт', 'код', 'презентация'.\n"
        "grading_hints пиши короткими русскими фразами.\n"
        "Если поле отсутствует, напиши короткий явный placeholder: 'не указано'.\n\n"
        "Сырой текст задания:\n{assignment_text}\n\n"
        "{format_instructions}"
    ),
    input_variables=["assignment_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def parse_assignment(assignment_text: str) -> AssignmentCard:
    chain = prompt | build_llm() | parser
    return chain.invoke({"assignment_text": assignment_text})


def print_result(card: AssignmentCard) -> None:
    print("\nValidated object:")
    print(json.dumps(card.model_dump(), ensure_ascii=False, indent=2))

    print("\nSummary:")
    print(f"- title: {card.title}")
    print(f"- subject: {card.subject}")
    print(f"- deadline_hint: {card.deadline_hint}")
    print(f"- deliverable_type: {card.deliverable_type}")
    print(f"- grading_hints: {', '.join(card.grading_hints) or 'not specified'}")


def read_input_text() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    print("Введите текст задания. Для примера нажмите Enter:")
    user_text = sys.stdin.readline().strip()
    if user_text:
        return user_text

    return (
        "Сдайте к пятнице мини-отчёт по LangChain: 2 страницы, упор на агентов. "
        "Оценка: за полноту и за пример кода."
    )


def main() -> None:
    assignment_text = read_input_text()
    card = parse_assignment(assignment_text)
    print_result(card)


if __name__ == "__main__":
    main()
