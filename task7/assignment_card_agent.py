# Задание 7. Превращаем сырой текст задания в "плоскую карточку".
# Один проход по цепочке: prompt -> llm -> PydanticOutputParser.
# Без диалога, без ручного разбора строк.

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")


# карточка задания — 5 полей из условия
class AssignmentCard(BaseModel):
    title: str = Field(
        description="Короткое название задания, 3-10 слов."
    )
    subject: str = Field(
        description="Предмет или тема курса (например 'LangChain', 'базы данных')."
    )
    deadline_hint: str = Field(
        description="Срок сдачи в свободной форме ('к пятнице', 'до 1 декабря'). "
                    "Если не указан — пиши 'не указано'."
    )
    deliverable_type: str = Field(
        description="Что именно сдаём: отчёт, код, презентация, эссе, ноутбук и т.п."
    )
    grading_hints: list[str] = Field(
        default_factory=list,
        description="Короткие пометки про оценивание, если есть в тексте. "
                    "Например ['полнота', 'пример кода']."
    )


# модель: по умолчанию локальный LM Studio, можно переопределить env-ми
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "google/gemma-4-26b-a4b"),
    base_url=os.getenv("LLM_BASE_URL", "http://192.168.0.120:1234/v1"),
    api_key=SecretStr(os.getenv("LLM_API_KEY", "lm-studio")),
    temperature=0,  # тут нам нужна предсказуемость, не творчество
)


parser = PydanticOutputParser(pydantic_object=AssignmentCard)

prompt = PromptTemplate(
    template=(
        "Ты получаешь сырой русский текст задания (как в переписке или на сайте курса) "
        "и превращаешь его в плоскую карточку.\n"
        "Не задавай уточняющих вопросов, не делай вложенных структур, "
        "не пиши пояснений. Один запрос — один ответ строго в нужном формате.\n"
        "Все строковые поля заполняй на русском.\n"
        "Если поле в тексте не упомянуто — ставь 'не указано'.\n\n"
        "Текст задания:\n{assignment_text}\n\n"
        "{format_instructions}"
    ),
    input_variables=["assignment_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# та самая цепочка из подсказки задания
chain = prompt | llm | parser


def read_text():
    # удобно для тестов: можно передать текст аргументом
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    print("Введите текст задания (Enter — пример из методички):")
    line = sys.stdin.readline().strip()
    if line:
        return line

    return (
        "Сдайте к пятнице мини-отчёт по LangChain: 2 страницы, упор на агентов. "
        "Оценка: за полноту и за пример кода."
    )


def main():
    text = read_text()
    card = chain.invoke({"assignment_text": text})

    # 1) валидированный объект
    print("\n--- Объект (model_dump) ---")
    print(json.dumps(card.model_dump(), ensure_ascii=False, indent=2))

    # 2) короткая человекочитаемая сводка
    print("\n--- Сводка ---")
    print(f"title: {card.title}")
    print(f"subject: {card.subject}")
    print(f"deadline_hint: {card.deadline_hint}")
    print(f"deliverable_type: {card.deliverable_type}")
    if card.grading_hints:
        print(f"grading_hints: {', '.join(card.grading_hints)}")
    else:
        print("grading_hints: не указано")


if __name__ == "__main__":
    main()
