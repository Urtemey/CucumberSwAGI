# Задание 10. RAG-агент: LM Studio + Qdrant + интерактивный CLI.

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_tools import add_to_knowledge_base, search_knowledge_base


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


SYSTEM_PROMPT = (
    "Ты ассистент с локальной базой знаний. Используй search_knowledge_base "
    "когда ответ может быть в загруженных документах — делай это до ответа по памяти. "
    "Если пользователь даёт новый материал и просит запомнить — вызывай "
    "add_to_knowledge_base. Опирайся на найденные фрагменты, если их нет — "
    "честно говори, что в базе этого нет."
)


def build_agent():
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "google/gemma-4-26b-a4b"),
        base_url=os.getenv("LLM_BASE_URL", "http://192.168.0.120:1234/v1"),
        api_key=SecretStr(os.getenv("LLM_API_KEY", "lm-studio")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )
    return create_agent(
        model=llm,
        tools=[search_knowledge_base, add_to_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
    )


def _read_multiline(prompt: str) -> str:
    print(prompt)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def handle_command(raw: str) -> bool:
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        print("Пока!")
        sys.exit(0)

    if cmd == "/add":
        title = arg.strip() or "без названия"
        content = _read_multiline(
            f"Введите содержимое документа '{title}' (пустая строка — конец):"
        )
        if not content.strip():
            print("Пустой документ, ничего не добавили.")
            return True
        print(add_to_knowledge_base.invoke({"content": content, "title": title}))
        return True

    if cmd == "/search":
        query = arg.strip() or input("Запрос: ").strip()
        if not query:
            print("Пустой запрос.")
            return True
        print(search_knowledge_base.invoke({"query": query, "max_results": 4}))
        return True

    return False


def chat_loop() -> None:
    print("RAG-агент готов. Команды: /add <title>, /search <запрос>, /quit.")
    print("Всё остальное идёт в агента — он сам решает, нужен ли поиск.\n")

    agent = build_agent()

    while True:
        try:
            user = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not user:
            continue

        if user.startswith("/") and handle_command(user):
            continue

        try:
            result = agent.invoke({"messages": [{"role": "human", "content": user}]})
            last = result["messages"][-1]
            print(f"Агент: {getattr(last, 'content', last)}\n")
        except Exception as exc:
            # частая причина — не поднят LM Studio Local Server или нет нужной модели
            print(f"[ошибка] {exc}")
            print("Проверь .env (LLM_BASE_URL для LM Studio, OLLAMA_BASE_URL для Ollama) "
                  "и что подняты chat- и embedding-модели.\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    chat_loop()


if __name__ == "__main__":
    main()
