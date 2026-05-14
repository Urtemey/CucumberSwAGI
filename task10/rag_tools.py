# RAG-инструменты: поиск и добавление в базу знаний.

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from vector_store import (
    QdrantVectorStore,
    add_documents,
    make_vector_store,
    split_into_documents,
)


_store: Optional[QdrantVectorStore] = None


def get_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = make_vector_store()
    return _store


@tool
def search_knowledge_base(query: str, max_results: int = 4) -> str:
    """Семантический поиск по локальной базе знаний.

    Используй когда ответ может быть в загруженных документах.
    Возвращает найденные фрагменты с метаданными и score (меньше — ближе).
    """
    if not query or not query.strip():
        return "Пустой запрос — нечего искать."

    results = get_store().similarity_search_with_score(query, k=max_results)
    if not results:
        return "В базе знаний ничего не найдено по этому запросу."

    blocks: list[str] = []
    for i, (doc, score) in enumerate(results, start=1):
        title = doc.metadata.get("title", "без названия")
        source = doc.metadata.get("source")
        header = f"[{i}] {title}"
        if source:
            header += f" ({source})"
        header += f" — score={score:.3f}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


@tool
def add_to_knowledge_base(content: str, title: str = "без названия") -> str:
    """Добавляет документ в базу знаний (с разбиением на чанки)."""
    if not content or not content.strip():
        return "Пустой текст — добавлять нечего."

    docs = split_into_documents(content, title=title)
    added = add_documents(get_store(), docs)
    return f"Документ '{title}' добавлен в базу: {added} чанков."
