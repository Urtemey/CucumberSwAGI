# Qdrant + эмбеддинги через LM Studio (OpenAI-совместимый API).

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(raw: str | None, default: Path) -> str:
    if not raw:
        return str(default)
    if raw == ":memory:":
        return raw
    p = Path(raw)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())


COLLECTION_NAME = os.getenv("RAG_COLLECTION", "knowledge_base")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://192.168.0.120:1234/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "lm-studio")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
QDRANT_PATH = _resolve_path(
    os.getenv("RAG_QDRANT_PATH"),
    Path(__file__).resolve().parent / "qdrant_data",
)


def make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=EMBED_BASE_URL,
        api_key=SecretStr(EMBED_API_KEY),
        check_embedding_ctx_length=False,  # LM Studio не сообщает context length
    )


def make_qdrant_client() -> QdrantClient:
    if QDRANT_PATH == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def make_vector_store() -> QdrantVectorStore:
    client = make_qdrant_client()
    ensure_collection(client)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=make_embeddings(),
    )


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_into_documents(
    content: str,
    *,
    title: str = "без названия",
    source: str | None = None,
    extra_metadata: dict | None = None,
) -> list[Document]:
    if not content or not content.strip():
        return []

    chunks = _splitter.split_text(content)
    meta_base: dict = {"title": title}
    if source:
        meta_base["source"] = source
    if extra_metadata:
        meta_base.update(extra_metadata)

    docs: list[Document] = []
    for i, chunk in enumerate(chunks):
        meta = dict(meta_base)
        meta["chunk"] = i
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def add_documents(store: QdrantVectorStore, docs: Iterable[Document]) -> int:
    docs_list = list(docs)
    if not docs_list:
        return 0
    store.add_documents(docs_list)
    return len(docs_list)
