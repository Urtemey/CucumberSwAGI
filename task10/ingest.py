# Загрузка документов из директории в RAG-хранилище.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vector_store import add_documents, make_vector_store, split_into_documents


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}


def iter_files(directory: Path, recursive: bool = True) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Директория не найдена: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Это не директория: {directory}")

    pattern = "**/*" if recursive else "*"
    files = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def ingest_directory(directory: Path, recursive: bool = True) -> None:
    files = iter_files(directory, recursive=recursive)
    if not files:
        print(f"В {directory} нет подходящих файлов ({', '.join(SUPPORTED_EXTENSIONS)}).")
        return

    store = make_vector_store()
    total = 0

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # бывает попадается cp1251 — пусть будет фоллбэк
            text = fp.read_text(encoding="cp1251", errors="replace")

        docs = split_into_documents(text, title=fp.stem, source=str(fp.relative_to(directory)))
        added = add_documents(store, docs)
        total += added
        print(f"  + {fp.name}: {added} чанков")

    print(f"\nГотово. Файлов: {len(files)}, чанков всего: {total}.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Загрузка документов в RAG-хранилище.")
    parser.add_argument("directory", type=Path, help="Каталог с документами.")
    parser.add_argument("--no-recursive", action="store_true", help="Не заходить в подпапки.")
    args = parser.parse_args()

    ingest_directory(args.directory, recursive=not args.no_recursive)


if __name__ == "__main__":
    main()
