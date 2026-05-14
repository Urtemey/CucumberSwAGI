# Задание 9. MCP-сервер памяти. Хранилище — JSON-файл с timestamp.

from __future__ import annotations

import fnmatch
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_storage_path() -> Path:
    raw = os.getenv("MEMORY_STORAGE_PATH")
    if not raw:
        return Path(__file__).resolve().parent / "memory_data.json"
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


DEFAULT_STORAGE = _resolve_storage_path()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _validate_key_part(part: str, what: str = "key") -> str:
    if not isinstance(part, str) or not part.strip():
        raise ValueError(f"{what} должен быть непустой строкой")
    if "/" in part or "\\" in part or ".." in part:
        raise ValueError(f"Недопустимые символы в {what}: {part!r}")
    return part


class MemoryServer:
    def __init__(self, storage_path: Path | str = DEFAULT_STORAGE):
        self.mcp = FastMCP("Memory-Server")
        self.storage_path = Path(storage_path)
        self._register_tools()

    def _load_memory(self) -> dict:
        if not self.storage_path.exists():
            return {}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save_memory(self, data: dict) -> None:
        parent = self.storage_path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        # атомарная запись: tmp -> replace, чтобы не получить битый JSON на ctrl-c
        tmp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.storage_path)

    # инструменты регистрирую внутри __init__, потому что @self.mcp.tool()
    # на уровне класса (как в методичке) не работает — self там не существует
    def _register_tools(self) -> None:
        mcp = self.mcp

        @mcp.tool()
        def save(key: str, value: Any) -> bool:
            """Сохраняет значение по ключу в память сервера."""
            try:
                _validate_key_part(key, "key")
                data = self._load_memory()
                data[key] = {"value": value, "timestamp": _now_iso()}
                self._save_memory(data)
                return True
            except Exception:
                return False

        @mcp.tool()
        def get(key: str) -> Optional[dict]:
            """Возвращает значение по ключу с метаданными."""
            data = self._load_memory()
            entry = data.get(key)
            if entry is None:
                return None
            return {"key": key, "value": entry["value"], "timestamp": entry["timestamp"]}

        @mcp.tool()
        def delete(key: str) -> bool:
            """Удаляет ключ из памяти сервера."""
            data = self._load_memory()
            if key not in data:
                return False
            del data[key]
            self._save_memory(data)
            return True

        @mcp.tool()
        def list_keys(pattern: str = "*") -> list[str]:
            """Возвращает список всех ключей с поддержкой wildcard-паттерна."""
            data = self._load_memory()
            return [k for k in data.keys() if fnmatch.fnmatch(k, pattern)]

        @mcp.tool()
        def save_with_namespace(key: str, value: Any, namespace: str = "default") -> bool:
            """Сохраняет значение в указанном namespace (ключ хранится как namespace:key)."""
            try:
                _validate_key_part(key, "key")
                _validate_key_part(namespace, "namespace")
                full_key = f"{namespace}:{key}"
                data = self._load_memory()
                data[full_key] = {"value": value, "timestamp": _now_iso()}
                self._save_memory(data)
                return True
            except Exception:
                return False

        @mcp.tool()
        def get_by_namespace(namespace: str = "default") -> list[dict]:
            """Возвращает все ключи из указанного namespace без префикса."""
            data = self._load_memory()
            prefix = f"{namespace}:"
            out: list[dict] = []
            for full_key, entry in data.items():
                if not full_key.startswith(prefix):
                    continue
                out.append({
                    "key": full_key[len(prefix):],
                    "namespace": namespace,
                    "value": entry["value"],
                    "timestamp": entry["timestamp"],
                })
            return out


def main() -> None:
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    server = MemoryServer()
    server.mcp.run(transport="stdio", show_banner=False, log_level="ERROR")


if __name__ == "__main__":
    main()
