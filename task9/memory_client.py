# Задание 9. Клиент проверяет MCP-сервер: save/get/delete/namespace/list_keys.

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


SERVER_SCRIPT = Path(__file__).resolve().parent / "memory_server.py"


def _unwrap(result):
    # fastmcp возвращает CallToolResult, в разных версиях по-разному
    if hasattr(result, "data"):
        return result.data
    if hasattr(result, "content") and result.content:
        text = getattr(result.content[0], "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except Exception:
                return text
    return result


async def demo() -> None:
    client = Client(PythonStdioTransport(script_path=str(SERVER_SCRIPT), python_cmd=sys.executable))

    async with client:
        saved = await client.call_tool(
            "save_with_namespace",
            {"key": "username", "value": "Алексей", "namespace": "default"},
        )
        print(f"Сохранено: {_unwrap(saved)}")

        await client.call_tool(
            "save_with_namespace",
            {"key": "user_name", "value": "Алексей Иванов", "namespace": "default"},
        )
        await client.call_tool(
            "save_with_namespace",
            {"key": "session_token", "value": "abc123xyz", "namespace": "agent_1"},
        )
        await client.call_tool("save", {"key": "global_setting", "value": {"theme": "dark"}})

        ns_items = _unwrap(await client.call_tool("get_by_namespace", {"namespace": "default"}))
        print("Данные namespace 'default':")
        for item in ns_items:
            print(f"  {item['key']}: {item['value']}")

        keys = _unwrap(await client.call_tool("list_keys", {"pattern": "*name*"}))
        print(f"Ключи с 'name': {keys}")

        one = _unwrap(await client.call_tool("get", {"key": "default:username"}))
        print(f"get('default:username') -> {one}")

        deleted = _unwrap(await client.call_tool("delete", {"key": "default:username"}))
        print(f"delete('default:username') -> {deleted}")

        again = _unwrap(await client.call_tool("get", {"key": "default:username"}))
        print(f"get после удаления -> {again}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(demo())


if __name__ == "__main__":
    main()
