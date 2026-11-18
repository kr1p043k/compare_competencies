"""Барьер: не плодить самодельные HTTP-клиенты и не блокировать event loop (7.8.4).

Цель: новые места обязаны использовать `src/infrastructure/http_client.py`
(настраиваемые connect/read-таймауты + ретраи из настроек), а `requests.*`
нельзя вызывать внутри `async def`.

Существующие самодельные клиенты зафиксированы в ALLOWED_ADHOC как техдолг:
их перевод на общий клиент требует правки тестов, которые патчат
`httpx.Client`/`httpx.AsyncClient` (передача `transport=` конфликтует).
Список должен только сокращаться.
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Техдолг (только сокращать). Причина: тесты подменяют httpx.Client с transport=.
ALLOWED_ADHOC = {
    "src/api_pkg/routers/admin.py",
    "src/api_pkg/routers/academic.py",
    "src/infrastructure/course_trends.py",
    "src/notifications/telegram.py",
}

ADHOC_CLIENT = re.compile(r"httpx\.(?:Async)?Client\(")
BLOCKING_ATTRS = {"get", "post", "put", "delete", "patch", "request"}


def _rel(p: pathlib.Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def test_no_new_adhoc_httpx_clients():
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = _rel(p)
        if rel == "src/infrastructure/http_client.py" or rel in ALLOWED_ADHOC:
            continue
        if ADHOC_CLIENT.search(p.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(rel)
    assert not offenders, (
        "Новый самодельный httpx-клиент (используйте new_async_client/new_sync_client): "
        + ", ".join(sorted(offenders))
    )


def test_allowlist_only_shrinks():
    """Файлы из ALLOWED_ADHOC больше не должны содержать самодельных клиентов.

    Если файл переведён на общий клиент - уберите его из ALLOWED_ADHOC
    (иначе список не сокращается и барьер слабеет).
    """
    stale = [
        rel
        for rel in sorted(ALLOWED_ADHOC)
        if not ADHOC_CLIENT.search((ROOT / rel).read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not stale, "Уберите из ALLOWED_ADHOC (уже на общем клиенте): " + ", ".join(stale)


def _blocking_calls_in_async(path: pathlib.Path) -> list[str]:
    """Найти `requests.<verb>(...)` внутри async def (блокирует event loop)."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    found: list[str] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AsyncFunctionDef):
                for sub in ast.walk(child):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "requests"
                        and sub.func.attr in BLOCKING_ATTRS
                    ):
                        found.append(f"{_rel(path)}:{sub.lineno}")
                walk(child)
            elif not isinstance(child, ast.FunctionDef):
                walk(child)

    walk(tree)
    return found


def test_no_blocking_requests_inside_async_functions():
    offenders: list[str] = []
    for p in (SRC / "api_pkg" / "routers").rglob("*.py"):
        offenders.extend(_blocking_calls_in_async(p))
    assert not offenders, (
        "Блокирующий requests внутри async def (уносите в asyncio.to_thread или "
        "используйте общий async-клиент): " + ", ".join(sorted(offenders))
    )
