"""6.2.1: барьер покрытия `response_model`.

Каждый роут обязан объявлять контракт ответа: либо JSON-схему
(`response_model`), либо явный media-type через `response_class` + `responses`
для бинарных/SSE/текстовых ответов. Иначе OpenAPI врёт клиенту
(«application/json» без структуры) и ломается кодогенерация.

Изначально долг был 143 роута. На 2026-11-23 он закрыт полностью, поэтому
`KNOWN_WITHOUT_SCHEMA` пуст: новый роут без контракта валит тест сразу.

Если контракт действительно невозможен, роут добавляется в список ниже с
комментарием почему - список только сокращается.
"""
from __future__ import annotations

from src.api_pkg import create_app

METHODS = ("get", "post", "put", "patch", "delete")


def _routes_without_schema() -> set[tuple[str, str]]:
    spec = create_app().openapi()
    out: set[tuple[str, str]] = set()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if method not in METHODS:
                continue
            responses = op.get("responses", {})
            codes = [c for c in responses if c.startswith("2")]
            if not codes:
                out.add((method.upper(), path))
                continue
            content: dict = {}
            for code in codes:
                content.update(responses[code].get("content") or {})
            # Бинарный/SSE/текстовый ответ без JSON-схемы легитимен, если
            # объявлен явный media-type (content-type != application/json).
            if content and "application/json" not in content:
                continue
            if not content.get("application/json", {}).get("schema"):
                out.add((method.upper(), path))
    return out


#: Долг по 6.2.1 на 2026-11-23: закрыт, список пуст.
KNOWN_WITHOUT_SCHEMA: set[tuple[str, str]] = set()


def test_no_new_untyped_routes() -> None:
    untyped = _routes_without_schema()
    new = untyped - KNOWN_WITHOUT_SCHEMA
    assert not new, (
        "Роуты без объявленного контракта ответа (нужен response_model или "
        f"response_class + media-type): {sorted(new)}"
    )
