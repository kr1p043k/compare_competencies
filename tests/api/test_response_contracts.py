"""Контракт ответов API с объявленными схемами (BACKLOG 7.5.2).

Оффлайн-проверки по `app.openapi()` (без БД и сети):
- каждый `$ref` в JSON-схеме 2xx-ответа резолвится в `components.schemas`;
- каждая компонентная схема структурно валидна (есть type/$ref/allOf/oneOf/
  anyOf/enum/properties) - ловит «пустые» схемы после рефакторинга;
- число GET-роутов с объявленной JSON-схемой не падает ниже порога - защита от
  тихого удаления `response_model` (сама полнота покрытия - задача 6.2.1).

Живую валидацию тел ответов против схем здесь не делаем: для этого нужен
поднятый стенд с БД; см. контрактные проверки в CI.
"""
from __future__ import annotations

from typing import Any

from src.api_pkg import create_app

#: Текущее значение ~50; держим запас, чтобы порог не был хрупким.
MIN_DECLARED_GET_SCHEMAS = 40

_METHODS = {"get", "post", "put", "patch", "delete"}


def _spec() -> dict[str, Any]:
    return create_app().openapi()


def _json_schema(op: dict[str, Any]) -> dict[str, Any] | None:
    for code in ("200", "201"):
        resp = op.get("responses", {}).get(code)
        if resp and "application/json" in resp.get("content", {}):
            return resp["content"]["application/json"].get("schema")
    return None


def _collect_refs(schema: Any, acc: set[str]) -> None:
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            acc.add(ref)
        for value in schema.values():
            _collect_refs(value, acc)
    elif isinstance(schema, list):
        for value in schema:
            _collect_refs(value, acc)


def test_all_response_refs_resolve() -> None:
    spec = _spec()
    schemas = spec.get("components", {}).get("schemas", {})
    missing: list[str] = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if method not in _METHODS:
                continue
            schema = _json_schema(op)
            if not schema:
                continue
            refs: set[str] = set()
            _collect_refs(schema, refs)
            for ref in refs:
                name = ref.rsplit("/", 1)[-1]
                if name not in schemas:
                    missing.append(f"{method.upper()} {path} -> {name}")
    assert not missing, "Ссылки на несуществующие схемы:\n" + "\n".join(missing)


def test_component_schemas_are_wellformed() -> None:
    schemas = _spec().get("components", {}).get("schemas", {})
    assert schemas, "В OpenAPI нет components.schemas"
    broken: list[str] = []
    meaningful = {"type", "$ref", "allOf", "oneOf", "anyOf", "enum", "properties", "items"}
    for name, schema in schemas.items():
        if not isinstance(schema, dict):
            broken.append(f"{name}: не объект")
            continue
        if not (meaningful & set(schema)):
            broken.append(f"{name}: нет type/$ref/properties")
    assert not broken, "Неполные схемы:\n" + "\n".join(broken)


def test_declared_get_schema_floor() -> None:
    spec = _spec()
    declared = sum(
        1
        for ops in spec["paths"].values()
        if "get" in ops and _json_schema(ops["get"])
    )
    assert declared >= MIN_DECLARED_GET_SCHEMAS, (
        f"GET-роутов с объявленной JSON-схемой стало {declared} "
        f"(порог {MIN_DECLARED_GET_SCHEMAS}) - проверьте удаление response_model"
    )
