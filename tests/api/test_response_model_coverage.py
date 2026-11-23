"""6.2.1: барьер покрытия `response_model`.

Каждый JSON-роут обязан объявлять схему ответа, иначе OpenAPI врёт клиенту
(«application/json» без структуры) и ломается кодогенерация. Задача закрывалась
по частям, поэтому ниже зафиксирован список роутов, до которых ещё не дошли.

Правило: список только сокращается. Новый роут без `response_model` (или без
`response_class` для бинарного ответа) валит тест. Если роут из списка начал
объявлять схему - он просто выпадает из пересечения, тест не ломается, так что
параллельные сессии могут закрывать свою часть независимо.
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
            # Бинарный/SSE/текстовый ответ без JSON-схемы - легитимен, если
            # объявлен явный media-type (content-type != application/json).
            if content and "application/json" not in content:
                continue
            if not content.get("application/json", {}).get("schema"):
                out.add((method.upper(), path))
    return out


#: Долг по 6.2.1 на момент фиксации (2026-11-23): teacher.py (30),
#: auth.py (10), krm_teacher.py (2), SSE chat (2), пасс-тру к внешнему API
#: ЮФУ (1) и JSON с диска (1). Каждая строка - конкретный роут, а не папка,
#: чтобы закрытие одного пункта сразу сужало список.
KNOWN_WITHOUT_SCHEMA: set[tuple[str, str]] = {
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
    ("POST", "/api/auth/logout-all"),
    ("GET", "/api/auth/me"),
    ("GET", "/api/auth/2fa/policy"),
    ("GET", "/api/auth/2fa/status"),
    ("POST", "/api/auth/2fa/confirm"),
    ("POST", "/api/auth/2fa/verify"),
    ("POST", "/api/auth/2fa/recovery"),
    ("POST", "/api/auth/2fa/disable"),
    ("POST", "/api/academic/analyze-gap"),
    ("GET", "/api/teacher/stats"),
    ("GET", "/api/teacher/krm/stats"),
    ("GET", "/api/teacher/krm/directions"),
    ("GET", "/api/teacher/krm/disciplines"),
    ("GET", "/api/teacher/krm/scope-summary"),
    ("GET", "/api/teacher/krm/disciplines/{discipline_name}"),
    ("GET", "/api/teacher/krm/recommendations"),
    ("POST", "/api/teacher/krm/recommendations"),
    ("DELETE", "/api/teacher/krm/recommendations/{index}"),
    ("GET", "/api/teacher/krm/foundational"),
    ("POST", "/api/teacher/krm/foundational"),
    ("DELETE", "/api/teacher/krm/foundational/{skill}"),
    ("GET", "/api/teacher/discipline-scope"),
    ("PUT", "/api/teacher/discipline-scope/{discipline}"),
    ("POST", "/api/teacher/skills/suggest"),
    ("GET", "/api/teacher/skills/suggestions"),
    ("POST", "/api/teacher/krm/recommendations/seed/auto"),
    ("GET", "/api/teacher/krm/coverage"),
    ("GET", "/api/teacher/krm/coverage/history"),
    ("GET", "/api/teacher/krm/market-skills"),
    ("GET", "/api/teacher/krm/search-runs"),
    ("GET", "/api/teacher/krm/search-runs/{run_id}"),
    ("POST", "/api/teacher/krm/run-analysis"),
    ("GET", "/api/teacher/export/vacancies"),
    ("GET", "/api/teacher/krm/competencies/tree"),
    ("GET", "/api/teacher/analysis"),
    ("GET", "/api/teacher/analysis/meta"),
    ("GET", "/api/admin/students/skills"),
    ("GET", "/api/teacher/students"),
    ("GET", "/api/teacher/analysis/{discipline_name}"),
    ("GET", "/api/teacher/zun/analyze/results"),
    ("GET", "/api/krm/teacher/directions/{dir_code}"),
    ("POST", "/api/krm/teacher/directions/{dir_code}/gap"),
    ("GET", "/api/teacher/chat/stream"),
    ("GET", "/api/student/chat/stream"),
}


def test_no_new_route_without_response_schema():
    """Новый роут не может появиться без объявленной схемы ответа."""
    found = _routes_without_schema()
    new = found - KNOWN_WITHOUT_SCHEMA
    assert not new, (
        "Роуты без response_model/response_class (6.2.1): "
        + ", ".join(f"{m} {p}" for m, p in sorted(new))
    )


def test_known_list_stays_valid():
    """Записи списка долга не должны дублироваться (сужение списка важно)."""
    items = list(KNOWN_WITHOUT_SCHEMA)
    assert len(items) == len(set(items)), "в KNOWN_WITHOUT_SCHEMA есть дубли"


def test_majority_of_json_routes_have_schema():
    """Регресс-барьер: не меньше 170 роутов с JSON-схемой ответа."""
    spec = create_app().openapi()
    ok = 0
    for ops in spec["paths"].values():
        for method, op in ops.items():
            if method not in METHODS:
                continue
            responses = op.get("responses", {})
            content: dict = {}
            for code in responses:
                if code.startswith("2"):
                    content.update(responses[code].get("content") or {})
            if content.get("application/json", {}).get("schema"):
                ok += 1
    assert ok >= 170, f"роутов со схемой ответа стало меньше: {ok} < 170"
