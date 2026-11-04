"""API-contract tests for the year-scoped teacher KRM endpoints (auto/ux-fixes).

Covers `GET /teacher/krm/disciplines` and `GET /teacher/krm/scope-summary`.

The endpoints read a psycopg pool via `get_pool` and delegate scope logic to
`src.teacher_scope`. There the pool is replaced by a fake that records the SQL
and its parameters and returns exactly the rows the query would produce, so no
live database is needed. The checks are behavioural: response shape, the scope
classification chain, and the fact that `year` reaches the query (the fix under
test - before it the endpoint returned both 2024 and 2025 rows with duplicate
names).
"""
from __future__ import annotations

import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api_pkg.routers import teacher as t
from tests.conftest import open_all_gates

DIR_CODE = "09.03.02"
YEAR = 2025

SCOPE_SOURCES = {"custom", "methodology", "default"}

SUMMARY_KEYS = {
    "dir_code",
    "year",
    "total",
    "in_scope",
    "excluded_manual",
    "excluded_no_data",
    "analyzed",
    "excluded_manual_names",
    "excluded_no_data_names",
}


class FakePool:
    """Records every `fetch` call; returns canned rows, no DB."""

    def __init__(self, overrides=(), rows=()):
        self._overrides = list(overrides)
        self._rows = list(rows)
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, tuple(args)))
        if "discipline_scope" in sql:
            return list(self._overrides)
        return list(self._rows)

    @property
    def main_calls(self) -> list[tuple[str, tuple]]:
        """Calls that are not the scope-overrides lookup."""
        return [c for c in self.calls if "discipline_scope" not in c[0]]


@pytest.fixture(autouse=True)
def reset_limiter():
    with contextlib.suppress(Exception):
        t.limiter._storage.reset()
    yield


def _client(monkeypatch, pool: FakePool) -> TestClient:
    """Minimal app with only the teacher router + auth gates stubbed."""
    monkeypatch.setattr(t, "get_pool", lambda: pool)
    app = FastAPI()
    app.state.limiter = t.limiter
    app.include_router(t.router)
    open_all_gates(app)
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------
# /teacher/krm/disciplines
# --------------------------------------------------------------------------

DISC_ROWS = [
    {"name": "Базы данных и СУБД", "competencies_count": 3, "skills_count": 5,
     "knowledge_count": 3, "abilities_count": 2, "semester": 4},
    {"name": "Проектирование ИС", "competencies_count": 2, "skills_count": 4,
     "knowledge_count": 2, "abilities_count": 2, "semester": 6},
]
DISC_OVERRIDES = [{"discipline_name": "Проектирование ИС", "included": False}]


def test_disciplines_items_carry_year_and_scope_flags(monkeypatch):
    pool = FakePool(overrides=DISC_OVERRIDES, rows=DISC_ROWS)
    client = _client(monkeypatch, pool)

    resp = client.get("/teacher/krm/disciplines",
                      params={"dir_code": DIR_CODE, "year": YEAR})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body) == 2

    by_name = {item["name"]: item for item in body}
    assert set(by_name) == {"Базы данных и СУБД", "Проектирование ИС"}
    for item in body:
        assert item["year"] == YEAR
        assert isinstance(item["in_scope"], bool)
        assert item["scope_source"] in SCOPE_SOURCES

    # A row present in the UI overrides is reported as custom with its value.
    custom = by_name["Проектирование ИС"]
    assert custom["in_scope"] is False
    assert custom["scope_source"] == "custom"


def test_disciplines_passes_year_filter_to_query(monkeypatch):
    pool = FakePool(rows=DISC_ROWS)
    client = _client(monkeypatch, pool)

    resp = client.get("/teacher/krm/disciplines",
                      params={"dir_code": DIR_CODE, "year": YEAR})

    assert resp.status_code == 200, resp.text
    main = pool.main_calls
    assert len(main) == 1
    sql, args = main[0]
    assert "JOIN disciplines disc" in sql
    assert "disc.year" in sql, "the SQL must filter by intake year"
    assert args == (DIR_CODE, YEAR)


def test_disciplines_forwards_2024_not_hardcoded_2025(monkeypatch):
    """The year must come from the request, not the default."""
    rows_2024 = [{"name": "Базы данных и СУБД", "competencies_count": 3,
                  "skills_count": 5, "knowledge_count": 3,
                  "abilities_count": 2, "semester": 4}]
    pool = FakePool(rows=rows_2024)
    client = _client(monkeypatch, pool)

    resp = client.get("/teacher/krm/disciplines",
                      params={"dir_code": DIR_CODE, "year": 2024})

    assert resp.status_code == 200, resp.text
    assert resp.json()[0]["year"] == 2024
    assert pool.main_calls[0][1] == (DIR_CODE, 2024)


# --------------------------------------------------------------------------
# /teacher/krm/scope-summary
# --------------------------------------------------------------------------

SUMMARY_ROWS = [
    # in scope: has competencies, KSA and competency_skills links
    {"name": "Анализируемая", "comps": 2, "ksa": 5, "links": 4},
    # excluded by an explicit manual override
    {"name": "Ручное исключение", "comps": 1, "ksa": 1, "links": 1},
    # no KSA at all
    {"name": "Нет KSA", "comps": 1, "ksa": 0, "links": 0},
    # KSA exists but zero competency_skills links -> no analysis data
    {"name": "Связей KSA нет", "comps": 1, "ksa": 2, "links": 0},
]
SUMMARY_OVERRIDES = [
    {"discipline_name": "Ручное исключение", "included": False},
    {"discipline_name": "Нет KSA", "included": True},
    {"discipline_name": "Связей KSA нет", "included": True},
]


def test_scope_summary_shape_and_classification(monkeypatch):
    pool = FakePool(overrides=SUMMARY_OVERRIDES, rows=SUMMARY_ROWS)
    client = _client(monkeypatch, pool)

    resp = client.get("/teacher/krm/scope-summary",
                      params={"dir_code": DIR_CODE, "year": YEAR})

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert set(body) == SUMMARY_KEYS
    assert body["dir_code"] == DIR_CODE
    assert body["year"] == YEAR
    assert body["total"] == 4
    assert body["in_scope"] == 1
    assert body["excluded_manual"] == 1
    assert body["excluded_no_data"] == 2
    assert body["excluded_manual_names"] == ["Ручное исключение"]
    assert set(body["excluded_no_data_names"]) == {"Нет KSA", "Связей KSA нет"}
    assert body["analyzed"] is None or isinstance(body["analyzed"], int)


def test_scope_summary_zero_links_is_no_data_even_with_ksa(monkeypatch):
    """Regression: 2 KSA rows + 0 links must not look like a real 0% result."""
    pool = FakePool(overrides=SUMMARY_OVERRIDES, rows=SUMMARY_ROWS)
    client = _client(monkeypatch, pool)

    body = client.get("/teacher/krm/scope-summary",
                      params={"dir_code": DIR_CODE, "year": YEAR}).json()

    assert "Связей KSA нет" in body["excluded_no_data_names"]
    assert "Связей KSA нет" not in body["excluded_manual_names"]
    # And it is not counted as analysed.
    assert body["in_scope"] == 1


def test_scope_summary_passes_year_filter_to_query(monkeypatch):
    pool = FakePool(rows=SUMMARY_ROWS)
    client = _client(monkeypatch, pool)

    resp = client.get("/teacher/krm/scope-summary",
                      params={"dir_code": DIR_CODE, "year": YEAR})

    assert resp.status_code == 200, resp.text
    main = pool.main_calls
    assert len(main) == 1
    sql, args = main[0]
    assert "FROM disciplines disc" in sql
    assert "disc.year" in sql, "the SQL must filter by intake year"
    assert args == (DIR_CODE, YEAR)
