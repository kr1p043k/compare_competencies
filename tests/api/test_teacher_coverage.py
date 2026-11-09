"""GET /api/teacher/krm/coverage and /coverage/history (no live DB).

Covers `src/api_pkg/routers/teacher.krm_coverage` and `krm_coverage_history`.
Both handlers read the SQLAlchemy session through
`src.database.async_session_factory`, which they import INSIDE the function,
so the fake session is installed on the src.database module attribute: the
import happens at request time and picks the patched value up.

Neither route accepts a direction parameter (the other /teacher/krm/* routes
do). "Unknown direction" therefore maps to the no-data cases below: an empty
coverage table (latest date is absent), a history discipline filter that
matches nothing, and a stray dir_code query argument that must be ignored
instead of producing 4xx/5xx.

- happy path: the latest analysis date (a tz-aware DB value must be
  normalized to naive ISO, that is the regression guard for the old 500) and
  per-discipline rows with the documented item shape;
- role guard: the admin/teacher/rop gate stays real - anonymous -> 401,
  student -> 403 (only get_current_user is stubbed);
- gate declarations are asserted per route via route_gate_roles.

Conventions follow tests/api/test_teacher_history.py (role gates,
route_gate_roles, open_all_gates) and tests/api/test_coverage_teacher_krm.py
(fake session on src.database.async_session_factory, limiter reset).
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql.selectable import Select

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.teacher as teacher_mod
import src.database as database_mod
from src.api_pkg import create_app
from tests.conftest import open_all_gates, route_gate_roles

COVERAGE_ENDPOINT = "/api/teacher/krm/coverage"
HISTORY_ENDPOINT = "/api/teacher/krm/coverage/history"
COVERAGE_ITEM_KEYS = {"name", "total_skills", "matched_skills", "coverage_ratio"}
HISTORY_ITEM_KEYS = {"discipline", "coverage_ratio", "total_skills", "matched_skills", "analysis_date"}


class FakeResult:
    """SQLAlchemy-like Result: scalar() for the max-date query, all() for rows.

    rowcount is a class attribute because the background notifier
    (RequestLogMiddleware -> maybe_resolve_api_errors) reads it after the
    response (same reason as _FakeResult in test_coverage_teacher_krm.py).
    """

    rowcount = 0

    def __init__(self, scalar=None, rows=None):
        self._scalar = scalar
        self._rows = rows if rows is not None else []

    def scalar(self):
        return self._scalar

    def all(self):
        return list(self._rows)


class FakeSession:
    """Async CM session: each execute() pops the next prepared result.

    Empty-queue requests get a permissive empty result: the background
    notifier borrows the patched async_session_factory after the response
    and must not explode inside its task.
    """

    def __init__(self, results):
        self._results = list(results)
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, query, *args):
        self.queries.append(query)
        if not self._results:
            return FakeResult(scalar=None, rows=[])
        return self._results.pop(0)

    async def commit(self):
        return None


def _coverage_row(total_skills, matched, ratio, analysis_date):
    """A CoverageAnalysis-like row: attribute access is all the route needs."""
    return SimpleNamespace(
        total_skills=total_skills,
        market_matched_skills=matched,
        coverage_ratio=ratio,
        analysis_date=analysis_date,
    )


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Shared module-level limiter: keep this file self-contained (tests/api convention)."""
    with contextlib.suppress(Exception):
        teacher_mod.limiter._storage.reset()
    yield


def _client(monkeypatch, session):
    """Gates stubbed via open_all_gates (role=teacher): endpoint logic only."""
    app = open_all_gates(create_app(), role="teacher")
    monkeypatch.setattr(database_mod, "async_session_factory", lambda: session)
    return TestClient(app, raise_server_exceptions=False)


# ----------------------------------------------------- krm/coverage

def test_coverage_returns_latest_analysis_rows(monkeypatch):
    session = FakeSession(results=[
        FakeResult(scalar=datetime(2026, 10, 5, 12, 0, tzinfo=UTC)),
        FakeResult(rows=[
            (_coverage_row(10, 4, 0.4, datetime(2026, 10, 5, 12, 0)), "Databases"),
            (_coverage_row(50, 5, 0.1, datetime(2026, 10, 5, 12, 0)), "Informatics"),
        ]),
    ])
    r = _client(monkeypatch, session).get(COVERAGE_ENDPOINT)
    assert r.status_code == 200, r.text
    body = r.json()
    # A tz-aware date straight from the DB must be normalized to naive ISO
    # (no +00:00 suffix): regression guard for the old tz-mixed comparison.
    assert body["analysis_date"] == "2026-10-05T12:00:00", body["analysis_date"]
    assert body["disciplines"] == [
        {"name": "Databases", "total_skills": 10, "matched_skills": 4, "coverage_ratio": 0.4},
        {"name": "Informatics", "total_skills": 50, "matched_skills": 5, "coverage_ratio": 0.1},
    ]
    for item in body["disciplines"]:
        assert set(item) == COVERAGE_ITEM_KEYS
    # Exactly the handler's two SELECTs (max-date + rows); the Delete issued
    # by the background notifier borrows the same session and is excluded.
    selects = [q for q in session.queries if isinstance(q, Select)]
    assert len(selects) == 2, "coverage must issue the max-date and rows queries"


def test_coverage_empty_db_returns_null_date(monkeypatch):
    session = FakeSession(results=[FakeResult(scalar=None), FakeResult(rows=[])])
    r = _client(monkeypatch, session).get(COVERAGE_ENDPOINT)
    assert r.status_code == 200, r.text
    assert r.json() == {"analysis_date": None, "disciplines": []}


def test_coverage_ignores_unknown_dir_code_query_param(monkeypatch):
    """The route reads the whole table (no direction input): a stray dir_code
    argument must be ignored with the usual 200 response, not 4xx/5xx."""
    session = FakeSession(results=[
        FakeResult(scalar=datetime(2026, 10, 5, 12, 0)),
        FakeResult(rows=[(_coverage_row(8, 2, 0.25, datetime(2026, 10, 5, 12, 0)), "Databases")]),
    ])
    r = _client(monkeypatch, session).get(COVERAGE_ENDPOINT, params={"dir_code": "00.00.00"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis_date"] == "2026-10-05T12:00:00"
    assert body["disciplines"][0]["name"] == "Databases"


# ----------------------------------------------------- krm/coverage/history

def test_history_returns_rows_with_iso_dates(monkeypatch):
    session = FakeSession(results=[
        FakeResult(rows=[
            (_coverage_row(20, 8, 0.4, datetime(2026, 10, 5, 12, 0)), "Databases"),
            (_coverage_row(30, 9, 0.3, datetime(2026, 9, 1, 9, 30)), "Informatics"),
        ]),
    ])
    r = _client(monkeypatch, session).get(HISTORY_ENDPOINT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list) and len(body) == 2
    for item in body:
        assert set(item) == HISTORY_ITEM_KEYS
    assert body[0] == {
        "discipline": "Databases",
        "coverage_ratio": 0.4,
        "total_skills": 20,
        "matched_skills": 8,
        "analysis_date": "2026-10-05T12:00:00",
    }


def test_history_unknown_discipline_returns_empty_list(monkeypatch):
    session = FakeSession(results=[FakeResult(rows=[])])
    r = _client(monkeypatch, session).get(
        HISTORY_ENDPOINT, params={"discipline": "NoSuchDiscipline"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == []


def test_history_empty_db_returns_empty_list(monkeypatch):
    session = FakeSession(results=[FakeResult(rows=[])])
    r = _client(monkeypatch, session).get(HISTORY_ENDPOINT)
    assert r.status_code == 200, r.text
    assert r.json() == []


# ----------------------------------------------------- gates

def test_coverage_route_declares_teacher_role_gate():
    roles = route_gate_roles(create_app(), COVERAGE_ENDPOINT, "GET")
    assert ("admin", "teacher", "rop") in roles, f"actual gates: {roles}"


def test_history_route_declares_teacher_role_gate():
    roles = route_gate_roles(create_app(), HISTORY_ENDPOINT, "GET")
    assert ("admin", "teacher", "rop") in roles, f"actual gates: {roles}"


def test_anonymous_coverage_gets_401():
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(COVERAGE_ENDPOINT)
    assert r.status_code == 401, r.text


def test_student_coverage_gets_403(monkeypatch):
    async def _student(_request):
        return {"uid": "s-1", "u": "student@example.com", "r": "student", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _student)
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(COVERAGE_ENDPOINT)
    assert r.status_code == 403, r.text


def test_anonymous_history_gets_401():
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(HISTORY_ENDPOINT)
    assert r.status_code == 401, r.text


def test_student_history_gets_403(monkeypatch):
    async def _student(_request):
        return {"uid": "s-1", "u": "student@example.com", "r": "student", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _student)
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(HISTORY_ENDPOINT)
    assert r.status_code == 403, r.text
