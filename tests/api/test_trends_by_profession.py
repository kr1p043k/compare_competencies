"""Тесты эндпоинтов трендов по профессиям: общий пул БД + rate limit.

Added later (additive, existing tests untouched): the snapshot branch of
/trends/by-profession (newest file must win), live query with an empty pool,
and the role guard - the router-level require_authenticated gate: no token
-> 401, while any logged-in role (student included) is admitted; only
get_current_user is stubbed, the gate itself stays real (convention of
tests/api/test_coverage_teacher_krm.py).
"""
from __future__ import annotations

import contextlib
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config
from src.api_pkg.routers import trends_by_profession as trends_prof


@pytest.fixture(autouse=True)
def reset_limiter():
    with contextlib.suppress(Exception):
        trends_prof.limiter._storage.reset()
    yield


async def _fake_user(request):
    return {"u": "test", "r": "admin"}


@pytest.fixture
def client(monkeypatch):
    """Fresh app with the router mounted and auth bypassed (logic-only tests)."""

    def factory() -> TestClient:
        import src.api_pkg.routers.auth as auth_mod

        monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
        app = FastAPI()
        app.state.limiter = trends_prof.limiter
        app.include_router(trends_prof.router)
        return TestClient(app, raise_server_exceptions=False)

    return factory


def test_list_professions_returns_200(client):
    resp = client().get("/trends/professions")
    assert resp.status_code == 200
    assert "professions" in resp.json()


def test_by_profession_live_uses_pool(client, monkeypatch):
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=[{"skill": "python", "cnt": 5}])
    monkeypatch.setattr(trends_prof, "get_pool", lambda: pool)

    resp = client().get("/trends/by-profession", params={"profession": "backend"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "live"
    assert body["skills"] == [{"skill": "python", "frequency": 5}]
    pool.fetch.assert_awaited_once()


def test_by_profession_pool_error_returns_503(client, monkeypatch):
    pool = AsyncMock()
    pool.fetch = AsyncMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(trends_prof, "get_pool", lambda: pool)

    resp = client().get("/trends/by-profession", params={"profession": "backend"})

    assert resp.status_code == 503


def test_by_profession_rate_limited_after_60(client):
    c = client()
    for _ in range(60):
        assert c.get("/trends/professions").status_code == 200
    assert c.get("/trends/professions").status_code == 429


# ----------------------------------------------------- snapshot branch + role guard

def _auth_client(monkeypatch, role=None, pool_rows=None):
    """Bare app with the REAL router-level gate; only get_current_user is
    stubbed (role=None -> anonymous, the real require_authenticated answers
    401 itself). Same convention as tests/api/test_coverage_teacher_krm._client.
    """
    import src.api_pkg.routers.auth as auth_mod

    user = None if role is None else {"u": f"{role}@example.com", "r": role, "t": None}

    async def _fake_user(_request):
        return user

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    if pool_rows is not None:
        pool = AsyncMock()
        pool.fetch = AsyncMock(return_value=pool_rows)
        monkeypatch.setattr(trends_prof, "get_pool", lambda: pool)
    app = FastAPI()
    app.state.limiter = trends_prof.limiter
    app.include_router(trends_prof.router)
    return TestClient(app, raise_server_exceptions=False)


def test_by_profession_reads_latest_snapshot(monkeypatch, tmp_path):
    """Snapshot branch: two months on disk, the newest file must win."""
    (tmp_path / "freq_profession_backend_2026-09.json").write_text(
        json.dumps({"_meta": {"snapshot_date": "2026-09-01"}, "cobol": 99}),
        encoding="utf-8",
    )
    (tmp_path / "freq_profession_backend_2026-10.json").write_text(
        json.dumps({
            "_meta": {
                "snapshot_date": "2026-10-01",
                "vacancy_count": 10,
                "methodology_version": "v2",
            },
            "python": 9,
            "sql": 12,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "HISTORY_DIR", tmp_path)

    resp = _auth_client(monkeypatch, role="teacher").get(
        "/trends/by-profession", params={"profession": "backend"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "snapshot"
    assert body["snapshot_date"] == "2026-10-01", "must pick the newest snapshot file"
    assert body["vacancy_count"] == 10
    assert body["synthetic"] is False
    assert body["methodology_version"] == "v2"
    # Skills are sorted by frequency descending.
    assert body["skills"] == [
        {"skill": "sql", "frequency": 12},
        {"skill": "python", "frequency": 9},
    ]


def test_by_profession_pool_without_rows_returns_empty_skills(monkeypatch):
    resp = _auth_client(monkeypatch, role="teacher", pool_rows=[]).get(
        "/trends/by-profession", params={"profession": "backend"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "live"
    assert body["skills"] == []


def test_by_profession_drops_skills_with_zero_frequency(monkeypatch):
    rows = [{"skill": "python", "cnt": 5}, {"skill": "dead", "cnt": 0}]
    resp = _auth_client(monkeypatch, role="teacher", pool_rows=rows).get(
        "/trends/by-profession", params={"profession": "backend"}
    )
    assert resp.status_code == 200
    assert resp.json()["skills"] == [{"skill": "python", "frequency": 5}]


def test_anonymous_request_gets_401(monkeypatch):
    """The trends gate is require_authenticated: no token -> 401."""
    resp = _auth_client(monkeypatch, role=None).get(
        "/trends/by-profession", params={"profession": "backend"}
    )
    assert resp.status_code == 401, resp.text


def test_authenticated_student_is_allowed(monkeypatch):
    """Unlike /teacher/* routes, this gate admits any logged-in role:
    a student gets 200 (documents the require_authenticated semantics)."""
    rows = [{"skill": "sql", "cnt": 3}]
    resp = _auth_client(monkeypatch, role="student", pool_rows=rows).get(
        "/trends/by-profession", params={"profession": "backend"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["skills"] == [{"skill": "sql", "frequency": 3}]
