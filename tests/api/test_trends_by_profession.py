"""Тесты эндпоинтов трендов по профессиям: общий пул БД + rate limit."""
from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
