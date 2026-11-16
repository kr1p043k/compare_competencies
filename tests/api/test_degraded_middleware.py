"""Общий флаг деградации сервисов (BACKLOG 2.1).

Юнит-проверка `degraded_state()` и контракт заголовков middleware:
на каждом ответе X-Service-Degraded = 0/1, при деградации есть X-Degraded-Reasons.
"""

from __future__ import annotations

from src.api_pkg import deps
from src import db
from src.monitoring.degraded import degraded_reasons, degraded_state


def test_degraded_when_nothing_ready(monkeypatch):
    monkeypatch.setattr(db, "pool", None)
    monkeypatch.setattr(deps, "evaluator", None)
    monkeypatch.setattr(deps, "recommendation_engine", None)
    state = degraded_state()
    assert state["degraded"] is True
    assert "db_pool_uninitialized" in state["reasons"]
    assert "evaluator_not_loaded" in state["reasons"]
    assert "recommendation_engine_not_loaded" in state["reasons"]
    assert state["count"] == len(state["reasons"]) == 3


def test_not_degraded_when_ready(monkeypatch):
    monkeypatch.setattr(db, "pool", object())
    monkeypatch.setattr(deps, "evaluator", object())
    monkeypatch.setattr(deps, "recommendation_engine", object())
    assert degraded_reasons() == []
    state = degraded_state()
    assert state["degraded"] is False
    assert state["count"] == 0


def test_middleware_sets_degraded_header(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api_pkg import create_app

    monkeypatch.setattr(db, "pool", None)
    monkeypatch.setattr(deps, "evaluator", None)
    monkeypatch.setattr(deps, "recommendation_engine", None)
    client = TestClient(create_app())
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.headers["X-Service-Degraded"] == "1"
    assert "db_pool_uninitialized" in resp.headers["X-Degraded-Reasons"]


def test_middleware_header_zero_when_ready(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api_pkg import create_app

    monkeypatch.setattr(db, "pool", object())
    monkeypatch.setattr(deps, "evaluator", object())
    monkeypatch.setattr(deps, "recommendation_engine", object())
    client = TestClient(create_app())
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.headers["X-Service-Degraded"] == "0"
    assert "X-Degraded-Reasons" not in resp.headers
