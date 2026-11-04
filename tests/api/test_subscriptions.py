"""Тесты подписок и уведомлений (src/api_pkg/routers/subscriptions.py).

Стиль как в test_publications.py / test_trends_by_profession.py: свежий FastAPI,
router монтируется через include_router, limiter подключается в app.state,
пользователь и фабрика сессий БД подменяются. Реальная БД и сеть не трогаются.
"""
from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api_pkg.routers.subscriptions as subs_mod
import src.database as database_mod

USER = {"uid": "u-1", "u": "u@example.com", "r": "teacher"}
_MISSING = object()


class FakeResult:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows if rows is not None else []
        self._scalar = scalar

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar


class FakeSession:
    """Минимальная async-сессия: execute/get/add/delete/commit/refresh."""

    def __init__(self, *, rows=None, scalar=0, get_obj=_MISSING):
        self.rows = rows if rows is not None else []
        self.scalar_value = scalar
        self.get_obj = get_obj
        self.added: list = []
        self.deleted: list = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, *_a, **_k):
        return FakeResult(self.rows, self.scalar_value)

    async def get(self, _model, _pk):
        return None if self.get_obj is _MISSING else self.get_obj

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.committed = True

    async def refresh(self, _obj):
        return None


class BrokenSession(FakeSession):
    async def execute(self, *_a, **_k):
        raise RuntimeError("db down: sensitive internal detail")


def _sub(**kw) -> SimpleNamespace:
    now = datetime.now(UTC)
    base = dict(
        id="sub-1",
        user_id="u-1",
        topic="AI",
        source="openalex+arxiv",
        telegram_chat_id="@me",
        email=None,
        is_active=True,
        last_checked_at=None,
        created_at=now,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _notif(**kw) -> SimpleNamespace:
    base = dict(
        id="n-1",
        subscription_id="sub-1",
        title="Новая статья",
        body="Текст",
        article_url="https://example.org/a",
        article_source="arxiv",
        severity=None,
        is_read=False,
        created_at=datetime.now(UTC),
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def reset_limiter():
    with contextlib.suppress(Exception):
        subs_mod.limiter._storage.reset()
    yield


def _make_client(monkeypatch, *, session=None, user=USER) -> TestClient:
    async def _fake_user(_request):
        return user

    monkeypatch.setattr(subs_mod, "get_current_user", _fake_user)
    if session is not None:
        monkeypatch.setattr(database_mod, "async_session_factory", lambda: session)
    app = FastAPI()
    app.state.limiter = subs_mod.limiter
    app.include_router(subs_mod.router)
    return TestClient(app, raise_server_exceptions=False)


# ------------------------------------------------------------------ list subs

def test_list_subscriptions_ok(monkeypatch):
    client = _make_client(monkeypatch, session=FakeSession(rows=[_sub()]))
    r = client.get("/subscriptions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    sub = body["subscriptions"][0]
    assert sub["id"] == "sub-1"
    assert sub["topic"] == "AI"
    assert sub["is_active"] is True
    assert sub["last_checked_at"] is None
    assert isinstance(sub["created_at"], str)


def test_list_subscriptions_requires_auth(monkeypatch):
    client = _make_client(monkeypatch, session=FakeSession(), user=None)
    r = client.get("/subscriptions")
    assert r.status_code == 401, r.text


# ---------------------------------------------------------------- create subs

def test_create_subscription_ok(monkeypatch):
    session = FakeSession()
    client = _make_client(monkeypatch, session=session)
    r = client.post("/subscriptions", json={"topic": "  ML  "})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"
    assert r.json()["id"]
    assert session.committed is True
    assert len(session.added) == 1
    created = session.added[0]
    assert created.topic == "ML"
    assert created.source == "openalex+arxiv"


def test_create_subscription_validation_error(monkeypatch):
    client = _make_client(monkeypatch, session=FakeSession())
    r = client.post("/subscriptions", json={})
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------- delete subs

def test_delete_own_subscription_ok(monkeypatch):
    session = FakeSession(get_obj=_sub())
    client = _make_client(monkeypatch, session=session)
    r = client.delete("/subscriptions/sub-1")
    assert r.status_code == 200, r.text
    assert session.deleted and session.committed


def test_delete_foreign_subscription_forbidden(monkeypatch):
    session = FakeSession(get_obj=_sub(user_id="other-user"))
    client = _make_client(monkeypatch, session=session)
    r = client.delete("/subscriptions/sub-1")
    assert r.status_code == 403, r.text
    assert session.deleted == []


def test_delete_missing_subscription_404(monkeypatch):
    client = _make_client(monkeypatch, session=FakeSession(get_obj=None))
    r = client.delete("/subscriptions/nope")
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------- notifications

def test_list_notifications_ok(monkeypatch):
    client = _make_client(monkeypatch, session=FakeSession(rows=[_notif()]))
    r = client.get("/notifications")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    assert body["notifications"][0]["severity"] == "info"
    assert body["notifications"][0]["is_read"] is False


def test_mark_read_foreign_notification_forbidden(monkeypatch):
    session = FakeSession(get_obj=_notif(user_id="other-user"))
    client = _make_client(monkeypatch, session=session)
    r = client.post("/notifications/n-1/read")
    assert r.status_code == 403, r.text


def test_unread_count_ok(monkeypatch):
    client = _make_client(monkeypatch, session=FakeSession(scalar=4))
    r = client.get("/notifications/unread-count")
    assert r.status_code == 200, r.text
    assert r.json() == {"unread_count": 4}


# ---------------------------------------------------------------- failures / limiter

def test_db_failure_returns_500_without_traceback(monkeypatch):
    client = _make_client(monkeypatch, session=BrokenSession())
    r = client.get("/subscriptions")
    assert r.status_code == 500
    assert "sensitive internal detail" not in r.text


def test_list_subscriptions_rate_limited_after_30(monkeypatch):
    client = _make_client(monkeypatch, session=FakeSession())
    for _ in range(30):
        assert client.get("/subscriptions").status_code == 200
    assert client.get("/subscriptions").status_code == 429
