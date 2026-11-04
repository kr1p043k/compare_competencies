"""Тесты роутера подписок и уведомлений.

Проверяем CRUD-эндпоинты и, главное, разграничение доступа (IDOR):
чужую подписку/уведомление обычный пользователь удалить/пометить не может,
admin может; несуществующий id -> 404; без аутентификации -> 401.

БД не трогаем: `async_session_factory` подменяется на фейковую сессию
(см. `FakeSession`), auth-зависимость `require_auth` - через
`dependency_overrides`. Приложение создаётся один раз на модуль
(`create_app` ~0.2 c), иначе тесты выходят за бюджет времени.
"""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.subscriptions as subs_mod
from src.api_pkg import create_app

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def app():
    return create_app()


# ─── фейковые результаты/сессия ────────────────────────────────────────────

class _Result:
    """Мини-замена Result: scalars().all(), scalar(), rowcount."""

    def __init__(self, rows=None, scalar=None, rowcount=0):
        self._rows = list(rows or [])
        self._scalar = scalar
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar


class FakeSession:
    """Эмулирует только то, что реально вызывает роутер.

    `execute` различает запросы по SQL: count -> scalar, notifications/
    subscriptions -> строки. `get` отдаёт объект из `store` по id.
    """

    def __init__(self, *, subs=(), notifs=(), unread=0):
        self.list_subs = list(subs)
        self.list_notifs = list(notifs)
        self.unread = unread
        # id уникальны между моделями в тестах - храним по строковому id.
        self.store: dict[str, object] = {}
        self.deleted: list[object] = []
        self.added: list[object] = []
        self.commits = 0

    def put(self, obj) -> None:
        self.store[str(obj.id)] = obj

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt, *args, **kwargs):
        sql = str(stmt).lower()
        if "count(" in sql:
            return _Result(scalar=self.unread)
        if "from notifications" in sql:
            return _Result(rows=self.list_notifs)
        return _Result(rows=self.list_subs)

    async def get(self, model, ident):
        return self.store.get(str(ident))

    async def delete(self, obj):
        self.deleted.append(obj)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1

    async def refresh(self, obj):
        return None


def _sub(sid: str, uid: str, **kw) -> SimpleNamespace:
    return SimpleNamespace(
        id=sid,
        user_id=uid,
        topic=kw.get("topic", "AI"),
        source=kw.get("source", "openalex+arxiv"),
        telegram_chat_id=kw.get("telegram_chat_id"),
        email=kw.get("email"),
        is_active=kw.get("is_active", True),
        last_checked_at=kw.get("last_checked_at"),
        created_at=kw.get("created_at", NOW),
    )


def _notif(nid: str, uid: str, **kw) -> SimpleNamespace:
    return SimpleNamespace(
        id=nid,
        user_id=uid,
        subscription_id=kw.get("subscription_id"),
        title=kw.get("title", "Новая статья"),
        body=kw.get("body", "текст"),
        article_url=kw.get("article_url"),
        article_source=kw.get("article_source"),
        severity=kw.get("severity", "info"),
        is_read=kw.get("is_read", False),
        created_at=kw.get("created_at", NOW),
    )


def _client(app, monkeypatch, session: FakeSession, user: dict | None):
    """TestClient с фейковой сессией; user=None -> неаутентифицированный запрос."""
    if user is None:
        async def _no_user(_request):
            return None

        monkeypatch.setattr(subs_mod, "get_current_user", _no_user)
    else:
        app.dependency_overrides[subs_mod.require_auth] = lambda: user

    # Роутер импортирует async_session_factory внутри функций из src.database,
    # поэтому патчим именно там, а не атрибут модуля subscriptions.
    monkeypatch.setattr("src.database.async_session_factory", lambda: session)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_overrides(app):
    yield
    app.dependency_overrides.clear()


# ─── DELETE /subscriptions/{sub_id} - IDOR ─────────────────────────────────

def test_delete_foreign_subscription_forbidden(app, monkeypatch):
    own = _sub("s-own", "u-1")
    foreign = _sub("s-foreign", "u-2")
    sess = FakeSession()
    sess.put(own)
    sess.put(foreign)
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.delete("/api/subscriptions/s-foreign")
    assert r.status_code == 403, r.text
    assert sess.deleted == []


def test_delete_own_subscription_ok(app, monkeypatch):
    own = _sub("s-own", "u-1")
    sess = FakeSession()
    sess.put(own)
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.delete("/api/subscriptions/s-own")
    assert r.status_code == 200, r.text
    assert own in sess.deleted


def test_delete_foreign_as_admin_ok(app, monkeypatch):
    foreign = _sub("s-foreign", "u-2")
    sess = FakeSession()
    sess.put(foreign)
    client = _client(app, monkeypatch, sess, {"uid": "admin-1", "r": "admin"})
    r = client.delete("/api/subscriptions/s-foreign")
    assert r.status_code == 200, r.text
    assert foreign in sess.deleted


def test_delete_missing_subscription_404(app, monkeypatch):
    sess = FakeSession()
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.delete("/api/subscriptions/does-not-exist")
    assert r.status_code == 404, r.text


# ─── POST /notifications/{notif_id}/read - IDOR ────────────────────────────

def test_mark_read_foreign_notification_forbidden(app, monkeypatch):
    own = _notif("n-own", "u-1")
    foreign = _notif("n-foreign", "u-2")
    sess = FakeSession()
    sess.put(own)
    sess.put(foreign)
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.post("/api/notifications/n-foreign/read")
    assert r.status_code == 403, r.text
    assert foreign.is_read is False


def test_mark_read_own_notification_ok(app, monkeypatch):
    own = _notif("n-own", "u-1")
    sess = FakeSession()
    sess.put(own)
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.post("/api/notifications/n-own/read")
    assert r.status_code == 200, r.text
    assert own.is_read is True
    assert sess.commits >= 1


def test_mark_read_foreign_as_admin_ok(app, monkeypatch):
    foreign = _notif("n-foreign", "u-2")
    sess = FakeSession()
    sess.put(foreign)
    client = _client(app, monkeypatch, sess, {"uid": "admin-1", "r": "admin"})
    r = client.post("/api/notifications/n-foreign/read")
    assert r.status_code == 200, r.text
    assert foreign.is_read is True


def test_mark_read_missing_notification_404(app, monkeypatch):
    sess = FakeSession()
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.post("/api/notifications/nope/read")
    assert r.status_code == 404, r.text


# ─── 401 без аутентификации ────────────────────────────────────────────────

@pytest.mark.parametrize("method,path", [
    ("get", "/api/subscriptions"),
    ("post", "/api/subscriptions"),
    ("delete", "/api/subscriptions/s-1"),
    ("get", "/api/notifications"),
    ("get", "/api/notifications/unread-count"),
    ("post", "/api/notifications/n-1/read"),
])
def test_unauthorized_returns_401(app, monkeypatch, method, path):
    sess = FakeSession()
    client = _client(app, monkeypatch, sess, user=None)
    r = client.request(method.upper(), path)
    assert r.status_code == 401, r.text


# ─── успешные ответы ───────────────────────────────────────────────────────

def test_list_subscriptions_shape(app, monkeypatch):
    s1 = _sub("s-1", "u-1", topic="LLM", email="a@b.c", last_checked_at=NOW)
    sess = FakeSession(subs=[s1])
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.get("/api/subscriptions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    item = body["subscriptions"][0]
    assert item["id"] == "s-1"
    assert item["topic"] == "LLM"
    assert item["email"] == "a@b.c"
    assert item["last_checked_at"] == NOW.isoformat()


def test_create_subscription_ok(app, monkeypatch):
    sess = FakeSession()
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.post("/api/subscriptions", json={"topic": "  AI  "})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["id"]
    assert sess.added and sess.added[0].user_id == "u-1"
    assert sess.added[0].topic == "AI"  # strip применён
    assert sess.commits >= 1


def test_list_notifications_and_unread_count(app, monkeypatch):
    n1 = _notif("n-1", "u-1", is_read=False)
    n2 = _notif("n-2", "u-1", is_read=True)
    sess = FakeSession(notifs=[n1, n2], unread=1)
    client = _client(app, monkeypatch, sess, {"uid": "u-1", "r": "teacher"})
    r = client.get("/api/notifications")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 2

    r2 = client.get("/api/notifications/unread-count")
    assert r2.status_code == 200, r2.text
    assert r2.json()["unread_count"] == 1
