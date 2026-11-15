"""Endpoint tests for src/api_pkg/routers/chat.py (offline, no live LLM / DB).

Binds the real chat router with mocked auth gates and mocked ``llm_chat``
primitives, so no database and no live model are touched:

  (a) teacher stream 200 -> ``text`` + ``block`` + ``done`` SSE events
  (b) teacher stream with the wrong role -> 403
  (c) student stream derives its context key from auth (no query param)
  (d) LLM_ASSESS_ENABLED=False -> deterministic text block, LLM never called
  (e) history endpoints return thread messages (mocked DB primitive)
  (f) get_thread_history assembles messages from a fake session
  (g) rate limiting + auth gates for chat.py and llm.py (backlog #3.4):
      burst past the slowapi limit -> 429, unauthenticated -> 401
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config
from src.api_pkg.routers import auth as auth_mod
from src.api_pkg.routers import chat as chat_mod
from src.api_pkg.user_keys import self_profile_name
from src.models.krm_models import ChatMessage, ChatThread
from src.services import llm_chat

# ── helpers ─────────────────────────────────────────────────────────────────

class _Obj:
    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(chat_mod.router, prefix="/api")
    return app


def _make_user(role: str, *, uid: str = "u-1", username: str = "test@example.com"):
    async def fake(request):
        return {"uid": uid, "u": username, "r": role}

    return fake


async def _fake_get_or_create_thread(user_id, context_type, context_key, **kw):
    return _Obj(id="thread-1", context_type=context_type, context_key=context_key)


async def _fake_append_turn(thread_id, role, blocks_json, model_used, **kw):
    return _Obj(id=f"msg-{role}")


async def _fake_assess(context_type, context_key, **kw):
    return {
        "text": "Привет, вот сводка.",
        "blocks": [{"type": "text", "content": "блок сводки"}],
        "text_model": "text-model",
        "blocks_model": "blocks-model",
        "llm_used": False,
    }


async def _fake_get_thread_history(user_id, context_type, context_key, **kw):
    return {"thread_id": "thread-1", "messages": []}


# ── (a) teacher stream 200 ──────────────────────────────────────────────────

def test_teacher_stream_200(monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher", uid="u-t"))
    monkeypatch.setattr(llm_chat, "get_or_create_thread", _fake_get_or_create_thread)
    monkeypatch.setattr(llm_chat, "append_turn", _fake_append_turn)
    monkeypatch.setattr(llm_chat, "get_thread_history", _fake_get_thread_history)
    monkeypatch.setattr(llm_chat, "assess", _fake_assess)

    client = TestClient(_make_app())
    resp = client.get(
        "/api/teacher/chat/stream",
        params={"discipline": "09.03.02", "message": "Какие пробелы?"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: text" in body
    assert "event: block" in body
    assert '"type": "text"' in body
    assert "event: done" in body
    assert "thread-1" in body
    assert "msg-assistant" in body


def test_teacher_stream_passes_question_to_assess(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher", uid="u-t"))
    monkeypatch.setattr(llm_chat, "get_or_create_thread", _fake_get_or_create_thread)
    monkeypatch.setattr(llm_chat, "append_turn", _fake_append_turn)
    monkeypatch.setattr(llm_chat, "get_thread_history", _fake_get_thread_history)

    async def capture_assess(context_type, context_key, **kw):
        calls["context_type"] = context_type
        calls["message"] = kw.get("message")
        calls["history"] = kw.get("history")
        return {
            "text": "ок",
            "blocks": [{"type": "text", "content": "x"}],
            "text_model": "t",
            "blocks_model": "b",
            "llm_used": False,
        }

    monkeypatch.setattr(llm_chat, "assess", capture_assess)

    client = TestClient(_make_app())
    resp = client.get(
        "/api/teacher/chat/stream",
        params={"discipline": "09.03.02", "message": "Какие пробелы?"},
    )

    assert resp.status_code == 200
    assert calls["message"] == "Какие пробелы?"
    assert calls["history"] == []


# ── (b) teacher stream wrong role -> 403 ────────────────────────────────────

def test_teacher_stream_forbidden_for_student(monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("student"))

    client = TestClient(_make_app())
    resp = client.get(
        "/api/teacher/chat/stream",
        params={"discipline": "09.03.02", "message": "hi"},
    )

    assert resp.status_code == 403


def test_student_stream_forbidden_for_teacher(monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher"))

    client = TestClient(_make_app())
    resp = client.get("/api/student/chat/stream", params={"message": "hi"})

    assert resp.status_code == 403


# ── (c) student stream context from auth ────────────────────────────────────

def test_student_stream_context_from_auth(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        auth_mod, "get_current_user", _make_user("student", uid="u-s", username="Student@Example.com")
    )
    monkeypatch.setattr(llm_chat, "get_or_create_thread", _fake_get_or_create_thread)
    monkeypatch.setattr(llm_chat, "append_turn", _fake_append_turn)
    monkeypatch.setattr(llm_chat, "get_thread_history", _fake_get_thread_history)

    async def capture_assess(context_type, context_key, **kw):
        calls["context_type"] = context_type
        calls["context_key"] = context_key
        return {
            "text": "ок",
            "blocks": [{"type": "text", "content": "x"}],
            "text_model": "t",
            "blocks_model": "b",
            "llm_used": False,
        }

    monkeypatch.setattr(llm_chat, "assess", capture_assess)

    client = TestClient(_make_app())
    resp = client.get("/api/student/chat/stream", params={"message": "hey"})

    assert resp.status_code == 200
    assert calls["context_type"] == "student"
    # Профиль выводится ИЗ auth (self_profile_name), не из query-параметра.
    assert calls["context_key"] == self_profile_name("Student@Example.com")


# ── (d) fallback: LLM_ASSESS_ENABLED=False, модель не вызывается ────────────

def test_teacher_stream_fallback_never_calls_llm(monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher"))
    monkeypatch.setattr(config, "LLM_ASSESS_ENABLED", False)
    monkeypatch.setattr(llm_chat, "get_or_create_thread", _fake_get_or_create_thread)
    monkeypatch.setattr(llm_chat, "append_turn", _fake_append_turn)
    monkeypatch.setattr(llm_chat, "get_thread_history", _fake_get_thread_history)

    async def fake_build(*args, **kwargs):
        return {
            "context_type": "teacher",
            "context_key": "09.03.02",
            "title": "Информатика",
            "metrics": {"coverage": 80.0, "quality": 70.0},
            "gaps": [{"skill": "Kafka"}],
        }

    monkeypatch.setattr(llm_chat, "build_context", fake_build)

    called: list[str] = []

    def _boom(*args, **kwargs):
        called.append("llm")
        raise AssertionError("LLM must not be called with LLM_ASSESS_ENABLED=False")

    monkeypatch.setattr(llm_chat, "answer_text", _boom)
    monkeypatch.setattr(llm_chat, "answer_blocks", _boom)

    client = TestClient(_make_app())
    resp = client.get(
        "/api/teacher/chat/stream",
        params={"discipline": "09.03.02", "message": "hi"},
    )

    assert resp.status_code == 200
    assert "event: text" in resp.text
    assert "event: done" in resp.text
    # Детерминированный текст содержит имя пробела.
    assert "Kafka" in resp.text
    assert called == []


# ── (e) history endpoints (mocked DB primitive) ─────────────────────────────

def test_teacher_history(monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher", uid="u-t"))

    async def fake_history(user_id, context_type, context_key, **kw):
        return {
            "thread_id": "t1",
            "messages": [
                {"role": "user", "blocks": {"text": "hi"}, "ts": "2024-01-01T00:00:00Z"},
            ],
        }

    monkeypatch.setattr(llm_chat, "get_thread_history", fake_history)

    client = TestClient(_make_app())
    resp = client.get("/api/teacher/chat/history", params={"discipline": "09.03.02"})

    assert resp.status_code == 200
    assert resp.json()["thread_id"] == "t1"
    assert resp.json()["messages"][0]["role"] == "user"
    assert resp.json()["messages"][0]["blocks"] == {"text": "hi"}


def test_student_history(monkeypatch):
    monkeypatch.setattr(
        auth_mod, "get_current_user", _make_user("student", uid="u-s", username="a@b.c")
    )

    captured: dict = {}

    async def fake_history(user_id, context_type, context_key, **kw):
        captured["user_id"] = user_id
        captured["context_type"] = context_type
        captured["context_key"] = context_key
        return {"thread_id": None, "messages": []}

    monkeypatch.setattr(llm_chat, "get_thread_history", fake_history)

    client = TestClient(_make_app())
    resp = client.get("/api/student/chat/history")

    assert resp.status_code == 200
    assert resp.json() == {"thread_id": None, "messages": []}
    assert captured["user_id"] == "u-s"
    assert captured["context_type"] == "student"
    assert captured["context_key"] == self_profile_name("a@b.c")


# ── (f) get_thread_history service (fake session) ───────────────────────────

class _HistThreadIdScalars:
    def __init__(self, thread_id) -> None:
        self.thread_id = thread_id

    def first(self):
        return self.thread_id


class _HistThreadIdResult:
    def __init__(self, thread_id) -> None:
        self.thread_id = thread_id

    def scalars(self):
        return _HistThreadIdScalars(self.thread_id)


class _HistMessagesScalars:
    def __init__(self, messages) -> None:
        self.messages = messages

    def all(self):
        return self.messages


class _HistMessagesResult:
    def __init__(self, messages) -> None:
        self.messages = messages

    def scalars(self):
        return _HistMessagesScalars(self.messages)


class _HistSession:
    """Фейк для get_thread_history.

    Тип запроса определяется по entity из ``column_descriptions``:
    ``select(ChatThread.id)`` отдаёт thread_id, ``select(ChatMessage)`` - сообщения.
    Значение LIMIT берётся из ``statement._limit_clause.value`` (без ``literal_binds``,
    чтобы не рендерить UUID-строку 'thread-9' как UUID-литерал). Порядок/лимит
    проверяются по ``str(statement)`` (``ORDER BY ... DESC`` + ``LIMIT``).
    """

    def __init__(self, thread_id, messages) -> None:
        self.thread_id = thread_id
        self.messages = messages
        self.calls = 0
        self.messages_sql = None

    async def execute(self, statement):
        self.calls += 1
        entity = statement.column_descriptions[0]["entity"]
        if entity is ChatThread:
            return _HistThreadIdResult(self.thread_id)
        if entity is not ChatMessage:
            raise AssertionError(f"unexpected query entity: {entity!r}")
        self.messages_sql = str(statement.compile())
        limit_clause = getattr(statement, "_limit_clause", None)
        limit = limit_clause.value if limit_clause is not None else None
        desc = sorted(self.messages, key=lambda m: (m.seq or 0), reverse=True)
        if limit is not None:
            desc = desc[:limit]
        return _HistMessagesResult(desc)

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def flush(self) -> None:
        pass


def _msg_stub(role, seq, text):
    return SimpleNamespace(
        role=role,
        seq=seq,
        blocks_json={"text": text},
        created_at=datetime(2024, 1, 1, 12, 0, seq, tzinfo=UTC),
    )


async def test_get_thread_history_assembles_messages():
    session = _HistSession(
        "thread-9",
        [_msg_stub("user", 1, "hi"), _msg_stub("assistant", 2, "hello")],
    )

    out = await llm_chat.get_thread_history("u", "teacher", "09.03.02", session=session)

    assert out["thread_id"] == "thread-9"
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert out["messages"][0]["blocks"] == {"text": "hi"}
    assert out["messages"][0]["ts"] == "2024-01-01T12:00:01+00:00"
    # Второй запрос обязан быть ORDER BY seq DESC + LIMIT.
    assert session.messages_sql is not None
    assert "ORDER BY" in session.messages_sql
    assert "DESC" in session.messages_sql
    assert "LIMIT" in session.messages_sql


async def test_get_thread_history_empty_thread():
    session = _HistSession(None, [])

    out = await llm_chat.get_thread_history("u", "teacher", "x", session=session)

    assert out == {"thread_id": None, "messages": []}


async def test_get_thread_history_limit_cuts_old_and_orders_by_seq():
    session = _HistSession(
        "thread-9",
        [
            _msg_stub("user", 1, "m1"),
            _msg_stub("assistant", 2, "m2"),
            _msg_stub("user", 3, "m3"),
            _msg_stub("assistant", 4, "m4"),
            _msg_stub("user", 5, "m5"),
        ],
    )

    out = await llm_chat.get_thread_history(
        "u", "teacher", "09.03.02", limit=3, session=session
    )

    assert out["thread_id"] == "thread-9"
    assert [m["blocks"]["text"] for m in out["messages"]] == ["m3", "m4", "m5"]


# ── (g) rate limit + auth на LLM-эндпоинтах (backlog #3.4) ───────────────────
#
# Медленный и дорогой ресурс (LLM) обязан быть под slowapi и под гейтом:
# burst выше лимита -> 429, анонимный вызов -> 401. Чистое приложение без
# lifespan и без БД (паттерн tests/api/test_subscriptions.py): лимитер
# общий (src.api_pkg.rate_limit), ключ без middleware аудита - ip:testclient.

from unittest.mock import MagicMock  # noqa: E402

from src.api_pkg.routers import llm as llm_mod  # noqa: E402


def _reset_limiter():
    with contextlib.suppress(Exception):
        chat_mod.limiter._storage.reset()


def _make_chat_app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = chat_mod.limiter
    app.include_router(chat_mod.router, prefix="/api")
    return app


def _make_llm_app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = llm_mod.limiter
    app.include_router(llm_mod.router, prefix="/api")
    return app


def test_teacher_stream_burst_gets_429(monkeypatch):
    """11-й запрос stream за минуту -> 429 (лимит 10/minute)."""
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher", uid="u-t"))
    monkeypatch.setattr(llm_chat, "get_or_create_thread", _fake_get_or_create_thread)
    monkeypatch.setattr(llm_chat, "append_turn", _fake_append_turn)
    monkeypatch.setattr(llm_chat, "get_thread_history", _fake_get_thread_history)
    monkeypatch.setattr(llm_chat, "assess", _fake_assess)
    _reset_limiter()

    client = TestClient(_make_chat_app(), raise_server_exceptions=False)
    for _ in range(10):
        resp = client.get(
            "/api/teacher/chat/stream",
            params={"discipline": "09.03.02", "message": "hi"},
        )
        assert resp.status_code == 200, resp.text
    blocked = client.get(
        "/api/teacher/chat/stream",
        params={"discipline": "09.03.02", "message": "hi"},
    )
    assert blocked.status_code == 429, blocked.text


def test_student_history_burst_gets_429(monkeypatch):
    """31-й запрос истории за минуту -> 429 (лимит 30/minute)."""
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("student", uid="u-s"))
    monkeypatch.setattr(llm_chat, "get_thread_history", _fake_get_thread_history)
    _reset_limiter()

    client = TestClient(_make_chat_app(), raise_server_exceptions=False)
    for _ in range(30):
        assert client.get("/api/student/chat/history").status_code == 200
    assert client.get("/api/student/chat/history").status_code == 429


def test_chat_stream_requires_auth():
    """Без токена get_current_user -> None -> гейт отдаёт 401 (без обращения к БД:
    пустой заголовок не доходит до resolve_session)."""
    _reset_limiter()
    client = TestClient(_make_chat_app(), raise_server_exceptions=False)
    resp = client.get(
        "/api/teacher/chat/stream",
        params={"discipline": "09.03.02", "message": "hi"},
    )
    assert resp.status_code == 401, resp.text


def test_llm_chat_post_burst_gets_429(monkeypatch):
    """11-й POST /llm/chat за минуту -> 429 (лимит 10/minute уже стоит, llm.py:31)."""
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher", uid="u-t"))
    fake_client = MagicMock()
    fake_client.generate.return_value = "ok"
    fake_client.model = "stub-model"
    monkeypatch.setattr(llm_mod, "LLMClient", MagicMock(return_value=fake_client))
    _reset_limiter()

    client = TestClient(_make_llm_app(), raise_server_exceptions=False)
    for _ in range(10):
        resp = client.post("/api/llm/chat", json={"message": "hi"})
        assert resp.status_code == 200, resp.text
    blocked = client.post("/api/llm/chat", json={"message": "hi"})
    assert blocked.status_code == 429, blocked.text


def test_llm_chat_post_requires_auth():
    """Анонимный POST /llm/chat -> 401 через dependencies=[Depends(require_any_role)]
    (llm.py:29), до входа в тело хэндлера (лимитер не тратится)."""
    _reset_limiter()
    client = TestClient(_make_llm_app(), raise_server_exceptions=False)
    resp = client.post("/api/llm/chat", json={"message": "hi"})
    assert resp.status_code == 401, resp.text
