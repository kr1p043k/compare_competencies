"""Endpoint tests for src/api_pkg/routers/chat.py (offline, no live LLM / DB).

Binds the real chat router with mocked auth gates and mocked ``llm_chat``
primitives, so no database and no live model are touched:

  (a) teacher stream 200 -> ``text`` + ``block`` + ``done`` SSE events
  (b) teacher stream with the wrong role -> 403
  (c) student stream derives its context key from auth (no query param)
  (d) LLM_ASSESS_ENABLED=False -> deterministic text block, LLM never called
  (e) history endpoints return thread messages (mocked DB primitive)
  (f) get_thread_history assembles messages from a fake session
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config
from src.api_pkg.routers import auth as auth_mod
from src.api_pkg.routers import chat as chat_mod
from src.api_pkg.user_keys import self_profile_name
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


# ── (a) teacher stream 200 ──────────────────────────────────────────────────

def test_teacher_stream_200(monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _make_user("teacher", uid="u-t"))
    monkeypatch.setattr(llm_chat, "get_or_create_thread", _fake_get_or_create_thread)
    monkeypatch.setattr(llm_chat, "append_turn", _fake_append_turn)
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

class _HistScalars:
    def __init__(self, thread) -> None:
        self.thread = thread

    def first(self):
        return self.thread


class _HistResult:
    def __init__(self, thread) -> None:
        self.thread = thread

    def scalars(self):
        return _HistScalars(self.thread)


class _HistSession:
    def __init__(self, thread) -> None:
        self.thread = thread

    async def execute(self, statement):
        return _HistResult(self.thread)

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def flush(self) -> None:
        pass


def _thread_stub():
    return _Obj(
        id="thread-9",
        messages=[
            SimpleNamespace(
                role="user",
                seq=1,
                blocks_json={"text": "hi"},
                created_at=datetime(2024, 1, 1, 12, 0, 0),
            ),
            SimpleNamespace(
                role="assistant",
                seq=2,
                blocks_json={"text": "hello", "blocks": []},
                created_at=datetime(2024, 1, 1, 12, 0, 1),
            ),
        ],
    )


async def test_get_thread_history_assembles_messages():
    session = _HistSession(_thread_stub())

    out = await llm_chat.get_thread_history("u", "teacher", "09.03.02", session=session)

    assert out["thread_id"] == "thread-9"
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert out["messages"][0]["blocks"] == {"text": "hi"}
    assert out["messages"][0]["ts"] == "2024-01-01T12:00:00"


async def test_get_thread_history_empty_thread():
    session = _HistSession(None)

    out = await llm_chat.get_thread_history("u", "teacher", "x", session=session)

    assert out == {"thread_id": None, "messages": []}
