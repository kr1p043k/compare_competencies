"""Роутер чата обязан передавать роль в retrieval (иначе ACL режет корпус)."""

from __future__ import annotations

from types import SimpleNamespace

from src.api_pkg.routers import chat as chat_router
from src.services import llm_chat


class _FakeThread:
    id = "thread-1"


class _FakeMessage:
    id = "msg-1"


async def test_stream_passes_role_to_assess(monkeypatch):
    captured: dict = {}

    async def fake_thread(user_id, context_type, context_key, **kwargs):
        return _FakeThread()

    async def fake_history(user_id, context_type, context_key, **kwargs):
        return {"messages": []}

    async def fake_append(thread_id, role, blocks, model, **kwargs):
        return _FakeMessage()

    async def fake_assess(context_type, context_key, **kwargs):
        captured.update(kwargs)
        return {
            "text": "ответ",
            "blocks": [{"type": "text", "content": "ответ"}],
            "rag": {"hits": 1, "groundedness": 1.0},
        }

    monkeypatch.setattr(llm_chat, "get_or_create_thread", fake_thread)
    monkeypatch.setattr(llm_chat, "get_thread_history", fake_history)
    monkeypatch.setattr(llm_chat, "append_turn", fake_append)
    monkeypatch.setattr(llm_chat, "assess", fake_assess)

    events = [
        event
        async for event in chat_router._stream(
            "teacher", "09.03.02", {"uid": "u1", "r": "teacher"}, "почему УК-3 не покрыт"
        )
    ]
    assert captured["rag_role"] == "teacher"
    assert captured["message"] == "почему УК-3 не покрыт"
    done = [event for event in events if event.startswith("event: done")]
    assert done and '"rag": {"hits": 1' in done[0]


async def test_stream_without_role_still_works(monkeypatch):
    captured: dict = {}

    async def fake_thread(user_id, context_type, context_key, **kwargs):
        return _FakeThread()

    async def fake_history(user_id, context_type, context_key, **kwargs):
        return {"messages": []}

    async def fake_append(thread_id, role, blocks, model, **kwargs):
        return _FakeMessage()

    async def fake_assess(context_type, context_key, **kwargs):
        captured.update(kwargs)
        return {"text": "", "blocks": [], "rag": {}}

    monkeypatch.setattr(llm_chat, "get_or_create_thread", fake_thread)
    monkeypatch.setattr(llm_chat, "get_thread_history", fake_history)
    monkeypatch.setattr(llm_chat, "append_turn", fake_append)
    monkeypatch.setattr(llm_chat, "assess", fake_assess)

    [_ async for _ in chat_router._stream("student", "self_x", {"uid": "u2"}, "вопрос")]
    assert captured["rag_role"] == ""


def test_pipeline_stage_is_noop_when_rag_disabled(monkeypatch):
    from src.pipeline import rag_stage

    monkeypatch.setattr(rag_stage.config, "RAG_ENABLED", False)
    called = False

    async def boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("index must not run when RAG disabled")

    monkeypatch.setattr(rag_stage.RagIndexStage, "_index", boom, raising=False)
    result = rag_stage.RagIndexStage(SimpleNamespace()).run()
    assert result.is_ok()
    assert result.unwrap()["rag_enabled"] is False
    assert called is False


def test_pipeline_stage_survives_index_failure(monkeypatch):
    from src.pipeline import rag_stage

    monkeypatch.setattr(rag_stage.config, "RAG_ENABLED", True)

    async def boom(self, limit):
        raise RuntimeError("db down")

    monkeypatch.setattr(rag_stage.RagIndexStage, "_index", boom)
    result = rag_stage.RagIndexStage(SimpleNamespace()).run()
    assert result.is_ok()
    payload = result.unwrap()
    assert payload["rag_enabled"] is True
    assert "db down" in payload["error"]
