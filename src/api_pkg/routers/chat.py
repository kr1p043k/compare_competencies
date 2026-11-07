"""SSE chat endpoints (teacher/student) surfaced over ``src.services.llm_chat``.

Контракт:
  * ``GET /teacher/chat/stream?discipline=<code>&message=<text>`` -> SSE
      события: ``text`` (сырой кусок текста), ``block`` (JSON блока),
      ``done`` (``{"thread_id","message_id"}``), ``error`` (``{"message"}``).
  * ``GET /teacher/chat/history?discipline=<code>`` -> ``{"thread_id","messages"}``
  * ``GET /student/chat/stream?message=<text>`` -> SSE (те же события)
  * ``GET /student/chat/history`` -> ``{"thread_id","messages"}``

Роль-гейты: teacher-сторона - ``admin``/``teacher``/``rop``; student-сторона -
``student`` (плюс ``admin``; см. ``require_role``). Ключ контекста студента
выводится из аутентификации через ``self_profile_name`` и НИКОГДА из параметра
запроса - студент физически не может читать чужой тред.

Стрим никогда не отдаёт HTTP 500 посреди ответа: ошибки LLM уже поглощены
``llm_chat.assess`` (fallback-first), а ошибки персистенции превращаются в SSE
``error``-событие.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.api_pkg.routers.auth import require_any_role, require_role
from src.api_pkg.user_keys import self_profile_name
from src.services import llm_chat

logger = structlog.get_logger("api")

router = APIRouter(tags=["chat"])

_TEACHER_ROLES = ("admin", "teacher", "rop")
_CHUNK_SIZE = 1024
_MAX_MESSAGE_LEN = 4000
# Сколько последних ходов треда подставляем в промпт LLM-вызова.
_MAX_HISTORY_TURNS = 10


def _sse(event: str, data: str) -> str:
    """Один SSE-фрейм: ``event:`` + многострочный ``data:`` (текст может быть многострочным)."""
    body = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"event: {event}\n{body}\n\n"


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _chunks(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    text = text or ""
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


async def _stream(
    context_type: str,
    context_key: str,
    user: dict,
    message: str,
) -> AsyncGenerator[str, None]:
    """Сохранить ход и отдать оценку как SSE.

    Любой сбой персистенции становится ``error``-событием; исключение наружу не
    пробрасывается - клиент видит оформленный ответ, а не оборванное соединение.
    """
    user_id = user.get("uid") if isinstance(user, dict) else None

    try:
        thread = await llm_chat.get_or_create_thread(user_id, context_type, context_key)
        history = await llm_chat.get_thread_history(
            user_id, context_type, context_key, limit=_MAX_HISTORY_TURNS
        )
        history_messages = (history.get("messages") or [])[-_MAX_HISTORY_TURNS:]
        await llm_chat.append_turn(thread.id, "user", {"text": message}, None)
        result = await llm_chat.assess(
            context_type, context_key, message=message, history=history_messages
        )
    except Exception:
        logger.exception(
            "chat_stream_setup_failed",
            context_type=context_type,
            context_key=context_key,
        )
        yield _sse("error", _json({"message": "Не удалось обработать запрос. Попробуйте позже."}))
        return

    text = str(result.get("text") or "")
    blocks = result.get("blocks")
    if not isinstance(blocks, list):
        blocks = []
    model_used = result.get("blocks_model") or result.get("text_model")

    try:
        assistant = await llm_chat.append_turn(
            thread.id,
            "assistant",
            {"text": text, "blocks": blocks},
            model_used,
        )
    except Exception:
        logger.exception(
            "chat_stream_assistant_save_failed",
            context_type=context_type,
            context_key=context_key,
        )
        yield _sse("error", _json({"message": "Не удалось сохранить ответ. Попробуйте позже."}))
        return

    for chunk in _chunks(text):
        yield _sse("text", chunk)
    for block in blocks:
        yield _sse("block", _json(block))
    yield _sse("done", _json({"thread_id": thread.id, "message_id": assistant.id}))


@router.get("/teacher/chat/stream")
async def teacher_chat_stream(
    discipline: str = Query(..., min_length=1, max_length=255),
    message: str = Query(..., min_length=1, max_length=_MAX_MESSAGE_LEN),
    user: dict = Depends(require_any_role(*_TEACHER_ROLES)),
):
    """SSE-ответ на вопрос преподавателя по дисциплине."""
    return StreamingResponse(
        _stream("teacher", discipline, user, message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/teacher/chat/history")
async def teacher_chat_history(
    discipline: str = Query(..., min_length=1, max_length=255),
    user: dict = Depends(require_any_role(*_TEACHER_ROLES)),
):
    """История треда преподавателя по дисциплине."""
    try:
        return await llm_chat.get_thread_history(user.get("uid"), "teacher", discipline)
    except Exception as exc:
        logger.error(
            "chat_history_failed", context_type="teacher", context_key=discipline, error=str(exc)
        )
        raise HTTPException(status_code=503, detail="Не удалось загрузить историю чата") from exc


@router.get("/student/chat/stream")
async def student_chat_stream(
    message: str = Query(..., min_length=1, max_length=_MAX_MESSAGE_LEN),
    user: dict = Depends(require_role("student")),
):
    """SSE-ответ студенту. Контекст - исключительно свой профиль из auth."""
    context_key = self_profile_name(str(user.get("u") or ""))
    return StreamingResponse(
        _stream("student", context_key, user, message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/student/chat/history")
async def student_chat_history(
    user: dict = Depends(require_role("student")),
):
    """История собственного треда студента (профиль из auth)."""
    context_key = self_profile_name(str(user.get("u") or ""))
    try:
        return await llm_chat.get_thread_history(user.get("uid"), "student", context_key)
    except Exception as exc:
        logger.error(
            "chat_history_failed", context_type="student", context_key=context_key, error=str(exc)
        )
        raise HTTPException(status_code=503, detail="Не удалось загрузить историю чата") from exc
