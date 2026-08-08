"""System error notifications → «Уведомления» in-app feed (admins only).

Паттерн: in-memory буфер + фоновая задача сброса в БД (как в request_logger).
queue_system_error безопасно вызывать из sync (CLI) и async (API) контекстов.
"""

import asyncio
from collections import deque
from datetime import datetime

import structlog

from src.models.krm_models import Notification, User, _uuid

logger = structlog.get_logger(__name__)

MAX_BUFFER = 500
FLUSH_INTERVAL = 5  # seconds
FLUSH_BATCH = 50

VALID_SEVERITIES = ("info", "warning", "error")

_error_buffer: deque[dict] = deque(maxlen=MAX_BUFFER)


def _severity(value: str) -> str:
    return value if value in VALID_SEVERITIES else "info"


async def notify_admins(title: str, body: str, severity: str = "error", article_url: str | None = None) -> int:
    """Создать уведомление для всех администраторов (без привязки к подписке)."""
    sev = _severity(severity)
    from sqlalchemy import select

    from src.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(User.id).where(User.role == "admin", User.is_active == True))
        admin_ids = [str(row[0]) for row in result.all()]
        if not admin_ids:
            logger.warning("system_notify_no_admins", severity=sev)
            return 0
        for uid in admin_ids:
            session.add(Notification(
                id=_uuid(),
                subscription_id=None,
                user_id=uid,
                title=title[:500],
                body=body[:4000],
                article_url=article_url,
                article_source=None,
                severity=sev,
                created_at=datetime.utcnow(),
            ))
        await session.commit()
    logger.info("system_notify_sent", severity=sev, admins=len(admin_ids))
    return len(admin_ids)


def queue_system_error(title: str, body: str, severity: str = "error", article_url: str | None = None) -> None:
    """Добавить системную ошибку/предупреждение в очередь уведомлений.

    Безопасна из любого контекста: при запущенном loop планирует немедленный
    сброс, иначе сброс произойдёт периодическим флушером или вручную
    через flush_system_errors_sync().
    """
    _error_buffer.append({
        "title": title,
        "body": body,
        "severity": _severity(severity),
        "article_url": article_url,
    })
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        loop.create_task(_flush_to_db())
    else:
        logger.debug("system_error_queued_sync", title=title[:80])


async def _flush_to_db() -> None:
    if not _error_buffer:
        return
    entries = list(_error_buffer)
    _error_buffer.clear()
    try:
        for e in entries:
            await notify_admins(
                title=e["title"],
                body=e["body"],
                severity=e["severity"],
                article_url=e["article_url"],
            )
    except Exception as exc:
        logger.warning("system_notify_flush_failed", error=str(exc))
        # не теряем данные — кладём обратно с ограничением размера
        for e in reversed(entries):
            _error_buffer.appendleft(e)
        while len(_error_buffer) > MAX_BUFFER:
            _error_buffer.pop()


async def _periodic_flush() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL)
        await _flush_to_db()


def start_system_error_flusher() -> None:
    """Start the background system-error flusher (call from startup)."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(_periodic_flush())
    else:
        loop.create_task(_periodic_flush())


def flush_system_errors_sync() -> None:
    """Принудительно сбросить очередь (для CLI/синхронных вызовов)."""
    if not _error_buffer:
        return
    try:
        asyncio.get_running_loop()
        return  # loop уже запущен — сброс произойдёт через create_task
    except RuntimeError:
        pass
    asyncio.run(_flush_to_db())
