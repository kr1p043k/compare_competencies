"""System error notifications → «Уведомления» in-app feed (admins only).

Паттерн: in-memory буфер + фоновая задача сброса в БД (как в request_logger).
queue_system_error безопасно вызывать из sync (CLI) и async (API) контекстов.
"""

import asyncio
import time
from collections import deque
from datetime import datetime

import structlog

from src.models.krm_models import Notification, User, _uuid

logger = structlog.get_logger(__name__)

MAX_BUFFER = 500
FLUSH_INTERVAL = 5  # seconds
FLUSH_BATCH = 50

VALID_SEVERITIES = ("info", "warning", "error")

API_SIGNATURE_PREFIX = "api:"
API_RESOLVE_WINDOW = 30.0  # seconds — троттлинг повторного резолва одного пути

_error_buffer: deque[dict] = deque(maxlen=MAX_BUFFER)
_resolve_buffer: deque[dict] = deque(maxlen=MAX_BUFFER)
_api_resolve_recent: dict[str, float] = {}


def _api_signature(method: str, path: str) -> str:
    return f"{API_SIGNATURE_PREFIX}{method} {path}"


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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


def _schedule_flush() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        loop.create_task(_flush_to_db())
    else:
        logger.debug("system_notify_queued_sync")


def queue_system_error(title: str, body: str, severity: str = "error", article_url: str | None = None, dedupe: bool = False) -> None:
    """Добавить системную ошибку/предупреждение в очередь уведомлений.

    Безопасна из любого контекста: при запущенном loop планирует немедленный
    сброс, иначе сброс произойдёт периодическим флушером или вручную
    через flush_system_errors_sync().

    dedupe=True: если в ленте уже есть непрочитанное уведомление с таким же
    (title, article_url), повторная запись не создаётся — ошибки не заваливают ленту.
    """
    _error_buffer.append({
        "title": title,
        "body": body,
        "severity": _severity(severity),
        "article_url": article_url,
        "dedupe": dedupe,
    })
    _schedule_flush()


def queue_pipeline_stage_resolved(stage_name: str) -> None:
    """Этап пайплайна успешно выполнился — снять «Ошибка этапа пайплайна: {stage_name}»."""
    _resolve_buffer.append({"type": "stage", "stage_name": stage_name})
    _schedule_flush()


def queue_pipeline_run_resolved() -> None:
    """Пайплайн успешно завершился — снять «Пайплайн не завершился»."""
    _resolve_buffer.append({"type": "pipeline_failed"})
    _schedule_flush()


async def resolve_warmup_failure(component: str) -> int:
    """Удалить системные предупреждения о неудачной загрузке компонента.

    Вызывается при успешной повторной инициализации компонента — если warning
    больше не актуален, он исчезает из ленты уведомлений.
    """
    from sqlalchemy import delete

    from src.database import async_session_factory

    title = f"Не удалось загрузить компонент: {component}"
    async with async_session_factory() as session:
        result = await session.execute(
            delete(Notification).where(
                Notification.title == title,
                Notification.subscription_id.is_(None),
            )
        )
        count = result.rowcount or 0
        await session.commit()
    if count:
        logger.info("warmup_failure_resolved", component=component, cleared=count)
    return count


async def resolve_pipeline_stage_errors(stage_name: str) -> int:
    """Удалить «Ошибка этапа пайплайна: {stage_name}» — этап снова отработал успешно."""
    from sqlalchemy import delete

    from src.database import async_session_factory

    title = f"Ошибка этапа пайплайна: {stage_name}"
    async with async_session_factory() as session:
        result = await session.execute(
            delete(Notification).where(
                Notification.title == title,
                Notification.subscription_id.is_(None),
            )
        )
        count = result.rowcount or 0
        await session.commit()
    if count:
        logger.info("pipeline_stage_resolved", stage=stage_name, cleared=count)
    return count


async def resolve_pipeline_failed() -> int:
    """Удалить «Пайплайн не завершился» — последний прогон прошёл успешно."""
    from sqlalchemy import delete

    from src.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(
            delete(Notification).where(
                Notification.title == "Пайплайн не завершился",
                Notification.subscription_id.is_(None),
            )
        )
        count = result.rowcount or 0
        await session.commit()
    if count:
        logger.info("pipeline_failed_resolved", cleared=count)
    return count


async def _has_unread_duplicate(title: str, article_url: str) -> bool:
    """Есть ли уже непрочитанное системное уведомление с такой сигнатурой."""
    from sqlalchemy import select

    from src.database import async_session_factory

    async with async_session_factory() as session:
        row = await session.execute(
            select(Notification.id)
            .where(
                Notification.title == title,
                Notification.article_url == article_url,
                Notification.subscription_id.is_(None),
                Notification.is_read == False,
            )
            .limit(1)
        )
        return row.scalar() is not None


async def resolve_api_errors(method: str, path: str) -> int:
    """Удалить системные уведомления об API-ошибках на этом пути.

    Вызывается при успешном ответе (код < 500) — «ошибка починилась»,
    и уведомление о ней больше не актуально, оно исчезает из ленты.
    Снимает и legacy-уведомления (article_url пуст):
      - «Ошибка API: {METHOD} {path}» — по title;
      - «Необработанная ошибка API: ...» — по префиксу body "{METHOD} {path} |".
    """
    from sqlalchemy import and_, delete, or_

    from src.database import async_session_factory

    sig = _api_signature(method, path)
    async with async_session_factory() as session:
        result = await session.execute(
            delete(Notification).where(
                Notification.subscription_id.is_(None),
                or_(
                    Notification.article_url == sig,
                    Notification.title == f"Ошибка API: {method} {path}",
                    and_(
                        Notification.title.like("Необработанная ошибка API:%"),
                        Notification.body.like(
                            f"{_like_escape(method)} {_like_escape(path)} |%", escape="\\"
                        ),
                    ),
                ),
            )
        )
        await session.commit()
        count = result.rowcount or 0
    if count:
        logger.info("api_error_resolved", method=method, path=path, cleared=count)
    return count


async def maybe_resolve_api_errors(method: str, path: str) -> int:
    """Обёртка с троттлингом, чтобы не делать UPDATE в БД на каждый запрос."""
    sig = f"{method} {path}"
    now = time.monotonic()
    last = _api_resolve_recent.get(sig)
    if last is not None and now - last < API_RESOLVE_WINDOW:
        return 0
    count = await resolve_api_errors(method, path)
    _api_resolve_recent[sig] = now
    if len(_api_resolve_recent) > 200:
        for key in [k for k, v in _api_resolve_recent.items() if now - v > API_RESOLVE_WINDOW]:
            _api_resolve_recent.pop(key, None)
    return count


async def _flush_to_db() -> None:
    if not _error_buffer and not _resolve_buffer:
        return
    entries = list(_error_buffer)
    _error_buffer.clear()
    resolves = list(_resolve_buffer)
    _resolve_buffer.clear()
    try:
        for e in entries:
            if e.get("dedupe") and e.get("article_url"):
                if await _has_unread_duplicate(e["title"], e["article_url"]):
                    continue
            await notify_admins(
                title=e["title"],
                body=e["body"],
                severity=e["severity"],
                article_url=e["article_url"],
            )
        for r in resolves:
            if r.get("type") == "stage":
                await resolve_pipeline_stage_errors(r["stage_name"])
            elif r.get("type") == "pipeline_failed":
                await resolve_pipeline_failed()
    except Exception as exc:
        logger.warning("system_notify_flush_failed", error=str(exc))
        # не теряем данные — кладём обратно с ограничением размера
        for e in reversed(entries):
            _error_buffer.appendleft(e)
        for r in reversed(resolves):
            _resolve_buffer.appendleft(r)
        while len(_error_buffer) > MAX_BUFFER:
            _error_buffer.pop()
        while len(_resolve_buffer) > MAX_BUFFER:
            _resolve_buffer.pop()


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
