"""Audit: durable-журнал событий (таблица audit_logs) с фильтрами и пагинацией.

Доступ только admin: строки содержат email исполнителей, IP и request_id.
Читаем через SQLAlchemy-модель AuditLog (портабельно), а не сырой SQL.
"""

from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from src.api_pkg.routers.auth import require_any_role
from src.database import async_session_factory
from src.models.krm_models import AuditLog

logger = structlog.get_logger("api")

router = APIRouter(tags=["audit"], dependencies=[Depends(require_any_role("admin"))])
limiter = Limiter(key_func=get_remote_address)


def _parse_iso(value: str, label: str) -> datetime:
    """ISO 8601 -> aware datetime (naive трактуем как UTC); иначе 400."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: expected ISO 8601 datetime") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _is_undefined_table(exc: BaseException) -> bool:
    """True, если ошибка - 'relation does not exist' (asyncpg UndefinedTableError).

    SQLAlchemy оборачивает asyncpg-исключение в ProgrammingError; исходная
    ошибка лежит в .orig / .__cause__, у неё sqlstate == '42P01'.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if type(node).__name__ == "UndefinedTableError":
            return True
        if getattr(node, "pgcode", None) == "42P01" or getattr(node, "sqlstate", None) == "42P01":
            return True
        nxt = getattr(node, "orig", None)
        node = nxt if nxt is not None else getattr(node, "__cause__", None)
    return False


def _serialize_audit(row: AuditLog) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "actor_email": row.actor_email,
        "actor_role": row.actor_role,
        "action": row.action,
        "target_type": row.target_type,
        "target_id": row.target_id,
        "target": row.target,
        "meta": row.meta,
        "ip": row.ip,
        "request_id": row.request_id,
        "created_at": row.created_at.isoformat() if row.created_at is not None else None,
    }


@router.get("/admin/audit")
@limiter.limit("30/minute")
async def list_audit(
    request: Request,
    action: str | None = Query(None, max_length=100),
    actor: str | None = Query(None, max_length=255),
    target_type: str | None = Query(None, max_length=50),
    target_id: str | None = Query(None, max_length=100),
    since: str | None = Query(None, max_length=40),
    until: str | None = Query(None, max_length=40),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0, le=100_000),
):
    """Журнал аудита. action - префикс; actor - точный email; since/until - ISO datetime."""
    conds: list[Any] = []
    if action:
        conds.append(AuditLog.action.startswith(action, autoescape=True))
    if actor:
        conds.append(AuditLog.actor_email == actor)
    if target_type:
        conds.append(AuditLog.target_type == target_type)
    if target_id:
        conds.append(AuditLog.target_id == target_id)
    if since:
        conds.append(AuditLog.created_at >= _parse_iso(since, "since"))
    if until:
        conds.append(AuditLog.created_at <= _parse_iso(until, "until"))

    async with async_session_factory() as session:
        try:
            total = await session.scalar(select(func.count()).select_from(AuditLog).where(*conds))
            rows = (
                (
                    await session.execute(
                        select(AuditLog)
                        .where(*conds)
                        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )
        except SQLAlchemyError as exc:
            if _is_undefined_table(exc):
                # Таблица ещё не создана (миграция не применена): пустая выборка, а не 500.
                return {"entries": [], "total": 0, "limit": limit, "offset": offset}
            raise

    return {
        "entries": [_serialize_audit(r) for r in rows],
        "total": int(total or 0),
        "limit": limit,
        "offset": offset,
    }


@router.get("/admin/audit/actions")
@limiter.limit("30/minute")
async def list_audit_actions(request: Request):
    """Уникальные action для фильтра UI (алфавит, максимум 200)."""
    stmt = select(AuditLog.action).distinct().order_by(AuditLog.action.asc()).limit(200)
    try:
        async with async_session_factory() as session:
            actions = (await session.execute(stmt)).scalars().all()
    except SQLAlchemyError as exc:
        if _is_undefined_table(exc):
            return {"actions": [], "total": 0}
        raise
    return {"actions": list(actions), "total": len(actions)}
