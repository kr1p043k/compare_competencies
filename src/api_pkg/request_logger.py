"""Request logging: in-memory buffer + periodic DB flush + frontend log API."""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import Request
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src import config
from src.models.krm_models import RequestLog

logger = structlog.get_logger(__name__)

MAX_LOGS = 2000
SECRET_KEY = config.SECRET_KEY
FLUSH_INTERVAL = 10  # seconds
FLUSH_BATCH = 100    # entries


def _extract_user(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64 = parts[0] + "=" * (4 - len(parts[0]) % 4)
        sig_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = base64.urlsafe_b64decode(payload_b64).decode()
        expected_sig = base64.urlsafe_b64decode(sig_b64)
        actual_sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        data = json.loads(payload)
        if data.get("t", 0) < time.time():
            return None
        return data.get("u")
    except Exception:
        return None


class LogEntry:
    __slots__ = ("method", "path", "status", "duration_ms", "user_email", "timestamp", "source")

    def __init__(self, method: str, path: str, status: int, duration_ms: float, user_email: str | None, source: str = "backend"):
        self.method = method
        self.path = path
        self.status = status
        self.duration_ms = round(duration_ms, 1)
        self.user_email = user_email or "anonymous"
        self.source = source
        self.timestamp = datetime.now(timezone.utc).replace(tzinfo=None)


_log_buffer: deque[LogEntry] = deque(maxlen=MAX_LOGS)


async def _flush_to_db() -> None:
    """Flush buffered logs to PostgreSQL."""
    if not _log_buffer:
        return
    try:
        from src.database import async_session_factory

        entries = list(_log_buffer)
        _log_buffer.clear()

        async with async_session_factory() as session:
            for e in entries:
                session.add(RequestLog(
                    method=e.method,
                    path=e.path,
                    status=e.status,
                    duration_ms=e.duration_ms,
                    user_email=e.user_email if e.user_email != "anonymous" else None,
                    source=e.source,
                    created_at=e.timestamp,
                ))
            await session.commit()
    except Exception as exc:
        logger.warning("log_flush_failed", error=str(exc))


async def _periodic_flush() -> None:
    """Background task: flush every FLUSH_INTERVAL seconds."""
    while True:
        await asyncio.sleep(FLUSH_INTERVAL)
        await _flush_to_db()


def start_log_flusher() -> None:
    """Start the background log flusher (call from startup)."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(_periodic_flush())
    else:
        loop.create_task(_periodic_flush())


def get_logs(user: str | None = None, limit: int = 100, source: str | None = None) -> list[dict[str, Any]]:
    """Return recent logs, merging buffer + DB."""
    entries = list(_log_buffer)
    if user:
        entries = [e for e in entries if e.user_email == user]
    if source:
        entries = [e for e in entries if e.source == source]
    entries = entries[-limit:]

    return [
        {
            "method": e.method,
            "path": e.path,
            "status": e.status,
            "duration_ms": e.duration_ms,
            "user": e.user_email,
            "source": e.source,
            "timestamp": e.timestamp.isoformat(),
        }
        for e in entries
    ]


def get_logs_by_user() -> dict[str, int]:
    entries = list(_log_buffer)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.user_email] = counts.get(e.user_email, 0) + 1
    return counts


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = datetime.now(timezone.utc)
        user = _extract_user(request)
        request.state.user = user
        response = await call_next(request)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        if not request.url.path.startswith("/api/"):
            return response
        _log_buffer.append(LogEntry(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=elapsed,
            user_email=user,
            source="backend",
        ))
        if len(_log_buffer) >= FLUSH_BATCH:
            asyncio.ensure_future(_flush_to_db())
        return response


class FrontendLogMiddleware(BaseHTTPMiddleware):
    """Log frontend-triggered actions via dedicated endpoint."""
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if request.method == "POST" and request.url.path.startswith("/api/admin/"):
            if request.url.path in ("/api/admin/logs", "/api/admin/users"):
                return response
            user = getattr(request.state, "user", None) or "frontend"
            _log_buffer.append(LogEntry(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=0,
                user_email=user,
                source="frontend",
            ))
        return response
