"""Request logging: in-memory buffer + periodic DB flush + frontend log API."""

import asyncio
import base64
import hashlib
import hmac
import json
import re
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
from src.monitoring.metrics import api_latency, api_requests_total
from src.models.krm_models import RequestLog

logger = structlog.get_logger(__name__)

MAX_LOGS = 2000
SECRET_KEY = config.get_secret_key()
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
        payload_b64 = parts[0] + "=" * ((4 - len(parts[0]) % 4) % 4)
        sig_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
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
    __slots__ = ("method", "path", "status", "duration_ms", "user_email", "timestamp", "source", "detail")

    def __init__(self, method: str, path: str, status: int, duration_ms: float, user_email: str | None, source: str = "backend", detail: str | None = None):
        self.method = method
        self.path = path
        self.status = status
        self.duration_ms = round(duration_ms, 1)
        self.user_email = user_email or "anonymous"
        self.source = source
        self.detail = detail
        self.timestamp = datetime.now(timezone.utc).replace(tzinfo=None)


_log_buffer: deque[LogEntry] = deque(maxlen=MAX_LOGS)


def _to_request_log(e: LogEntry, with_detail: bool = True) -> RequestLog:
    """Build RequestLog ORM row from a buffered entry."""
    kwargs: dict[str, Any] = {
        "method": e.method,
        "path": e.path,
        "status": e.status,
        "duration_ms": e.duration_ms,
        "user_email": e.user_email if e.user_email != "anonymous" else None,
        "source": e.source,
        "created_at": e.timestamp,
    }
    if with_detail:
        kwargs["detail"] = e.detail
    return RequestLog(**kwargs)


async def _flush_to_db() -> None:
    """Flush buffered logs to PostgreSQL."""
    if not _log_buffer:
        return
    try:
        from src.database import async_session_factory

        entries = list(_log_buffer)
        _log_buffer.clear()

        try:
            async with async_session_factory() as session:
                for e in entries:
                    session.add(_to_request_log(e, with_detail=True))
                await session.commit()
        except Exception:
            # Pre-migration DB without request_logs.detail — retry slim so rows survive.
            async with async_session_factory() as session:
                for e in entries:
                    session.add(_to_request_log(e, with_detail=False))
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


def get_logs(user: str | None = None, limit: int = 100, source: str | None = None, action: str | None = None) -> list[dict[str, Any]]:
    """Return recent logs from the in-memory buffer.

    NOTE: rows already flushed to PostgreSQL (request_logs) are NOT merged back —
    the admin view is a rolling window of the last MAX_LOGS entries. Audit (AUDIT)
    entries carry actor/action/target in ``detail`` so admin can answer
    "who added/changed recommendation X" within that window.
    """
    entries = list(_log_buffer)
    if user:
        entries = [e for e in entries if e.user_email == user]
    if source:
        entries = [e for e in entries if e.source == source]
    if action:
        entries = [e for e in entries if (e.detail or "").startswith(action)]
    entries = entries[-limit:]

    return [
        {
            "method": e.method,
            "path": e.path,
            "status": e.status,
            "duration_ms": e.duration_ms,
            "user": e.user_email,
            "source": e.source,
            "detail": e.detail,
            "timestamp": e.timestamp.isoformat(),
        }
        for e in entries
    ]


async def actor_from_request(request: Request) -> tuple[str, str]:
    """Resolve (email, role) for audit without failing the request.

    Prefers the session-checked user (same source as require_any_role), falls back
    to signature-verified JWT decode (no DB hit), then to the middleware-attached
    user, else anonymous. Never raises.
    """
    try:
        from src.api_pkg.routers.auth import get_current_user
        user = await get_current_user(request)
        if isinstance(user, dict) and user.get("u"):
            return str(user.get("u")), str(user.get("r") or "?")
    except Exception:
        pass
    try:
        from src.api_pkg.routers.auth import _decode_token
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            data = _decode_token(auth[7:])
            if isinstance(data, dict) and data.get("u"):
                return str(data.get("u")), str(data.get("r") or "?")
    except Exception:
        pass
    state_user: Any = getattr(request.state, "user", None)
    if isinstance(state_user, str) and state_user:
        return state_user, "?"
    if isinstance(state_user, dict) and state_user.get("u"):
        return str(state_user.get("u")), str(state_user.get("r") or "?")
    scope_user = request.scope.get("user")
    if isinstance(scope_user, str) and scope_user:
        return scope_user, "?"
    return "anonymous", "?"


async def audit_action(request: Request, action: str, target: str = "") -> dict[str, Any]:
    """Append an AUDIT entry: who (email/role) did what (action) to which target.

    Call AFTER a successful mutation only — failed attempts stay visible via the
    regular middleware row (method/path/status) without an AUDIT row. Never raises.
    Detail format: "<action> | <target> | by <email> (<role>)".
    """
    try:
        email, role = await actor_from_request(request)
        target = (target or "").strip()
        detail = f"{action} | {target} | by {email} ({role})" if target else f"{action} | by {email} ({role})"
        entry = LogEntry(
            method="AUDIT",
            path=request.url.path,
            status=200,
            duration_ms=0,
            user_email=email,
            source="backend",
            detail=detail,
        )
        _log_buffer.append(entry)
        logger.info("audit", action=action, target=target[:200], actor=email, role=role)
        return {"actor_email": email, "actor_role": role, "action": action, "target": target, "detail": detail}
    except Exception as exc:
        logger.warning("audit_failed", action=action, error=str(exc))
        return {"actor_email": "anonymous", "actor_role": "?", "action": action, "target": target, "detail": None}


def get_logs_by_user() -> dict[str, int]:
    entries = list(_log_buffer)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.user_email] = counts.get(e.user_email, 0) + 1
    return counts


def _metric_path(request: Request) -> str:
    """Normalize path for metric labels: use the route template to keep cardinality bounded."""
    route = request.scope.get("route")
    if route is not None:
        template = getattr(route, "path", None)
        if template:
            return template
    return re.sub(r"\d+", "{id}", request.url.path)


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = datetime.now(timezone.utc)
        user = _extract_user(request)
        request.scope["user"] = user
        request.state.user = user
        response = await call_next(request)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        if not request.url.path.startswith("/api/"):
            return response
        label_path = _metric_path(request)
        api_requests_total.labels(method=request.method, path=label_path, status=response.status_code).inc()
        api_latency.labels(method=request.method, path=label_path, status=response.status_code).observe(elapsed / 1000)
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
        if response.status_code >= 500 and not getattr(request.state, "system_error_notified", False):
            request.state.system_error_notified = True
            from src.notifications.system import queue_system_error
            queue_system_error(
                f"Ошибка API: {request.method} {request.url.path}",
                f"Статус {response.status_code} за {elapsed:.0f} мс | user={user or 'anonymous'} | path={request.url.path}",
                severity="error" if response.status_code >= 500 else "warning",
                article_url=f"api:{request.method} {request.url.path}",
                dedupe=True,
            )
        elif response.status_code < 500:
            from src.notifications.system import maybe_resolve_api_errors
            asyncio.ensure_future(maybe_resolve_api_errors(request.method, request.url.path))
        return response


class FrontendLogMiddleware(BaseHTTPMiddleware):
    """Log frontend-triggered actions via dedicated endpoint."""
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if request.method == "POST" and request.url.path.startswith("/api/admin/"):
            if request.url.path in ("/api/admin/logs", "/api/admin/users"):
                return response
            user = request.scope.get("user", None) or "frontend"
            _log_buffer.append(LogEntry(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=0,
                user_email=user,
                source="frontend",
            ))
        return response
