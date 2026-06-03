import json
import time
import base64
import hmac
import hashlib
from collections import deque
from datetime import datetime, timezone
from typing import Any
import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = structlog.get_logger(__name__)

MAX_LOGS = 2000
SECRET_KEY = "compare-competencies-secret-key-change-in-production"


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
    __slots__ = ("method", "path", "status", "duration_ms", "user", "timestamp")

    def __init__(self, method: str, path: str, status: int, duration_ms: float, user: str | None):
        self.method = method
        self.path = path
        self.status = status
        self.duration_ms = round(duration_ms, 1)
        self.user = user or "anonymous"
        self.timestamp = datetime.now(timezone.utc).isoformat()


_log_buffer: deque[LogEntry] = deque(maxlen=MAX_LOGS)


def get_logs(user: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    entries = list(_log_buffer)
    if user:
        entries = [e for e in entries if e.user == user]
    entries = entries[-limit:]
    return [
        {"method": e.method, "path": e.path, "status": e.status, "duration_ms": e.duration_ms, "user": e.user, "timestamp": e.timestamp}
        for e in entries
    ]


def get_logs_by_user() -> dict[str, int]:
    entries = list(_log_buffer)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.user] = counts.get(e.user, 0) + 1
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
            user=user,
        ))
        return response
