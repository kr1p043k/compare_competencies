"""Проверка Origin для WebSocket-хендшейков (защита от CSWSH).

Браузер прикладывает Origin к WS-апгрейду, и без проверки сторонний сайт
может открыть сокет от имени залогиненного пользователя (SameSite=Lax
смягчает, но не закрывает: cookie уходит при навигации верхнего уровня).

Разрешено: пустой Origin (не-браузерные клиенты: скрипты, smoke-тесты) и
Origin, совпадающий с Host запроса либо с одним из ALLOWED_ORIGINS.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import structlog

logger = structlog.get_logger(__name__)


def _host_of(origin: str) -> str:
    try:
        return urlsplit(origin).netloc.lower()
    except ValueError:
        return ""


def origin_allowed(websocket, allowed_origins: str = "") -> bool:
    """True, если Origin сокета допустим.

    allowed_origins - строка из конфига (список через запятую), допускается '*'.
    """
    origin = websocket.headers.get("origin", "")
    if not origin:
        # Не браузер: проверять нечего, авторизация всё равно по токену.
        return True

    allowed = [o.strip().lower() for o in (allowed_origins or "").split(",") if o.strip()]
    if "*" in allowed:
        return True

    origin_host = _host_of(origin)
    if not origin_host:
        return False

    # Тот же хост, что и у сокета: обычный случай (UI на том же origin).
    host = websocket.headers.get("host", "").lower()
    if host and origin_host == host:
        return True

    return any(_host_of(a) == origin_host for a in allowed)


def reject_if_bad_origin(websocket, allowed_origins: str = "") -> bool:
    """True, если сокет нужно закрыть (Origin не разрешён)."""
    if origin_allowed(websocket, allowed_origins):
        return False
    logger.warning(
        "ws_origin_rejected",
        origin=websocket.headers.get("origin"),
        host=websocket.headers.get("host"),
        path=websocket.url.path,
    )
    return True
