"""Общий лимитер запросов с честным ключом клиента.

Проблема: почти все роутеры создавали свой `Limiter(key_func=get_remote_address)`.
В проде весь трафик приходит через nginx (docker-сеть), поэтому
`request.client.host` — это IP nginx-контейнера, одинаковый для всех
пользователей. Лимит «60/minute» действовал на весь университет сразу:
на лекции с аудиторией первый же поток запросов упирался в лимит,
который никто не тратил.

Ключ здесь: email из JWT, если запрос авторизован (`request.state.user`
заполняется зависимостями auth), иначе реальный IP через `client_ip` —
он не доверяет `X-Forwarded-For` с недоверенных соединений, иначе лимит
обходится подменой заголовка.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter

from src import config
from src.api_pkg.client_ip import client_ip


def rate_limit_key(request: Request) -> str:
    """Ключ лимита: пользователь (если авторизован) или реальный IP."""
    user = getattr(request.state, "user", None)
    if user and isinstance(user, dict):
        email = user.get("u")
        if email:
            return f"user:{email}"
    return f"ip:{client_ip(request)}"


limiter = Limiter(
    key_func=rate_limit_key,
    enabled=getattr(config.settings, "RATELIMIT_ENABLED", True),
)
