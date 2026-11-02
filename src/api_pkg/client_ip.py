"""Определение IP клиента без доверия к подделываемым заголовкам.

Проблема: и rate-limit, и журнал сессий брали первый адрес из `X-Forwarded-For`.
Этот заголовок присылает сам клиент, поэтому `curl -H 'X-Forwarded-For: 10.0.0.N'`
давал неограниченное число попыток входа в обход лимита 10/мин.

Правило: заголовку верим только если TCP-соединение пришло от доверенного
прокси (loopback или частная сеть — nginx в docker-сети). Из цепочки берём
самый правый адрес, который не является доверенным прокси, то есть тот,
который добавил наш собственный прокси.
"""

from __future__ import annotations

import ipaddress

from fastapi import Request

#: Сети, которым разрешено присылать X-Forwarded-For (наш nginx / docker bridge).
_TRUSTED_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)

UNKNOWN_IP = "unknown"


def _is_trusted(value: str) -> bool:
    try:
        addr = ipaddress.ip_address(value.strip())
    except ValueError:
        return False
    return any(addr in net for net in _TRUSTED_NETWORKS)


def _peer(request: Request) -> str:
    client = request.client
    return client.host if client and client.host else UNKNOWN_IP


def client_ip(request: Request) -> str:
    """Реальный IP клиента: peer, а XFF — только от доверенного прокси."""
    peer = _peer(request)
    if not _is_trusted(peer):
        # Прямое соединение из интернета: заголовок игнорируем полностью.
        return peer

    raw = request.headers.get("X-Forwarded-For", "")
    if not raw:
        raw = request.headers.get("X-Real-IP", "")
    if not raw:
        return peer

    hops = [h.strip() for h in raw.split(",") if h.strip()]
    valid: list[str] = []
    for hop in hops:
        try:
            ipaddress.ip_address(hop)
        except ValueError:
            # Мусор в заголовке не должен становиться ключом лимита: иначе
            # клиент подставляет произвольные строки и обходит ограничение.
            continue
        valid.append(hop)

    for hop in reversed(valid):
        if not _is_trusted(hop):
            return hop
    return valid[0] if valid else peer


def client_ip_chain_depth(request: Request) -> int:
    """Сколько адресов в цепочке XFF (для аудита подозрительных запросов)."""
    raw = request.headers.get("X-Forwarded-For", "")
    return len([h for h in raw.split(",") if h.strip()])
