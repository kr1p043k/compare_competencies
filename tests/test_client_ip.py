"""Unit tests for src.api_pkg.client_ip: X-Forwarded-For is honoured only from a
trusted proxy peer (loopback / private network).

No DB and no FastAPI app: minimal fake ``Request`` objects exposing only
``.client.host`` and ``.headers.get``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from fastapi import Request

from src.api_pkg.client_ip import client_ip


def _req(peer: str, headers: dict[str, str] | None = None) -> Request:
    """Псевдо-Request: только .client.host и .headers.get()."""
    ns = SimpleNamespace(
        client=SimpleNamespace(host=peer),
        headers=headers if headers is not None else {},
    )
    return cast(Request, ns)


def test_direct_peer_ignores_forwarded_header() -> None:
    # Прямое соединение из интернета: XFF подставляет сам клиент -> игнорируем.
    req = _req("203.0.113.7", {"X-Forwarded-For": "10.0.0.1"})
    assert client_ip(req) == "203.0.113.7"


def test_trusted_peer_uses_forwarded_header() -> None:
    req = _req("127.0.0.1", {"X-Forwarded-For": "198.51.100.9"})
    assert client_ip(req) == "198.51.100.9"


def test_trusted_peer_takes_rightmost_non_trusted_hop() -> None:
    # 10.0.0.5 — доверенный (добавил наш прокси), берём более правый недоверенный.
    req = _req("127.0.0.1", {"X-Forwarded-For": "1.2.3.4, 10.0.0.5"})
    assert client_ip(req) == "1.2.3.4"


def test_trusted_peer_without_header_returns_peer() -> None:
    req = _req("127.0.0.1")
    assert client_ip(req) == "127.0.0.1"


def test_garbage_header_without_usable_hops_falls_back_to_peer() -> None:
    # Заголовок есть, но пригодных адресов нет -> остаётся peer.
    req = _req("127.0.0.1", {"X-Forwarded-For": " , , "})
    assert client_ip(req) == "127.0.0.1"


def test_unparseable_hop_is_skipped() -> None:
    # Мусор в заголовке не должен становиться ключом лимита: иначе клиент
    # подставляет произвольные строки и обходит ограничение.
    req = _req("127.0.0.1", {"X-Forwarded-For": "garbage"})
    assert client_ip(req) == "127.0.0.1"


def test_garbage_before_valid_hop_still_uses_valid_hop() -> None:
    req = _req("127.0.0.1", {"X-Forwarded-For": "not-an-ip, 198.51.100.7"})
    assert client_ip(req) == "198.51.100.7"
