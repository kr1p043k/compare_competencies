"""Единый исходящий HTTP-клиент: таймауты connect/read + ретраи на транспорте.

Все внешние вызовы (hh, hub, уведомления, ...) должны строить клиент здесь, а
не собирать ``httpx.Client(...)`` руками в каждом модуле. Это даёт:

  * раздельные ``connect``/``read`` таймауты из конфига (``HTTP_CONNECT_TIMEOUT_S``
    / ``HTTP_READ_TIMEOUT_S``): «медленно соединяется» и «медленно отвечает» -
    разные сбои с разной ценой ожидания;
  * ретраи на уровне транспорта (``HTTP_RETRIES``) для идемпотентных запросов;
  * одну точку настройки вместо россыпи ``timeout=`` по коду.

Явные переопределения (``timeout``/``transport``/``retries``/...) нужны тестам и
особым случаям: например, ``transport`` позволяет подставить ``MockTransport``,
а ``timeout`` - сохранить историческое поведение конкретного клиента.
"""
from __future__ import annotations

import httpx

from src import config


def _timeout(
    timeout: float | httpx.Timeout | None,
    connect: float | None,
    read: float | None,
    write: float | None,
    pool: float | None,
) -> httpx.Timeout | float:
    """Явный ``timeout`` важнее поэтапных; иначе собираем из конфига."""
    if timeout is not None:
        return timeout
    connect = config.settings.HTTP_CONNECT_TIMEOUT_S if connect is None else connect
    read = config.settings.HTTP_READ_TIMEOUT_S if read is None else read
    return httpx.Timeout(
        connect=connect,
        read=read,
        write=read if write is None else write,
        pool=connect if pool is None else pool,
    )


def new_async_client(
    *,
    timeout: float | httpx.Timeout | None = None,
    connect: float | None = None,
    read: float | None = None,
    write: float | None = None,
    pool: float | None = None,
    retries: int | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    headers: dict[str, str] | None = None,
    limits: httpx.Limits | None = None,
) -> httpx.AsyncClient:
    """``httpx.AsyncClient`` с общими таймаутами и ретраями транспорта."""
    if transport is None:
        transport = httpx.AsyncHTTPTransport(
            retries=config.settings.HTTP_RETRIES if retries is None else retries
        )
    return httpx.AsyncClient(
        timeout=_timeout(timeout, connect, read, write, pool),
        transport=transport,
        headers=headers,
        limits=limits,
    )


def new_sync_client(
    *,
    timeout: float | httpx.Timeout | None = None,
    connect: float | None = None,
    read: float | None = None,
    write: float | None = None,
    pool: float | None = None,
    retries: int | None = None,
    transport: httpx.BaseTransport | None = None,
    headers: dict[str, str] | None = None,
    limits: httpx.Limits | None = None,
) -> httpx.Client:
    """``httpx.Client`` с общими таймаутами и ретраями транспорта."""
    if transport is None:
        transport = httpx.HTTPTransport(
            retries=config.settings.HTTP_RETRIES if retries is None else retries
        )
    return httpx.Client(
        timeout=_timeout(timeout, connect, read, write, pool),
        transport=transport,
        headers=headers,
        limits=limits,
    )
