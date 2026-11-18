"""Единый исходящий HTTP-клиент: таймауты connect/read и ретраи транспорта."""
from __future__ import annotations

import httpx

from src.infrastructure import http_client
from src.infrastructure.hub_publications import HubPublicationsClient


def test_sync_client_uses_config_timeouts():
    c = http_client.new_sync_client()
    assert c.timeout.connect == http_client.config.settings.HTTP_CONNECT_TIMEOUT_S
    assert c.timeout.read == http_client.config.settings.HTTP_READ_TIMEOUT_S


def test_async_client_uses_config_timeouts():
    c = http_client.new_async_client()
    assert c.timeout.connect == http_client.config.settings.HTTP_CONNECT_TIMEOUT_S
    assert c.timeout.read == http_client.config.settings.HTTP_READ_TIMEOUT_S


def test_explicit_timeout_overrides_all_phases():
    c = http_client.new_sync_client(timeout=2.0)
    assert c.timeout.connect == 2.0
    assert c.timeout.read == 2.0


def test_explicit_connect_read_kwargs():
    c = http_client.new_async_client(connect=1.0, read=9.0)
    assert c.timeout.connect == 1.0
    assert c.timeout.read == 9.0


def test_retries_passed_to_transport(monkeypatch):
    seen: dict[str, object] = {}
    real = httpx.HTTPTransport

    def spy(*args, **kwargs):
        seen["retries"] = kwargs.get("retries")
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "HTTPTransport", spy)
    http_client.new_sync_client(retries=7)
    assert seen["retries"] == 7


def test_transport_injection_used():
    mt = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    c = http_client.new_sync_client(transport=mt)
    assert c.get("https://example.test/").status_code == 200


def test_hub_builds_client_via_factory(monkeypatch):
    calls: dict[str, object] = {}
    real = http_client.new_async_client

    def spy(**kwargs):
        calls.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(http_client, "new_async_client", spy)
    client = HubPublicationsClient(base_url="https://hub.sfedu.ru", token=None)
    built = client._http()
    assert calls.get("timeout") == client.timeout
    assert "transport" in calls
    assert isinstance(built, httpx.AsyncClient)
