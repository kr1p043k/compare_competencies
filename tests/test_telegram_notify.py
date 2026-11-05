"""Tests for src.notifications.telegram."""

import pytest

from src import config
from src.notifications import telegram as tg


def _make_fake_client_class(response_status=200, text="ok", raise_on_post=None):
    class FakeResponse:
        def __init__(self):
            self.status_code = response_status
            self.text = text

    class FakeClient:
        instances = []

        def __init__(self, *args, **kwargs):
            self.calls = []
            FakeClient.instances.append(self)

        async def post(self, url, json=None):
            if raise_on_post:
                raise raise_on_post
            self.calls.append((url, json))
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    return FakeClient


@pytest.fixture
def fake_client_class(monkeypatch):
    cls = _make_fake_client_class()
    monkeypatch.setattr(tg.httpx, "AsyncClient", cls)
    return cls


@pytest.fixture(autouse=True)
def reset_token(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "test-token-123")


@pytest.mark.asyncio
async def test_send_telegram_success(fake_client_class):
    result = await tg.send_telegram("123", "hello")
    assert result is True
    client = fake_client_class.instances[-1]
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_send_telegram_url_contains_token(fake_client_class):
    await tg.send_telegram("12345", "hello", parse_mode="HTML")
    client = fake_client_class.instances[-1]
    url, _ = client.calls[0]
    assert "bottest-token-123/sendMessage" in url


@pytest.mark.asyncio
async def test_send_telegram_payload_preserved(fake_client_class):
    await tg.send_telegram("777", "*bold*", parse_mode="Markdown")
    client = fake_client_class.instances[-1]
    _, payload = client.calls[0]
    assert payload["chat_id"] == "777"
    assert payload["text"] == "*bold*"
    assert payload["parse_mode"] == "Markdown"
    assert payload["disable_web_page_preview"] is False


@pytest.mark.asyncio
async def test_send_telegram_failure_returns_false(monkeypatch):
    cls = _make_fake_client_class(response_status=503, text="busy")
    monkeypatch.setattr(tg.httpx, "AsyncClient", cls)
    result = await tg.send_telegram("123", "hello")
    assert result is False


@pytest.mark.asyncio
async def test_send_telegram_exception_returns_false(monkeypatch):
    cls = _make_fake_client_class(raise_on_post=ConnectionError("boom"))
    monkeypatch.setattr(tg.httpx, "AsyncClient", cls)
    result = await tg.send_telegram("123", "hello")
    assert result is False


@pytest.mark.asyncio
async def test_send_telegram_empty_token(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "")
    result = await tg.send_telegram("123", "hello")
    assert result is False


@pytest.mark.asyncio
async def test_send_telegram_none_token(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", None)
    result = await tg.send_telegram("123", "hello")
    assert result is False
