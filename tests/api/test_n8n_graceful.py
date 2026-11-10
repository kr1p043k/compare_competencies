"""n8n-интеграция должна отдавать graceful-коды, а не 500.

Регресс на жалобу "GET /api/n8n/webhooks | 500: N8N_API_KEY not configured":
- ключ не сконфигурирован -> 503 (сервис не настроен), НЕ 500;
- ключ задан, но заголовок отсутствует/неверный -> 401;
- не-ASCII ключ -> 401, а не 500 (сравнение через compare_digest на bytes).
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import SecretStr

from src import config
from src.api_pkg import create_app


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def test_not_configured_is_503(monkeypatch) -> None:
    monkeypatch.setattr(config, "N8N_API_KEY", None, raising=False)
    r = _client().get("/api/n8n/webhooks")
    assert r.status_code == 503
    assert r.status_code != 500


def test_missing_header_is_401(monkeypatch) -> None:
    monkeypatch.setattr(config, "N8N_API_KEY", SecretStr("secret"), raising=False)
    r = _client().get("/api/n8n/webhooks")
    assert r.status_code == 401


def test_configured_with_wrong_key_is_401(monkeypatch) -> None:
    monkeypatch.setattr(config, "N8N_API_KEY", SecretStr("secret"), raising=False)
    r = _client().get("/api/n8n/webhooks", headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401
