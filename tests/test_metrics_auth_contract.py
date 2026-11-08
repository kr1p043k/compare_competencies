"""Контрактный тест авторизации /metrics (заголовок X-Metrics-Token).

Фиксирует ровно тот контракт, нарушение которого ломало скрейп Prometheus:
backend читает секрет из заголовка `X-Metrics-Token` (константное сравнение),
а НЕ из `Authorization: Bearer`. Раньше `prometheus/prometheus.yml` слал
`authorization: {type: Bearer}`, backend отвечал 401/403, и `up{job="backend"}`
падал в 0 вместе со всеми алертами на `api_requests_total`.

Стиль - как в tests/api/test_health_hardening.py: TestClient(create_app())
без lifespan + monkeypatch.
"""

from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth
from src import config
from src.api_pkg import create_app

TOKEN = "correct-metrics-token"


def _client() -> TestClient:
    """Свежий TestClient без lifespan: состояние задаётся monkeypatch'ем."""
    return TestClient(create_app(), raise_server_exceptions=False)


def test_metrics_without_credentials_rejected(monkeypatch):
    # Токен сконфигурирован, но заголовок не прислан -> роль admin не получена.
    monkeypatch.setattr(config.settings, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics")
    assert resp.status_code != 200


def test_metrics_200_with_correct_x_metrics_token(monkeypatch):
    monkeypatch.setattr(config.settings, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics", headers={"X-Metrics-Token": TOKEN})
    assert resp.status_code == 200


def test_metrics_wrong_token_rejected(monkeypatch):
    monkeypatch.setattr(config.settings, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics", headers={"X-Metrics-Token": "wrong-token"})
    assert resp.status_code != 200


def test_metrics_bearer_authorization_accepted(monkeypatch):
    # Prometheus ставит только `authorization: {type: Bearer}` (prometheus.yml),
    # поэтому /metrics принимает токен и в `Authorization: Bearer`, и в
    # `X-Metrics-Token` (см. _metrics_access в src/api_pkg/__init__.py).
    monkeypatch.setattr(config.settings, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200


def test_metrics_wrong_bearer_rejected(monkeypatch):
    monkeypatch.setattr(config.settings, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code != 200


def test_metrics_admin_override_when_token_unset(monkeypatch):
    # Фолбэк сохранён: без METRICS_TOKEN доступ по роли admin.
    async def _admin(_request: object) -> dict:
        return {"uid": "u-a", "u": "admin@example.com", "r": "admin", "t": None}

    monkeypatch.setattr(auth, "get_current_user", _admin)
    monkeypatch.setattr(config.settings, "METRICS_TOKEN", None)
    assert _client().get("/metrics").status_code == 200
