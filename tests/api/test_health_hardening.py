"""Hardening-тесты health/ready/metrics (fail-closed readiness + metrics token).

/ready раньше всегда отвечал 200, даже когда данные не загружены. Эти тесты
фиксируют новый контракт: 503 + status == "not_ready", пока критическое
состояние (deps.is_ready / пул БД) отсутствует, а /metrics скрейпится по
X-Metrics-Token без роли admin. Стиль - как в tests/api/test_zun_idor_guard.py:
TestClient(create_app()) без запуска lifespan + monkeypatch.
"""

from fastapi.testclient import TestClient

import src.db as db
from src import config
from src.api_pkg import create_app, deps

TOKEN = "correct-metrics-token"


def _client():
    """Свежий TestClient без lifespan: global-состояние задаётся monkeypatch'ем."""
    return TestClient(create_app(), raise_server_exceptions=False)


class TestHealthAlways200:
    def test_health_200_even_when_not_ready(self, monkeypatch):
        monkeypatch.setattr(deps, "is_ready", False)
        monkeypatch.setattr(deps, "evaluator", None)
        resp = _client().get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestReadyFailClosed:
    def _ready(self, monkeypatch, *, is_ready, pool):
        monkeypatch.setattr(deps, "is_ready", is_ready)
        monkeypatch.setattr(db, "pool", pool)
        return _client().get("/ready")

    def test_ready_503_when_is_ready_false(self, monkeypatch):
        resp = self._ready(monkeypatch, is_ready=False, pool=object())
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert "data_not_loaded" in body["problems"]

    def test_ready_503_when_db_pool_none(self, monkeypatch):
        # is_ready=True, но пула БД нет - fail-closed по второму критическому условию.
        resp = self._ready(monkeypatch, is_ready=True, pool=None)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert "db_pool_unavailable" in body["problems"]

    def test_ready_200_when_ready(self, monkeypatch):
        resp = self._ready(monkeypatch, is_ready=True, pool=object())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["problems"] == []

    def test_ready_preserves_components_field(self, monkeypatch):
        resp = self._ready(monkeypatch, is_ready=True, pool=object())
        assert set(resp.json()["components"]) == {
            "evaluator",
            "recommendation_engine",
            "clusterer",
            "trend_analyzer",
        }

    def test_ready_reports_both_problems(self, monkeypatch):
        resp = self._ready(monkeypatch, is_ready=False, pool=None)
        assert resp.status_code == 503
        body = resp.json()
        assert "data_not_loaded" in body["problems"]
        assert "db_pool_unavailable" in body["problems"]


class TestMetricsToken:
    def test_metrics_401_without_credentials(self, monkeypatch):
        monkeypatch.setattr(config.settings, "METRICS_TOKEN", None)
        resp = _client().get("/metrics")
        assert resp.status_code in (401, 403)

    def test_metrics_200_with_correct_token(self, monkeypatch):
        monkeypatch.setattr(config.settings, "METRICS_TOKEN", TOKEN)
        resp = _client().get("/metrics", headers={"X-Metrics-Token": TOKEN})
        assert resp.status_code == 200

    def test_metrics_401_with_wrong_token(self, monkeypatch):
        # Неверный токен не раскрывает наличие токена: падаем в role-check (401).
        monkeypatch.setattr(config.settings, "METRICS_TOKEN", TOKEN)
        resp = _client().get("/metrics", headers={"X-Metrics-Token": "wrong"})
        assert resp.status_code in (401, 403)

    def test_metrics_200_for_admin_override(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        async def _admin(_request):
            return {"uid": "u-a", "u": "admin@example.com", "r": "admin", "t": None}

        monkeypatch.setattr(auth, "get_current_user", _admin)
        monkeypatch.setattr(config.settings, "METRICS_TOKEN", None)
        assert _client().get("/metrics").status_code == 200
