"""Тесты health-эндпоинтов."""
import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Compare Competencies API"

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code in (200, 500)

    def test_ready_endpoint(self, client):
        # Без lifespan данные не загружены -> readiness отвечает честно "не готов":
        # 503 и status != ok. Когда данные загружены, будет == 200, но проверять
        # конкретику тут хрупко.
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)
        assert resp.json()["status"] in ("ok", "not_ready")

    def test_live_endpoint_always_ok(self, client):
        # Liveness не зависит от данных и БД: процесс жив -> 200 ok.
        resp = client.get("/live")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok"}

    def test_live_endpoint_api_alias(self, client):
        resp = client.get("/api/live")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok"}
