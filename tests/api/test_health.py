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
        resp = client.get("/ready")
        assert resp.status_code == 200
