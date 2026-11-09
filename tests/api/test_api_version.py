"""GET /api/version отдаёт версию публичного контракта."""
from fastapi.testclient import TestClient

from src import config
from src.api_pkg import create_app


def test_api_version_endpoint():
    client = TestClient(create_app())
    r = client.get("/api/version")
    assert r.status_code == 200, r.text
    assert r.json() == {"api_version": config.API_VERSION}
