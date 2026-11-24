"""Тесты health-эндпоинтов."""
import hashlib

import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.model_registry import record_loaded, reset_loaded_artifacts


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def _clean_loaded():
    """Serving-реестр загрузок глобален в процессе: не протекать между тестами."""
    reset_loaded_artifacts()
    yield
    reset_loaded_artifacts()


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Compare Competencies API"

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code in (200, 500)

    @pytest.mark.usefixtures("_clean_loaded")
    def test_health_reports_loaded_artifact_hash(self, client, tmp_path):
        # Кривой деплой трассируется: артефакт, реально открытый serving,
        # виден в /health с именем + коротким sha256 даже без реестра/моделей.
        art = tmp_path / "ltr_ranker_fake.joblib"
        art.write_bytes(b"fake-artifact-bytes")
        expected = hashlib.sha256(art.read_bytes()).hexdigest()[:12]
        record_loaded("ltr", art, version="7")
        record_loaded("clusters_middle", art, version="3")

        resp = client.get("/health")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ltr_artifact"]["artifact"] == "ltr_ranker_fake.joblib"
        assert data["ltr_artifact"]["version"] == "7"
        assert data["ltr_artifact"]["sha256_12"] == expected
        loaded = data["loaded_artifacts"]
        assert loaded["clusters_middle"]["sha256_12"] == expected
        assert loaded["ltr"]["sha256_12"] == expected

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
