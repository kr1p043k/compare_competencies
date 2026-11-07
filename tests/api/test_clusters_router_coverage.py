"""Покрытие роутера /clusters: сводка и кластеры по уровню.

DB и joblib не трогаем: подменяем _load_clusterer фейком с нужными методами.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.clusters as cl
from tests.conftest import open_all_gates


class _FakeClusterer:
    def __init__(self, n_clusters: int = 2, fitted: bool = True) -> None:
        self.n_clusters_ = n_clusters
        self.is_fitted = fitted
        self.clusterer_type = "kmeans"

    def _generate_cluster_name(self, cid: int) -> str:
        return f"Cluster {cid}"

    def get_top_skills_in_cluster(self, _cid: int, top_n: int = 5) -> list[str]:
        return ["python", "sql"][:top_n]


@pytest.fixture()
def client():
    from src.api_pkg import create_app

    app = create_app()
    # FastAPI 0.162 включает роутеры лениво, поэтому обход app.routes зависимостей
    # не видит; open_all_gates снимает гейты правильно.
    open_all_gates(app, role="admin")
    for lim in [getattr(app.state, "limiter", None), getattr(cl, "limiter", None)]:
        if lim is not None:
            lim.reset()
    return TestClient(app)


def test_summary_lists_all_levels(client, monkeypatch):
    monkeypatch.setattr(cl, "_load_clusterer", lambda _lvl: _FakeClusterer())
    r = client.get("/api/clusters/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    # Ключ на каждый уровень опыта.
    assert body
    for value in body.values():
        assert value["clusters"] == 2
        assert value["type"] == "kmeans"
        assert len(value["top_clusters"]) == 2


def test_summary_marks_not_fitted(client, monkeypatch):
    monkeypatch.setattr(cl, "_load_clusterer", lambda _lvl: None)
    r = client.get("/api/clusters/summary")
    assert r.status_code == 200
    assert all(v == {"error": "not_fitted"} for v in r.json().values())


def test_summary_marks_unfitted_clusterer(client, monkeypatch):
    monkeypatch.setattr(cl, "_load_clusterer", lambda _lvl: _FakeClusterer(fitted=False))
    r = client.get("/api/clusters/summary")
    assert r.status_code == 200
    assert all(v == {"error": "not_fitted"} for v in r.json().values())


def test_by_level_returns_clusters(client, monkeypatch):
    monkeypatch.setattr(cl, "_load_clusterer", lambda _lvl: _FakeClusterer())
    r = client.get("/api/clusters/middle")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level"] == "middle"
    assert len(body["clusters"]) == 2
    assert body["clusters"][0]["top_skills"] == ["python", "sql"]


def test_by_level_503_when_not_fitted(client, monkeypatch):
    monkeypatch.setattr(cl, "_load_clusterer", lambda _lvl: None)
    r = client.get("/api/clusters/junior")
    assert r.status_code == 503
    assert "не загружен" in r.json()["detail"]


def test_by_level_422_on_bad_level(client, monkeypatch):
    monkeypatch.setattr(cl, "_load_clusterer", lambda _lvl: _FakeClusterer())
    r = client.get("/api/clusters/unknown")
    assert r.status_code == 422
