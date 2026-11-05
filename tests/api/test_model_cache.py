"""Кэш моделей кластеризации: один joblib.load на версию файла, инвалидация по mtime, reset."""
import os

import joblib
import pytest

from src import config
from src.analyzers.clustering.vacancy_clustering import (
    load_cached_clusterer,
    load_cached_model_data,
    reset_model_cache,
)


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    d = tmp_path / "clusters"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "VACANCY_CLUSTERS_CACHE_DIR", d)
    reset_model_cache()
    yield d
    reset_model_cache()


def _counting_load(monkeypatch):
    """Подмена joblib.load, считающая реальные вызовы (dump не трогаем)."""
    calls = {"n": 0}
    real_load = joblib.load

    def counting(filepath, *args, **kwargs):
        calls["n"] += 1
        return real_load(filepath, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting)
    return calls


def test_accessor_loads_file_once(model_dir, monkeypatch):
    joblib.dump({"version": 1}, model_dir / "vacancy_clusters_junior.joblib")
    calls = _counting_load(monkeypatch)

    first = load_cached_model_data("junior")
    second = load_cached_model_data("junior")

    assert first == second == {"version": 1}
    assert calls["n"] == 1


def test_mtime_change_triggers_reload(model_dir, monkeypatch):
    path = model_dir / "vacancy_clusters_junior.joblib"
    joblib.dump({"version": 1}, path)
    calls = _counting_load(monkeypatch)

    load_cached_model_data("junior")
    assert calls["n"] == 1

    # Только mtime: содержимое и размер не меняются.
    st = path.stat()
    os.utime(path, (st.st_atime + 5, st.st_mtime + 5))

    load_cached_model_data("junior")
    assert calls["n"] == 2


def test_reset_forces_reload(model_dir, monkeypatch):
    joblib.dump({"version": 1}, model_dir / "vacancy_clusters_junior.joblib")
    calls = _counting_load(monkeypatch)

    load_cached_model_data("junior")
    reset_model_cache()
    load_cached_model_data("junior")

    assert calls["n"] == 2


def test_cached_clusterer_single_load(model_dir, monkeypatch):
    joblib.dump(
        {
            "model": None,
            "clusterer_type": "kmeans",
            "labels": [0, 1, 1],
            "centers": None,
            "vacancy_ids": ["1", "2", "3"],
            "vacancy_skills": [["python"], ["sql"], ["sql"]],
            "n_clusters": 2,
            "min_cluster_size": 15,
            "label_to_center_idx": {0: 0, 1: 1},
            "skill_weights": {},
        },
        model_dir / "vacancy_clusters_junior.joblib",
    )
    calls = _counting_load(monkeypatch)

    first = load_cached_clusterer("junior")
    second = load_cached_clusterer("junior")

    assert calls["n"] == 1
    assert first is not None and first.is_fitted and first.n_clusters_ == 2
    assert second is not None and second is not first


def test_missing_file_returns_none(model_dir):
    assert load_cached_model_data("senior") is None
