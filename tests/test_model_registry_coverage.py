"""Покрытие src/model_registry.py: версии артефактов, регистрация, prune.

Реальные модели не нужны: артефакты - обычные файлы в tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.model_registry import ModelRegistry


def _artifact(tmp_path: Path, name: str = "model.joblib", size: int = 16) -> Path:
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return p


# ---------- индекс ----------


def test_fresh_registry_has_empty_index(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    assert reg.list_versions("ltr") == []
    assert reg.latest("ltr") is None
    assert reg.get("ltr", "1") is None


def test_index_persists_between_instances(tmp_path: Path):
    art = _artifact(tmp_path)
    ModelRegistry(tmp_path).register("ltr", art, metrics={"ndcg": 0.9})

    reg2 = ModelRegistry(tmp_path)
    versions = reg2.list_versions("ltr")
    assert len(versions) == 1
    assert versions[0]["metrics"] == {"ndcg": 0.9}
    assert versions[0]["size_bytes"] == 16


def test_broken_index_is_ignored(tmp_path: Path):
    (tmp_path / "registry_index.json").write_text("{ broken", encoding="utf-8")
    reg = ModelRegistry(tmp_path)
    assert reg.list_versions("ltr") == []


# ---------- register ----------


def test_register_missing_artifact_returns_err(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    res = reg.register("ltr", tmp_path / "nope.joblib")
    assert res.is_err()


def test_register_assigns_incrementing_versions(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    a1 = _artifact(tmp_path, "a.joblib")
    a2 = _artifact(tmp_path, "b.joblib")
    assert reg.register("ltr", a1).unwrap() == "1"
    assert reg.register("ltr", a2).unwrap() == "2"
    assert [v["version"] for v in reg.list_versions("ltr")] == ["1", "2"]


def test_latest_returns_last_registered(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    a1 = _artifact(tmp_path, "a.joblib")
    a2 = _artifact(tmp_path, "b.joblib")
    reg.register("ltr", a1)
    reg.register("ltr", a2)
    assert reg.latest("ltr")["path"] == str(a2.resolve())


def test_get_by_version(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    reg.register("ltr", _artifact(tmp_path, "a.joblib"))
    reg.register("ltr", _artifact(tmp_path, "b.joblib"))
    assert reg.get("ltr", "1")["version"] == "1"
    assert reg.get("ltr", "9") is None


def test_register_writes_tags(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    reg.register("clusters", _artifact(tmp_path), tags={"level": "middle"})
    assert reg.latest("clusters")["tags"] == {"level": "middle"}


# ---------- prune ----------


def test_prune_keeps_last_n_and_deletes_files(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    paths = []
    for i in range(5):
        p = _artifact(tmp_path, f"m{i}.joblib")
        paths.append(p)
        reg.register("ltr", p)

    removed = reg.prune("ltr", keep_last=2)
    assert removed == 3
    assert len(reg.list_versions("ltr")) == 2
    # Удалены первые три артефакта, последние два остались.
    assert not paths[0].exists()
    assert not paths[2].exists()
    assert paths[3].exists() and paths[4].exists()


def test_prune_noop_when_not_enough_versions(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    reg.register("ltr", _artifact(tmp_path))
    assert reg.prune("ltr", keep_last=3) == 0


def test_prune_tolerates_already_deleted_file(tmp_path: Path):
    reg = ModelRegistry(tmp_path)
    for i in range(4):
        reg.register("ltr", _artifact(tmp_path, f"m{i}.joblib"))
    # Файл первой версии уже пропал - prune не должен падать.
    first = reg.list_versions("ltr")[0]["path"]
    Path(first).unlink()
    assert reg.prune("ltr", keep_last=1) == 3


# ---------- register_* обёртки ----------


def test_register_helpers_return_err_when_artifact_absent(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models", raising=False)
    monkeypatch.setattr(config, "EMBEDDINGS_CACHE_DIR", tmp_path / "emb", raising=False)
    monkeypatch.setattr(config, "VACANCY_CLUSTERS_CACHE_DIR", tmp_path / "cl", raising=False)

    reg = ModelRegistry(tmp_path / "reg")
    assert reg.register_ltr().is_err()
    assert reg.register_embeddings("middle").is_err()
    assert reg.register_clusters("middle").is_err()


def test_register_embeddings_ok(tmp_path, monkeypatch):
    import src.config as config

    cache = tmp_path / "emb"
    cache.mkdir()
    (cache / "market_embeddings_middle.joblib").write_bytes(b"x")
    monkeypatch.setattr(config, "EMBEDDINGS_CACHE_DIR", cache, raising=False)

    reg = ModelRegistry(tmp_path / "reg")
    res = reg.register_embeddings("middle")
    assert not res.is_err()
    assert reg.latest("embeddings")["tags"] == {"level": "middle"}


def test_register_clusters_ok(tmp_path, monkeypatch):
    import src.config as config

    cache = tmp_path / "cl"
    cache.mkdir()
    (cache / "vacancy_clusters_senior.pkl").write_bytes(b"x")
    monkeypatch.setattr(config, "VACANCY_CLUSTERS_CACHE_DIR", cache, raising=False)

    reg = ModelRegistry(tmp_path / "reg")
    res = reg.register_clusters("senior")
    assert not res.is_err()
    assert reg.latest("clusters")["tags"] == {"level": "senior"}


def test_registry_creates_directory(tmp_path: Path):
    target = tmp_path / "nested" / "registry"
    reg = ModelRegistry(target)
    assert target.exists()
    # Файл индекса появляется после первой записи.
    assert reg.list_versions("ltr") == []
    reg.register("ltr", _artifact(tmp_path, "a.joblib"))
    assert json.loads((target / "registry_index.json").read_text(encoding="utf-8"))["ltr"]
