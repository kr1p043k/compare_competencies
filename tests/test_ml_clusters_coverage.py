"""Покрытие src/ml/clusters.py: подготовка вакансий и обучение кластеров.

Реальный sklearn/парсер не запускаем: подменяем VacancyParser, VacancyClusterer,
read_json и пути config, чтобы тесты были быстрыми и детерминированными.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import src.ml.clusters as clusters


class _Ok:
    def __init__(self, value: Any) -> None:
        self._value = value

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> Any:
        return self._value

    def unwrap_or(self, default: Any) -> Any:
        return self._value


class _Err:
    def is_err(self) -> bool:
        return True

    def unwrap(self) -> Any:
        raise RuntimeError("err")

    def unwrap_or(self, default: Any) -> Any:
        return default


class _FakeParser:
    """Возвращает навыки из описания детерминированно."""

    def __init__(self, skills: list[str] | None = None) -> None:
        self._skills = skills or []

    def extract_skills_from_description(self, _desc: str) -> Any:
        return _Ok(list(self._skills))


class _FakeClusterer:
    def __init__(self, fitted: bool = True, labels: list[int] | None = None) -> None:
        self.is_fitted = fitted
        self._labels = labels
        self.labels_: list[int] = labels if labels is not None else [0, 1, 0, 1]
        self.fitted_with: list = []

    def fit(self, data, level="all") -> Any:
        self.fitted_with.append((len(data), level))
        # Кластеры должны быть по одной метке на вакансию, иначе падает индексация.
        if self._labels is None:
            self.labels_ = [i % 2 for i in range(len(data))]
        return self


# ---------- prepare_vacancies_for_clustering ----------


def test_prepare_collects_key_skills_and_filters_empty(monkeypatch):
    monkeypatch.setattr(clusters, "VacancyParser", lambda: _FakeParser([]))
    raw = [
        {"name": "A", "key_skills": [{"name": "python"}, {"name": "sql"}]},
        {"name": "B", "key_skills": []},  # без навыков и описания -> выпадает
    ]
    out = clusters.prepare_vacancies_for_clustering(raw)
    assert len(out) == 1
    assert out[0]["skills"] == ["python", "sql"]
    assert out[0]["name"] == "A"


def test_prepare_merges_description_skills_and_dedups(monkeypatch):
    monkeypatch.setattr(clusters, "VacancyParser", lambda: _FakeParser(["sql", "docker"]))
    raw = [{
        "name": "A",
        "key_skills": [{"name": "python"}],
        "snippet": {"requirement": "python sql"},
    }]
    out = clusters.prepare_vacancies_for_clustering(raw)
    assert out[0]["skills"] == ["python", "sql", "docker"]


def test_prepare_maps_salary_area_employer(monkeypatch):
    monkeypatch.setattr(clusters, "VacancyParser", lambda: _FakeParser([]))
    raw = [{
        "name": "A",
        "key_skills": [{"name": "go"}],
        "experience": {"id": "between1And3"},
        "salary": {"from": 100, "to": 200, "currency": "RUR"},
        "area": {"name": "Москва"},
        "employer": {"name": "ООО"},
        "published_at": "2026-01-01",
        "alternate_url": "https://hh.ru/1",
    }]
    out = clusters.prepare_vacancies_for_clustering(raw)[0]
    assert out["experience"] == "between1And3"
    assert out["salary_from"] == 100
    assert out["salary_to"] == 200
    assert out["currency"] == "RUR"
    assert out["area"] == "Москва"
    assert out["employer"] == "ООО"


def test_prepare_tolerates_missing_optional_blocks(monkeypatch):
    monkeypatch.setattr(clusters, "VacancyParser", lambda: _FakeParser([]))
    raw = [{"name": "A", "key_skills": [{"name": "go"}]}]
    out = clusters.prepare_vacancies_for_clustering(raw)[0]
    assert out["salary_from"] is None
    assert out["area"] is None
    assert out["employer"] is None


# ---------- train_clusters ----------


def _patch_common(monkeypatch, tmp_path: Path, *, raw, clusterer=None, read_err=False,
                  level_value="all"):
    """Общая настройка: пути, read_json, кластеризатор, ExperienceLevel."""
    processed = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    processed.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    basic = raw_dir / "hh_vacancies_basic.json"
    basic.write_text(json.dumps(raw), encoding="utf-8")

    monkeypatch.setattr(clusters.config, "DATA_PROCESSED_DIR", processed, raising=False)
    monkeypatch.setattr(clusters.config, "DATA_RAW_DIR", raw_dir, raising=False)
    monkeypatch.setattr(clusters.config, "REPORTS_DIR", tmp_path / "reports", raising=False)
    monkeypatch.setattr(clusters.config, "REFERENCE_DIR", tmp_path / "ref", raising=False)

    if read_err:
        monkeypatch.setattr(clusters, "read_json", lambda _p: _Err())
    else:
        monkeypatch.setattr(clusters, "read_json", lambda _p: _Ok(raw))

    monkeypatch.setattr(clusters, "VacancyParser", lambda: _FakeParser([]))
    created: list[_FakeClusterer] = []

    def make_clusterer():
        c = clusterer or _FakeClusterer()
        created.append(c)
        return c

    monkeypatch.setattr(clusters, "VacancyClusterer", make_clusterer)

    # Один уровень, чтобы не гонять все ExperienceLevel (в enum их три).
    from src.models.enums import ExperienceLevel as _RealLevel

    level = _RealLevel.JUNIOR
    monkeypatch.setattr(clusters, "ExperienceLevel", [level], raising=False)
    # ExperienceLevel импортируется внутри функции из src.models.enums.
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.models.enums",
        SimpleNamespace(ExperienceLevel=[level]),
    )
    return created


def test_train_returns_false_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(clusters.config, "DATA_PROCESSED_DIR", tmp_path / "nope", raising=False)
    monkeypatch.setattr(clusters.config, "DATA_RAW_DIR", tmp_path / "nope2", raising=False)
    assert clusters.train_clusters() is False


def test_train_returns_false_on_read_error(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, raw=[], read_err=True)
    assert clusters.train_clusters() is False


def test_train_returns_false_on_empty_vacancies(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, raw=[])
    assert clusters.train_clusters() is False


def test_train_returns_false_when_clusterer_not_fitted(monkeypatch, tmp_path):
    raw = [{"name": "A", "key_skills": [{"name": "python"}]}]
    _patch_common(monkeypatch, tmp_path, raw=raw, clusterer=_FakeClusterer(fitted=False))
    assert clusters.train_clusters() is False


def test_train_success_writes_report(monkeypatch, tmp_path):
    raw = [{"name": "A", "key_skills": [{"name": "python"}]} for _ in range(12)]
    _patch_common(monkeypatch, tmp_path, raw=raw, clusterer=_FakeClusterer(labels=[0, 1]))
    ok = clusters.train_clusters(level="all", save_report=True, interpret=False)
    assert ok is True
    reports = list((tmp_path / "reports").glob("cluster_report_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["num_clusters"] == 2
    assert report["total_vacancies"] == 12


def test_train_handles_items_wrapper(monkeypatch, tmp_path):
    raw = {"items": [{"name": "A", "key_skills": [{"name": "go"}]} for _ in range(12)]}
    _patch_common(monkeypatch, tmp_path, raw=raw, clusterer=_FakeClusterer())
    assert clusters.train_clusters(save_report=False) is True


def test_train_uses_known_taxonomy_without_crash(monkeypatch, tmp_path):
    raw = [{"name": "A", "key_skills": [{"name": "python"}]} for _ in range(12)]
    created = _patch_common(monkeypatch, tmp_path, raw=raw)
    ref = tmp_path / "ref"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "it_skills.json").write_text(json.dumps(["python", "sql"]), encoding="utf-8")
    assert clusters.train_clusters(save_report=True) is True


def test_train_skips_level_with_too_few_vacancies(monkeypatch, tmp_path):
    raw = [{"name": "A", "key_skills": [{"name": "python"}]}]
    _patch_common(monkeypatch, tmp_path, raw=raw, clusterer=_FakeClusterer(), level_value="junior")
    assert clusters.train_clusters(save_report=False) is True
