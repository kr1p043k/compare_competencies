"""Научная вкладка: рыночные навыки получают вес из корпуса публикаций."""

from __future__ import annotations

import numpy as np
import pytest

from src.analyzers import academic_gap as ag


class _FakeComparator:
    """Все навыки одинаково близки к теме - проверяем только веса, не ранжирование."""

    def embed_skills(self, skills):
        return np.ones((len(skills), 4), dtype=float)


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.setattr(ag, "_load_it_skills", lambda: ["python", "sql", "docker"])
    monkeypatch.setattr(ag.AcademicGapAnalyzer, "_get_comparator", lambda self: _FakeComparator())
    monkeypatch.setattr(ag.AcademicGapAnalyzer, "_get_taxonomy_cats", lambda self: None)
    return ag.AcademicGapAnalyzer()


def test_market_stats_attaches_science_weights(monkeypatch, analyzer):
    monkeypatch.setattr(ag, "_load_science_weights", lambda: {"python": 0.8, "sql": 0.25})
    stats = analyzer._market_stats(["нейросети"])
    assert stats["market_weighting"] == "science"
    weights = {item["skill"]: item["weight"] for item in stats["top_market"]}
    assert weights == {"python": 0.8, "sql": 0.25, "docker": 0.0}


def test_market_stats_without_artifact_keeps_old_shape(monkeypatch, analyzer):
    monkeypatch.setattr(ag, "_load_science_weights", lambda: {})
    stats = analyzer._market_stats(["нейросети"])
    assert stats["market_weighting"] == "none"
    assert all(item["weight"] == 0.0 for item in stats["top_market"])


def test_market_stats_without_market_returns_none_weighting(monkeypatch, analyzer):
    monkeypatch.setattr(ag, "_load_it_skills", lambda: [])
    monkeypatch.setattr(ag, "_load_science_weights", lambda: {"python": 0.8})
    stats = analyzer._market_stats(["нейросети"])
    assert stats["market_weighting"] == "none"
    assert stats["top_market"] == []


def test_science_weights_loader_is_fail_soft(monkeypatch):
    """Битый артефакт не должен ломать вкладку: пустой словарь, без исключения."""
    def boom():
        raise RuntimeError("bad artifact")

    monkeypatch.setattr("src.analyzers.skills.science_weights.default_mix", boom)
    assert ag._load_science_weights() == {}
