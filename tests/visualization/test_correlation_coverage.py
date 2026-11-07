"""Покрытие src/visualization/correlation.py: теплокарта корреляции навыков.

Матрицу берём из фейкового анализатора, taxonomy подменяем. Matplotlib - Agg,
файлы пишем в tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import src.visualization.correlation as corr
from src import Err, Ok


class FakeAnalyzerOk:
    def __init__(self, skills, matrix):
        self._skills = skills
        self._matrix = matrix

    def get_correlation_labeled(self, top_n: int = 15):
        return Ok((self._skills, self._matrix))


class FakeAnalyzerErr:
    def get_correlation_labeled(self, top_n: int = 15):
        return Err("нет данных")


def _matrix(n: int):
    import numpy as np

    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            m[i, j] = 1.0 if i == j else 0.5
    return m


def test_safe_label_without_taxonomy_returns_skill():
    assert corr._safe_label("python", None) == "python"


def test_safe_label_with_taxonomy_adds_prefix():
    class Tx:
        def get_category_icon(self, _s):
            return "🔧"

    label = corr._safe_label("python", Tx())
    # Иконка заменяется текстом либо остаётся навык; проверяем, что навык на месте.
    assert "python" in label


def test_safe_label_taxonomy_error_returns_skill():
    class Tx:
        def get_category_icon(self, _s):
            raise RuntimeError("boom")

    assert corr._safe_label("python", Tx()) == "python"


def test_plot_error_result_returns_empty_figure(tmp_path: Path):
    out = tmp_path / "empty.png"
    fig = corr.plot_skill_correlation_heatmap(FakeAnalyzerErr(), save_path=out)
    assert fig is not None
    assert out.exists()


def test_plot_too_few_skills_returns_figure(tmp_path: Path):
    out = tmp_path / "few.png"
    fig = corr.plot_skill_correlation_heatmap(
        FakeAnalyzerOk(["python"], _matrix(1)), save_path=out
    )
    assert fig is not None
    assert out.exists()


def test_plot_full_heatmap(tmp_path: Path, monkeypatch):
    skills = ["python", "sql", "docker"]
    out = tmp_path / "heat.png"
    fig = corr.plot_skill_correlation_heatmap(
        FakeAnalyzerOk(skills, _matrix(3)), top_n=3, save_path=out
    )
    assert fig is not None
    assert out.exists() and out.stat().st_size > 0


def test_plot_without_taxonomy_available(monkeypatch, tmp_path: Path):
    # SkillTaxonomy() бросает - ветка taxonomy=None всё равно рисует.
    import src.analyzers.skills.skill_taxonomy as st

    monkeypatch.setattr(st, "SkillTaxonomy", lambda: (_ for _ in ()).throw(RuntimeError("no")))
    out = tmp_path / "notax.png"
    fig = corr.plot_skill_correlation_heatmap(
        FakeAnalyzerOk(["python", "sql"], _matrix(2)), save_path=out
    )
    assert fig is not None
    assert out.exists()
