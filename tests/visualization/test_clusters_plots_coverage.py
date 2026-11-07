"""Покрытие src/visualization/clusters.py: график ближайших кластеров.

Matplotlib работает в headless-режиме (Agg), файлы пишем в tmp_path.
Проверяем как отрисовку при валидном контексте, так и ветки пропуска.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import src.visualization.clusters as viz


def _pngs(root: Path) -> list[Path]:
    return list(root.rglob("*.png"))


def test_plot_skips_profile_without_context(tmp_path: Path):
    results = {"student_a": {"student_skills": ["python"]}}
    viz.plot_cluster_insights(results, tmp_path)
    assert _pngs(tmp_path) == []


def test_plot_skips_profile_with_empty_closest(tmp_path: Path):
    results = {"student_a": {"cluster_context": {"closest_clusters": []}}}
    viz.plot_cluster_insights(results, tmp_path)
    assert _pngs(tmp_path) == []


def test_plot_creates_png(tmp_path: Path):
    results = {
        "student_a": {
            "student_skills": ["python", "sql"],
            "cluster_context": {
                "closest_clusters": [
                    {"id": 1, "name": "Backend: python", "similarity": 0.91},
                    {"id": 2, "name": "Data", "similarity": 0.72},
                ],
                "skills": {"python": 1, "sql": 1, "docker": 1},
            },
        }
    }
    viz.plot_cluster_insights(results, tmp_path)
    out = tmp_path / "student_a" / "cluster_insights_student_a.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_handles_long_names_and_no_skills(tmp_path: Path):
    long_name = "Очень длинное название кластера которое точно превышает лимит знаков"
    results = {
        "student_b": {
            "cluster_context": {
                "closest_clusters": [{"id": 7, "name": long_name, "similarity": 0.5}],
                "skills": {},
            }
        }
    }
    viz.plot_cluster_insights(results, tmp_path)
    assert (tmp_path / "student_b" / "cluster_insights_student_b.png").exists()


def test_plot_name_without_explicit_name_uses_id(tmp_path: Path):
    results = {
        "student_c": {
            "student_skills": ["go"],
            "cluster_context": {
                "closest_clusters": [{"id": 3, "similarity": 0.4}],
                "skills": {"go": 1},
            },
        }
    }
    viz.plot_cluster_insights(results, tmp_path)
    assert (tmp_path / "student_c" / "cluster_insights_student_c.png").exists()
