# tests/visualization/test_charts.py
import matplotlib
matplotlib.use("Agg")  # Без окна — важно для тестов на CI и Windows

import pytest
import pandas as pd

from src.visualization.charts import (
    plot_radar_chart,
    plot_heatmap,
    plot_coverage_comparison,
    plot_top_deficits,
    plot_skill_trends,
)


def test_plot_radar_chart(sample_student_skills, sample_market_top_skills):
    fig = plot_radar_chart(
        student_skills=sample_student_skills,
        market_top=sample_market_top_skills,
        student_name="Анна Иванова"
    )
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_plot_coverage_comparison(sample_results_dict):
    fig = plot_coverage_comparison(sample_results_dict)
    assert fig is not None


def test_plot_top_deficits():
    deficits = [
        {"skill": "FastAPI", "frequency": 87},
        {"skill": "Docker", "frequency": 76},
    ]
    fig = plot_top_deficits(deficits, student_name="Анна Иванова")
    assert fig is not None


def test_plot_heatmap():
    df = pd.DataFrame({
        "Python": [1, 0],
        "SQL": [1, 1],
        "Docker": [0, 1]
    }, index=["Анна", "Иван"])
    fig = plot_heatmap(df)
    assert fig is not None


def test_plot_skill_trends():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "skill": ["Python"] * 5,
        "frequency": [45, 52, 61, 58, 70]
    })
    fig = plot_skill_trends(df, skill="Python")
    assert fig is not None