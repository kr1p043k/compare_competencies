import matplotlib
matplotlib.use("Agg")

import pytest
import pandas as pd

from src.visualization.charts import (
    plot_radar_chart, plot_heatmap, plot_coverage_comparison, plot_top_deficits
)

def test_plot_radar_chart(sample_student_skills, sample_market_top_skills):
    fig = plot_radar_chart(sample_student_skills, sample_market_top_skills, "base")
    assert fig is not None

def test_plot_coverage_comparison(sample_results_dict):
    fig = plot_coverage_comparison(sample_results_dict)
    assert fig is not None

def test_charts_generate_gap_chart(charts_module, gap_analyzer, sample_student):
    """Генерация графика gap-анализа"""
    import matplotlib.pyplot as plt
    
    # Простой тест без сохранения файла (проверяем, что метод не падает)
    fig = charts_module.generate_gap_chart(sample_student, gap_analyzer)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)  # очищаем память


def test_charts_generate_skill_radar(charts_module, sample_student):
    """Радаровая диаграмма навыков"""
    import matplotlib.pyplot as plt
    
    fig = charts_module.generate_skill_radar(sample_student)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_charts_generate_comparison_bar(charts_module, sample_student):
    """Столбчатая диаграмма сравнения студент ↔ рынок"""
    import matplotlib.pyplot as plt
    
    market_skills = {"python": 95, "react": 80, "docker": 60}
    fig = charts_module.generate_comparison_bar(sample_student, market_skills)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)