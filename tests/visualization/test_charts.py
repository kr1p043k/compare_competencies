# tests/visualization/test_charts.py
import pytest
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import sys

# Переключаем backend matplotlib на неинтерактивный для тестов
plt.switch_backend('Agg')

from src.visualization.charts import (
    plot_coverage_comparison,
    plot_top_deficits,
    plot_radar_chart,
    plot_skill_trends,
    plot_heatmap,
    save_all_charts,
    run_notebook,
    show_context_info,
)

# --------------------------------------------------------------
# Фикстуры
# --------------------------------------------------------------

@pytest.fixture
def sample_deficits():
    return [
        {"skill": "python", "frequency": 150},
        {"skill": "sql", "frequency": 120},
        {"skill": "docker", "frequency": 90},
    ]

@pytest.fixture
def sample_market_top():
    return ["python", "react", "docker", "fastapi", "sql"]

# --------------------------------------------------------------
# Тесты plot_coverage_comparison
# --------------------------------------------------------------

def test_plot_coverage_comparison_empty():
    fig = plot_coverage_comparison({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_coverage_comparison_with_weighted(tmp_path):
    results = {
        "student1": {
            "coverage_percent": 65,
            "coverage_details": {"covered_weight": 130, "total_weight": 200}
        },
        "student2": {
            "weighted_coverage_percent": 85
        }
    }
    save_path = tmp_path / "coverage.png"
    fig = plot_coverage_comparison(results, save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

# --------------------------------------------------------------
# Тесты plot_top_deficits
# --------------------------------------------------------------

def test_plot_top_deficits_empty():
    fig = plot_top_deficits([], "Student X")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_top_deficits_with_data(sample_deficits, tmp_path):
    save_path = tmp_path / "deficits.png"
    fig = plot_top_deficits(sample_deficits, "Student Y", save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

# --------------------------------------------------------------
# Тесты plot_radar_chart
# --------------------------------------------------------------

def test_plot_radar_chart_basic(sample_market_top, tmp_path):
    student_skills = ["python", "docker"]
    save_path = tmp_path / "radar.png"
    fig = plot_radar_chart(student_skills, sample_market_top, "Student Z", save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_radar_chart_no_overlap(sample_market_top):
    # студент не имеет ни одного навыка из market_top
    fig = plot_radar_chart([], sample_market_top, "Empty")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

# --------------------------------------------------------------
# Тесты plot_skill_trends
# --------------------------------------------------------------

def test_plot_skill_trends_with_data(tmp_path):
    df = pd.DataFrame({
        "skill": ["python"] * 3,
        "date": pd.date_range("2024-01-01", periods=3),
        "frequency": [10, 20, 15]
    })
    save_path = tmp_path / "trend.png"
    fig = plot_skill_trends(df, "python", save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_skill_trends_no_data():
    df = pd.DataFrame(columns=["skill", "date", "frequency"])
    fig = plot_skill_trends(df, "missing")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

# --------------------------------------------------------------
# Тесты plot_heatmap
# --------------------------------------------------------------

def test_plot_heatmap(tmp_path):
    matrix = pd.DataFrame(
        [[1, 0, 1], [0, 1, 0]],
        index=["student1", "student2"],
        columns=["python", "sql", "docker"]
    )
    save_path = tmp_path / "heatmap.png"
    fig = plot_heatmap(matrix, save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

# --------------------------------------------------------------
# Тесты save_all_charts
# --------------------------------------------------------------

@patch("src.visualization.charts.plot_coverage_comparison")
@patch("src.visualization.charts.plot_top_deficits")
@patch("src.visualization.charts.plot_radar_chart")
def test_save_all_charts(mock_radar, mock_deficits, mock_coverage, tmp_path):
    # Подготавливаем mock данные для skill_weights.json
    skill_weights_path = tmp_path / "data/processed/skill_weights.json"
    skill_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(skill_weights_path, "w", encoding="utf-8") as f:
        json.dump({"python": 100, "sql": 80, "frontend английский": 10}, f)

    # Мокаем config.DATA_PROCESSED_DIR
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = skill_weights_path.parent

        results = {
            "base": {
                "coverage_percent": 60,
                "high_demand_gaps": [{"skill": "sql", "frequency": 50}],
                "covered_skills": ["python"],
            }
        }
        output_dir = tmp_path / "output"
        save_all_charts(results, output_dir)

        # Проверяем, что функции были вызваны
        mock_coverage.assert_called_once()
        mock_deficits.assert_called_once()
        mock_radar.assert_called_once()

        # Проверяем создание директорий
        assert (output_dir / "base").exists()

# --------------------------------------------------------------
# Тесты run_notebook
# --------------------------------------------------------------

@patch("subprocess.run")
def test_run_notebook_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    # Создаём временный ноутбук (мокаем существование файла)
    with patch("pathlib.Path.exists", return_value=True):
        result = run_notebook("test_notebook.ipynb")
        assert result is True

@patch("subprocess.run")
def test_run_notebook_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="Error")
    with patch("pathlib.Path.exists", return_value=True):
        result = run_notebook("fail.ipynb")
        assert result is False

def test_run_notebook_file_missing():
    with patch("pathlib.Path.exists", return_value=False):
        result = run_notebook("missing.ipynb")
        assert result is False

# --------------------------------------------------------------
# Тесты show_context_info
# --------------------------------------------------------------

@patch("builtins.print")
@patch("src.visualization.charts.config")
def test_show_context_info_all_missing(mock_config, mock_print, tmp_path):
    # Настраиваем пути так, чтобы файлы отсутствовали
    mock_config.DATA_PROCESSED_DIR = tmp_path / "processed"
    mock_config.COMPETENCY_MAPPING_FILE = tmp_path / "mapping.json"
    mock_config.STUDENTS_DIR = tmp_path / "students"
    mock_config.DATA_DIR = tmp_path / "data"

    show_context_info()

    # Проверяем, что вывод содержит ожидаемые сообщения
    printed = "\n".join(str(call) for call in mock_print.call_args_list)
    assert "не найден" in printed

@patch("builtins.print")
@patch("src.visualization.charts.config")
def test_show_context_info_with_data(mock_config, mock_print, tmp_path):
    # Создаём тестовые файлы
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    market_file = processed_dir / "competency_frequency.json"
    market_file.write_text(json.dumps({"python": 100, "sql": 80}), encoding="utf-8")
    mock_config.DATA_PROCESSED_DIR = processed_dir

    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"comp1": ["python"]}), encoding="utf-8")
    mock_config.COMPETENCY_MAPPING_FILE = mapping_file

    students_dir = tmp_path / "students"
    students_dir.mkdir()
    student_file = students_dir / "test_competency.json"
    student_file.write_text(json.dumps({"навыки": ["python", "sql"]}), encoding="utf-8")
    mock_config.STUDENTS_DIR = students_dir

    result_dir = tmp_path / "data/result/base"
    result_dir.mkdir(parents=True)
    (result_dir / "comparison_report_base.json").touch()
    mock_config.DATA_DIR = tmp_path / "data"

    show_context_info()

    printed = "\n".join(str(call) for call in mock_print.call_args_list)
    assert "python" in printed
    assert "test" in printed