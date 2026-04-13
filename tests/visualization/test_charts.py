# tests/visualization/test_charts.py
import pytest
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call


plt.switch_backend('Agg')

from src.visualization.charts import (
    plot_coverage_comparison,
    plot_ml_importance,
    plot_skill_comparison_radar,
    plot_weight_distribution,
    save_all_charts,
    run_notebook,
    show_context_info,
    load_skill_weights,
    load_hybrid_weights,
    load_ml_recommendations,
)


@pytest.fixture
def sample_market_top():
    return ["python", "react", "docker", "fastapi", "sql"]


# --------------------------------------------------------------
# Тесты load_skill_weights
# --------------------------------------------------------------
def test_load_skill_weights_exists(tmp_path):
    weights_path = tmp_path / "skill_weights.json"
    weights_path.write_text(json.dumps({"python": 1.0}), encoding='utf-8')
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = tmp_path
        result = load_skill_weights()
        assert result == {"python": 1.0}


def test_load_skill_weights_missing():
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = Path("/nonexistent")
        result = load_skill_weights()
        assert result == {}


def test_load_hybrid_weights_exists(tmp_path):
    weights_path = tmp_path / "hybrid_weights.json"
    weights_path.write_text(json.dumps({"docker": 0.9}), encoding='utf-8')
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = tmp_path
        result = load_hybrid_weights()
        assert result == {"docker": 0.9}


def test_load_hybrid_weights_missing():
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = Path("/nonexistent")
        result = load_hybrid_weights()
        assert result == {}


def test_load_ml_recommendations_exists(tmp_path):
    result_dir = tmp_path / "result" / "base"
    result_dir.mkdir(parents=True)
    rec_file = result_dir / "ml_recommendations_base.json"
    rec_file.write_text(json.dumps({"recommendations": [{"skill": "k8s", "score": 78.0, "explanation": "..."}]}))

    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_DIR = tmp_path
        result = load_ml_recommendations("base")
        assert len(result) == 1
        assert result[0][0] == "k8s"


def test_load_ml_recommendations_missing():
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_DIR = Path("/nonexistent")
        result = load_ml_recommendations("base")
        assert result == []


def test_load_ml_recommendations_corrupted(tmp_path):
    result_dir = tmp_path / "result" / "base"
    result_dir.mkdir(parents=True)
    rec_file = result_dir / "ml_recommendations_base.json"
    rec_file.write_text("{not valid json", encoding='utf-8')

    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_DIR = tmp_path
        result = load_ml_recommendations("base")
        assert result == []


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


def test_plot_coverage_comparison_with_readiness(tmp_path):
    results = {
        "base": {"coverage_percent": 50, "readiness_score": 65}
    }
    save_path = tmp_path / "coverage.png"
    fig = plot_coverage_comparison(results, save_path=save_path)
    assert save_path.exists()
    plt.close(fig)


# --------------------------------------------------------------
# Тесты plot_ml_importance
# --------------------------------------------------------------
@patch("src.visualization.charts.load_ml_recommendations")
def test_plot_ml_importance_empty(mock_load, tmp_path):
    mock_load.return_value = []
    save_path = tmp_path / "ml_importance.png"
    fig = plot_ml_importance("base", save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert not save_path.exists()
    plt.close(fig)


@patch("src.visualization.charts.load_ml_recommendations")
def test_plot_ml_importance_with_data(mock_load, tmp_path):
    mock_load.return_value = [("docker", 85.5, "explanation")]
    save_path = tmp_path / "ml_importance.png"
    fig = plot_ml_importance("base", save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# --------------------------------------------------------------
# Тесты plot_skill_comparison_radar
# --------------------------------------------------------------
def test_plot_skill_comparison_radar_basic(sample_market_top, tmp_path):
    student_skills = ["python", "docker"]
    save_path = tmp_path / "radar.png"
    fig = plot_skill_comparison_radar(student_skills, sample_market_top, "Student Z", save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_skill_comparison_radar_no_overlap(sample_market_top):
    fig = plot_skill_comparison_radar([], sample_market_top, "Empty")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_skill_comparison_radar_all_overlap(sample_market_top):
    fig = plot_skill_comparison_radar(sample_market_top, sample_market_top, "Full")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# --------------------------------------------------------------
# Тесты plot_weight_distribution
# --------------------------------------------------------------
def test_plot_weight_distribution_empty(tmp_path):
    save_path = tmp_path / "weight_dist.png"
    fig = plot_weight_distribution({}, save_path=save_path)
    assert not save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_weight_distribution_with_data(tmp_path):
    weights = {"python": 1.0, "sql": 0.8, "docker": 0.5}
    save_path = tmp_path / "weight_dist.png"
    fig = plot_weight_distribution(weights, save_path=save_path)
    assert save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# --------------------------------------------------------------
# Тесты save_all_charts
# --------------------------------------------------------------
@patch("src.visualization.charts.plot_coverage_comparison")
@patch("src.visualization.charts.plot_skill_comparison_radar")
@patch("src.visualization.charts.plot_ml_importance")
def test_save_all_charts(mock_ml, mock_radar, mock_coverage, tmp_path):
    skill_weights_path = tmp_path / "data/processed/skill_weights.json"
    skill_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(skill_weights_path, "w", encoding="utf-8") as f:
        json.dump({"python": 100, "sql": 80, "frontend английский": 10}, f)

    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = skill_weights_path.parent

        results = {
            "base": {
                "coverage_percent": 60,
                "covered_skills": ["python"],
                "high_demand_gaps": [{"skill": "sql", "frequency": 50}],
            }
        }
        output_dir = tmp_path / "output"
        save_all_charts(results, output_dir, use_ml=True)

        mock_coverage.assert_called_once()
        mock_radar.assert_called_once()
        mock_ml.assert_called_once()
        assert (output_dir / "base").exists()


@patch("src.visualization.charts.plot_coverage_comparison")
def test_save_all_charts_empty_results(mock_coverage, tmp_path):
    output_dir = tmp_path / "output"
    save_all_charts({}, output_dir)
    mock_coverage.assert_called_once()


@patch("src.visualization.charts.plot_coverage_comparison")
def test_save_all_charts_no_skill_weights(mock_coverage, tmp_path):
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = tmp_path / "nonexistent"
        results = {"base": {"coverage_percent": 60}}
        output_dir = tmp_path / "output"
        save_all_charts(results, output_dir, use_ml=False)
        mock_coverage.assert_called_once()
        # Радар не должен вызываться, так как нет top_skills
        assert not (output_dir / "base" / "radar_base.png").exists()


# --------------------------------------------------------------
# Тесты run_notebook
# --------------------------------------------------------------
@patch("subprocess.run")
def test_run_notebook_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
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
    mock_config.DATA_PROCESSED_DIR = tmp_path / "processed"
    mock_config.COMPETENCY_MAPPING_FILE = tmp_path / "mapping.json"
    mock_config.STUDENTS_DIR = tmp_path / "students"
    mock_config.DATA_DIR = tmp_path / "data"

    show_context_info()

    printed = "\n".join(str(call) for call in mock_print.call_args_list)
    assert "не найден" in printed


@patch("builtins.print")
@patch("src.visualization.charts.config")
def test_show_context_info_with_data(mock_config, mock_print, tmp_path):
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


def test_show_context_info_with_corrupted_market_file(tmp_path, capsys):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    market_file = processed_dir / "competency_frequency.json"
    market_file.write_text("{not valid", encoding='utf-8')

    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = processed_dir
        mock_config.COMPETENCY_MAPPING_FILE = tmp_path / "mapping.json"
        mock_config.STUDENTS_DIR = tmp_path / "students"
        mock_config.DATA_DIR = tmp_path / "data"

        show_context_info()

    captured = capsys.readouterr().out
    assert "Не удалось загрузить" in captured

# Дополнительные тесты для достижения 100% покрытия

def test_plot_weight_distribution_single_element(tmp_path):
    """Покрытие случая с одним элементом (max == min)."""
    weights = {"python": 1.0}
    save_path = tmp_path / "weight_dist.png"
    fig = plot_weight_distribution(weights, save_path=save_path)
    assert save_path.exists()
    plt.close(fig)


def test_save_all_charts_empty_results_no_ml(tmp_path):
    """Покрытие случая с пустыми результатами и use_ml=False."""
    output_dir = tmp_path / "output"
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = tmp_path / "processed"
        mock_config.DATA_DIR = tmp_path
        save_all_charts({}, output_dir, use_ml=False)
    # Файл создаётся, т.к. plot_coverage_comparison вызывается с save_path
    assert (output_dir / "coverage_comparison.png").exists()


def test_plot_skill_comparison_radar_single_skill(tmp_path):
    """Радар с одним навыком в market_top."""
    student_skills = ["python"]
    market_top = ["python"]
    save_path = tmp_path / "radar_single.png"
    fig = plot_skill_comparison_radar(student_skills, market_top, "Test", save_path=save_path)
    assert save_path.exists()
    plt.close(fig)


def test_plot_ml_importance_no_save_path():
    """Покрытие ветки без сохранения файла."""
    with patch("src.visualization.charts.load_ml_recommendations") as mock_load:
        mock_load.return_value = [("docker", 85.5, "explanation")]
        fig = plot_ml_importance("base", save_path=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_plot_coverage_comparison_with_readiness_only(tmp_path):
    """Покрытие случая, когда есть только readiness_score."""
    results = {"base": {"readiness_score": 75}}
    save_path = tmp_path / "coverage.png"
    fig = plot_coverage_comparison(results, save_path=save_path)
    assert save_path.exists()
    plt.close(fig)


def test_load_skill_weights_corrupted(tmp_path):
    """Повреждённый JSON при загрузке skill_weights."""
    weights_path = tmp_path / "skill_weights.json"
    weights_path.write_text("{not valid", encoding='utf-8')
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = tmp_path
        result = load_skill_weights()
        assert result == {}


def test_load_hybrid_weights_corrupted(tmp_path):
    """Повреждённый JSON при загрузке hybrid_weights."""
    weights_path = tmp_path / "hybrid_weights.json"
    weights_path.write_text("{not valid", encoding='utf-8')
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = tmp_path
        result = load_hybrid_weights()
        assert result == {}


def test_plot_coverage_comparison_no_data():
    """Покрытие случая с пустым DataFrame."""
    results = {"base": {}}
    fig = plot_coverage_comparison(results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_weight_distribution_no_save():
    weights = {"python": 1.0}
    fig = plot_weight_distribution(weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
def test_plot_skill_comparison_radar_empty_skills(sample_market_top):
    fig = plot_skill_comparison_radar([], sample_market_top, "Empty")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
def test_load_skill_weights_file_not_found():
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = Path("/nonexistent")
        assert load_skill_weights() == {}
        
def test_load_hybrid_weights_file_not_found():
    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = Path("/nonexistent")
        assert load_hybrid_weights() == {}

def test_load_skill_weights_io_error(tmp_path):
    """Имитация ошибки ввода-вывода при чтении skill_weights."""
    weights_path = tmp_path / "skill_weights.json"
    weights_path.write_text('{"python": 1.0}', encoding='utf-8')
    with patch("builtins.open", side_effect=IOError("Read error")):
        with patch("src.visualization.charts.config") as mock_config:
            mock_config.DATA_PROCESSED_DIR = tmp_path
            result = load_skill_weights()
            assert result == {}


def test_load_hybrid_weights_io_error(tmp_path):
    """Имитация ошибки ввода-вывода при чтении hybrid_weights."""
    weights_path = tmp_path / "hybrid_weights.json"
    weights_path.write_text('{"docker": 0.9}', encoding='utf-8')
    with patch("builtins.open", side_effect=IOError("Read error")):
        with patch("src.visualization.charts.config") as mock_config:
            mock_config.DATA_PROCESSED_DIR = tmp_path
            result = load_hybrid_weights()
            assert result == {}


def test_plot_coverage_comparison_save_empty(tmp_path):
    """Сохранение графика при пустых данных."""
    results = {"base": {}}
    save_path = tmp_path / "empty.png"
    fig = plot_coverage_comparison(results, save_path=save_path)
    assert save_path.exists()
    plt.close(fig)


def test_plot_skill_comparison_radar_save_no_path():
    """Радар без сохранения файла."""
    student_skills = ["python"]
    market_top = ["python", "docker"]
    fig = plot_skill_comparison_radar(student_skills, market_top, "Test")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_weight_distribution_save_empty(tmp_path):
    """Попытка сохранить распределение при пустых весах."""
    save_path = tmp_path / "empty_weight.png"
    fig = plot_weight_distribution({}, save_path=save_path)
    # Файл не должен создаться, но фигура возвращается
    assert not save_path.exists()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_all_charts_no_ml_with_skill_weights(tmp_path):
    """Интеграционный тест save_all_charts без ML, но с весами."""
    skill_weights_path = tmp_path / "processed" / "skill_weights.json"
    skill_weights_path.parent.mkdir(parents=True)
    skill_weights_path.write_text(json.dumps({"python": 100, "docker": 80}))

    results = {
        "base": {
            "coverage_percent": 60,
            "covered_skills": ["python"],
            "high_demand_gaps": [{"skill": "docker", "frequency": 5}]
        }
    }

    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = skill_weights_path.parent
        mock_config.DATA_DIR = tmp_path
        output_dir = tmp_path / "charts_output"
        save_all_charts(results, output_dir, use_ml=False)

    assert (output_dir / "coverage_comparison.png").exists()
    assert (output_dir / "base" / "radar_base.png").exists()
    assert (output_dir / "base" / "deficits_base.png").exists()
    # ML-график не должен создаваться
    assert not (output_dir / "base" / "ml_importance_base.png").exists()


def test_show_context_info_mapping_corrupted(tmp_path, capsys):
    """Повреждённый файл маппинга в show_context_info."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    market_file = processed_dir / "competency_frequency.json"
    market_file.write_text(json.dumps({"python": 100}))

    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text("{not valid", encoding='utf-8')

    with patch("src.visualization.charts.config") as mock_config:
        mock_config.DATA_PROCESSED_DIR = processed_dir
        mock_config.COMPETENCY_MAPPING_FILE = mapping_file
        mock_config.STUDENTS_DIR = tmp_path / "students"
        mock_config.DATA_DIR = tmp_path

        show_context_info()

    captured = capsys.readouterr().out
    assert "Не удалось загрузить" in captured