# tests/visualization/test_visualization.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, ANY

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization._config import EMOJI_TO_TEXT
from src.visualization._utils import (
    load_skill_weights,
    load_hybrid_weights,
    load_ml_recommendations,
    load_profile_evaluation,
)
from src.visualization.clusters import plot_cluster_insights
from src.visualization.correlation import plot_skill_correlation_heatmap
from src.visualization.coverage import plot_coverage_comparison, plot_skills_heatmap
from src.visualization.importance import plot_ml_importance, plot_weight_distribution
from src.visualization.radar import plot_skill_comparison_radar
from src.visualization.orchestration import (
    save_all_charts,
    run_notebook,
    show_context_info,
)

# отключаем показ графиков во время тестов
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def close_plots():
    """Закрываем все фигуры после каждого теста."""
    yield
    plt.close("all")


@pytest.fixture
def sample_results():
    """Минимальная структура results для графиков."""
    return {
        "student1": {
            "market_coverage_score": 72.0,
            "skill_coverage": 65.0,
            "domain_coverage_score": 60.0,
            "readiness_score": 68.0,
            "market_skill_coverage": 45.5,
            "student_skills": ["python", "sql", "git"],
            "cluster_context": {
                "closest_clusters": [
                    {"id": 0, "name": "Backend", "similarity": 0.85},
                ],
                "skills": {"docker": 0.9, "python": 0.5},
            },
            "skill_metrics": {
                "docker": {"cluster_relevance": 0.8, "category": "missing"},
            },
            "domain_coverage": {"Backend": {"coverage": 0.5}},
            "top_recommendations": [("docker", 0.85)],
        }
    }


# ---------------------------------------------------------------------------
# Tests for _config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_emoji_to_text_not_empty(self):
        assert isinstance(EMOJI_TO_TEXT, dict)
        assert len(EMOJI_TO_TEXT) > 0

    def test_mpl_rcparams_set(self):
        # просто убедимся, что импорт _config выполнился и установил параметры
        assert plt.rcParams["figure.figsize"] == [14.0, 9.0]


# ---------------------------------------------------------------------------
# Tests for _utils
# ---------------------------------------------------------------------------
class TestLoadUtils:
    def test_load_skill_weights_success(self, tmp_path, monkeypatch):
        data = {"python": 0.9, "sql": 0.7}
        path = tmp_path / "skill_weights.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr("src.visualization._utils.config.DATA_PROCESSED_DIR", tmp_path)
        result = load_skill_weights()
        assert result == data

    def test_load_skill_weights_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.visualization._utils.config.DATA_PROCESSED_DIR", tmp_path)
        result = load_skill_weights()
        assert result == {}

    def test_load_skill_weights_invalid_json(self, tmp_path, monkeypatch):
        path = tmp_path / "skill_weights.json"
        path.write_text("{invalid")
        monkeypatch.setattr("src.visualization._utils.config.DATA_PROCESSED_DIR", tmp_path)
        result = load_skill_weights()
        assert result == {}

    def test_load_hybrid_weights_success(self, tmp_path, monkeypatch):
        data = {"docker": 0.8}
        path = tmp_path / "hybrid_weights.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr("src.visualization._utils.config.DATA_PROCESSED_DIR", tmp_path)
        result = load_hybrid_weights()
        assert result == data

    def test_load_hybrid_weights_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.visualization._utils.config.DATA_PROCESSED_DIR", tmp_path)
        result = load_hybrid_weights()
        assert result == {}

    def test_load_ml_recommendations_found(self, tmp_path, monkeypatch):
        data = {
            "recommendations": [
                {"skill": "docker", "importance_score": 85, "why_important": "..."},
                {"skill": "k8s", "importance": 45, "explanation": "..."},
            ]
        }
        profile = "test"
        result_dir = tmp_path / "result" / profile
        result_dir.mkdir(parents=True)
        file = result_dir / f"ltr_recommendations_{profile}.json"
        file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr("src.visualization._utils.config.DATA_DIR", tmp_path)
        recs = load_ml_recommendations(profile)
        assert len(recs) == 2
        assert recs[0][0] == "docker"
        assert recs[0][1] == 85
        assert recs[1][1] == 45

    def test_load_ml_recommendations_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.visualization._utils.config.DATA_DIR", tmp_path)
        recs = load_ml_recommendations("unknown")
        assert recs == []

    def test_load_ml_recommendations_second_file(self, tmp_path, monkeypatch):
        data = {"recommendations": [{"skill": "java", "importance_score": 90, "explanation": ""}]}
        profile = "x"
        result_dir = tmp_path / "result" / profile
        result_dir.mkdir(parents=True)
        file = result_dir / f"full_recommendations_{profile}.json"
        file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr("src.visualization._utils.config.DATA_DIR", tmp_path)
        recs = load_ml_recommendations(profile)
        assert len(recs) == 1
        assert recs[0][0] == "java"

    def test_load_profile_evaluation_success(self, tmp_path, monkeypatch):
        data = {
            "evaluations": {
                "stud1": {"readiness_score": 70}
            }
        }
        path = tmp_path / "profiles_comparison_summary.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr("src.visualization._utils.config.DATA_RESULT_DIR", tmp_path)
        result = load_profile_evaluation("stud1")
        assert result == {"readiness_score": 70}

    def test_load_profile_evaluation_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.visualization._utils.config.DATA_RESULT_DIR", tmp_path)
        result = load_profile_evaluation("nobody")
        assert result is None


# ---------------------------------------------------------------------------
# Tests for clusters.py
# ---------------------------------------------------------------------------
class TestClusterInsights:
    def test_plot_with_context(self, sample_results, tmp_path):
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            plot_cluster_insights(sample_results, tmp_path)
        # файл должен был создаться
        save_path = tmp_path / "student1" / "cluster_insights_student1.png"
        # проверяем, что savefig вызывался с этим путём (опционально)
        # главное — нет исключений

    def test_plot_no_context(self, sample_results, tmp_path):
        results = {"stud": {"student_skills": ["py"], "cluster_context": None}}
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_cluster_insights(results, tmp_path)
            mock_save.assert_not_called()

    def test_plot_cluster_insights_no_student_skills(self, tmp_path):
        results = {
            "stud": {
                "student_skills": [],
                "cluster_context": {
                    "closest_clusters": [{"id": 0, "name": "Backend", "similarity": 0.8}],
                    "skills": {"python": 0.9}
                }
            }
        }
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_cluster_insights(results, tmp_path)
            mock_save.assert_called_once()  # должно сохранить график с coverage=0


# ---------------------------------------------------------------------------
# Tests for correlation.py
# ---------------------------------------------------------------------------
class TestCorrelationHeatmap:
    def test_heatmap_with_skills(self, tmp_path):
        # мокаем корреляционный анализатор
        mock_analyzer = MagicMock()
        mock_analyzer.get_correlation_labeled.return_value = (
            ["python", "sql", "java"],
            np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]]),
        )
        save_path = tmp_path / "heatmap.png"
        with patch("matplotlib.pyplot.savefig") as mock_save, \
             patch("src.analyzers.skills.skill_taxonomy.SkillTaxonomy") as mock_tax:
            mock_tax.return_value.get_category_label.return_value = "devops"
            mock_tax.return_value.get_category_icon.return_value = "💻"
            fig = plot_skill_correlation_heatmap(
                mock_analyzer, top_n=5, save_path=save_path
            )
            assert isinstance(fig, plt.Figure)
            mock_save.assert_called_once()

    def test_heatmap_insufficient_data(self, tmp_path):
        mock_analyzer = MagicMock()
        mock_analyzer.get_correlation_labeled.return_value = (["a"], np.array([[1.0]]))
        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_skill_correlation_heatmap(mock_analyzer, save_path=tmp_path / "heat.png")
            assert isinstance(fig, plt.Figure)
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for coverage.py
# ---------------------------------------------------------------------------
class TestCoveragePlots:
    def test_coverage_comparison(self, sample_results, tmp_path):
        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_coverage_comparison(sample_results, save_path=tmp_path / "cov.png")
            assert isinstance(fig, plt.Figure)
            mock_save.assert_called_once()

    def test_skills_heatmap(self, sample_results, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.visualization.coverage.load_skill_weights",
            lambda: {"python": 0.9, "sql": 0.7, "docker": 0.5},
        )
        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_skills_heatmap(sample_results, save_path=tmp_path / "heat.png")
            assert isinstance(fig, plt.Figure)
            mock_save.assert_called_once()

    def test_skills_heatmap_no_weights(self, sample_results, tmp_path, monkeypatch):
        monkeypatch.setattr("src.visualization.coverage.load_skill_weights", lambda: {})
        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_skills_heatmap(sample_results, save_path=tmp_path / "heat.png")
            assert isinstance(fig, plt.Figure)
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for importance.py
# ---------------------------------------------------------------------------
class TestImportancePlots:
    def test_plot_ml_importance_empty(self, tmp_path):
        with patch("src.visualization.importance.load_ml_recommendations", return_value=[]):
            with patch("matplotlib.pyplot.savefig") as mock_save:
                fig = plot_ml_importance("profile", save_path=tmp_path / "ml.png")
                assert isinstance(fig, plt.Figure)
                mock_save.assert_called_once()

    def test_plot_ml_importance_with_data(self, tmp_path):
        recs = [("docker", 90, ""), ("k8s", 70, "")]
        with patch("src.visualization.importance.load_ml_recommendations", return_value=recs):
            with patch("matplotlib.pyplot.savefig") as mock_save:
                fig = plot_ml_importance("profile", save_path=tmp_path / "ml.png")
                assert isinstance(fig, plt.Figure)
                mock_save.assert_called_once()

    def test_plot_weight_distribution_empty(self, tmp_path):
        # При пустом словаре savefig не вызывается, а рисуется текст
        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_weight_distribution({}, save_path=tmp_path / "w.png")
            assert isinstance(fig, plt.Figure)
            mock_save.assert_not_called()   # <-- изменено с assert_called_once на assert_not_called

    def test_plot_weight_distribution_data(self, tmp_path):
        weights = {"python": 0.9, "sql": 0.7, "java": 0.5}
        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_weight_distribution(weights, save_path=tmp_path / "w.png")
            assert isinstance(fig, plt.Figure)
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for radar.py
# ---------------------------------------------------------------------------
class TestRadar:
    def test_radar(self, tmp_path):
        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_skill_comparison_radar(
                ["python", "sql"], ["python", "java", "docker"], "Test", save_path=tmp_path / "radar.png"
            )
            assert isinstance(fig, plt.Figure)
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for orchestration.py
# ---------------------------------------------------------------------------
class TestOrchestration:
    def test_save_all_charts(self, sample_results, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.visualization.orchestration.load_skill_weights",
            lambda: {"python": 0.9},
        )
        with patch("src.visualization.orchestration.plot_coverage_comparison") as mock_cov, \
             patch("src.visualization.orchestration.plot_skill_comparison_radar") as mock_radar, \
             patch("src.visualization.orchestration.plot_ml_importance") as mock_ml, \
             patch("src.visualization.orchestration.plot_weight_distribution") as mock_w, \
             patch("src.visualization.orchestration.plot_skill_correlation_heatmap") as mock_heat, \
             patch("src.visualization.orchestration.plot_skills_heatmap") as mock_sh, \
             patch("src.visualization.orchestration.plot_cluster_insights") as mock_cluster, \
             patch("src.visualization.orchestration.load_ml_recommendations", return_value=[]):
            save_all_charts(sample_results, tmp_path)
            mock_cov.assert_called()
            mock_radar.assert_called()
            mock_w.assert_called()
            mock_sh.assert_called()
            mock_cluster.assert_called()

    def test_run_notebook_found_and_success(self, tmp_path):
        # Создаем временную структуру, подменяем __file__
        import src.visualization.orchestration as orch
        original_file = orch.__file__
        try:
            base_dir = tmp_path / "compare_competencies"
            base_dir.mkdir()
            notebooks_dir = base_dir / "notebooks"
            notebooks_dir.mkdir()
            (notebooks_dir / "test_nb.ipynb").write_text("{}")
            orch.__file__ = str(base_dir / "src" / "visualization" / "orchestration.py")
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                assert run_notebook("test_nb.ipynb") is True
        finally:
            orch.__file__ = original_file

    def test_run_notebook_missing(self, tmp_path):
        import src.visualization.orchestration as orch
        original_file = orch.__file__
        try:
            orch.__file__ = str(tmp_path / "src" / "visualization" / "orchestration.py")
            result = run_notebook("missing.ipynb")
            assert result is False
        finally:
            orch.__file__ = original_file

    def test_run_notebook_failure(self, tmp_path):
        import src.visualization.orchestration as orch
        original_file = orch.__file__
        try:
            base_dir = tmp_path / "project"
            base_dir.mkdir()
            notebooks_dir = base_dir / "notebooks"
            notebooks_dir.mkdir()
            (notebooks_dir / "test.ipynb").write_text("{}")
            orch.__file__ = str(base_dir / "src" / "visualization" / "orchestration.py")
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stderr = "error"
                result = run_notebook("test.ipynb")
                assert result is False
        finally:
            orch.__file__ = original_file

    def test_show_context_info(self, capsys, monkeypatch, tmp_path):
        # Подготовим файлы
        freq_file = tmp_path / "competency_frequency.json"
        freq_file.write_text(json.dumps({"python": 10}))
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text(json.dumps({"comp1": "desc"}))
        students_dir = tmp_path / "students"
        students_dir.mkdir()
        (students_dir / "john_competency.json").write_text(json.dumps({"навыки": ["py"]}))
        results_dir = tmp_path / "result"
        results_dir.mkdir()
        (results_dir / "john").mkdir()
        (results_dir / "john" / "full_recommendations_john.json").write_text("{}")

        monkeypatch.setattr("src.visualization.orchestration.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.COMPETENCY_MAPPING_FILE", mapping_file)
        monkeypatch.setattr("src.visualization.orchestration.config.STUDENTS_DIR", students_dir)
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_DIR", tmp_path)

        show_context_info()
        captured = capsys.readouterr()
        assert "john" in captured.out
        assert "1" in captured.out  # количество профилей

class TestExtendedCoverage:
    """Тесты для покрытия оставшихся строк визуализации."""

    def test_load_skill_weights_exception(self, tmp_path, monkeypatch):
        """Строки 37-39: ошибка при загрузке JSON."""
        file = tmp_path / "skill_weights.json"
        file.write_text("{invalid")
        monkeypatch.setattr("src.visualization._utils.config.DATA_PROCESSED_DIR", tmp_path)
        result = load_skill_weights()
        assert result == {}

    def test_load_hybrid_weights_exception(self, tmp_path, monkeypatch):
        """Строки 62-63: ошибка загрузки гибридных весов."""
        file = tmp_path / "hybrid_weights.json"
        file.write_text("{corrupt")
        monkeypatch.setattr("src.visualization._utils.config.DATA_PROCESSED_DIR", tmp_path)
        result = load_hybrid_weights()
        assert result == {}

    def test_load_ml_recommendations_exception(self, tmp_path, monkeypatch):
        """Строки 80, 82-84: исключение при разборе JSON в load_ml_recommendations."""
        profile = "err"
        result_dir = tmp_path / "result" / profile
        result_dir.mkdir(parents=True)
        file = result_dir / "ltr_recommendations_err.json"
        file.write_text("{bad")
        monkeypatch.setattr("src.visualization._utils.config.DATA_DIR", tmp_path)
        recs = load_ml_recommendations(profile)
        assert recs == []

    def test_plot_cluster_insights_with_skills(self, sample_results, tmp_path):
        """Покрытие ветки с навыками студента (строки 25, 37, 47)."""
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            # student_skills и cluster_skills_map присутствуют
            plot_cluster_insights(sample_results, tmp_path)
        # Должен создать файл для student1
        save_path = tmp_path / "student1" / "cluster_insights_student1.png"
        assert save_path.exists()

    def test_plot_skill_correlation_heatmap_taxonomy_exception(self, tmp_path):
        """Строки 47-48: ошибка получения категории в _safe_label."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_correlation_labeled.return_value = (
            ["python", "java"],
            np.array([[1.0, 0.8], [0.8, 1.0]]),
        )
        # SkillTaxonomy выбросит исключение при создании
        with patch("src.analyzers.skills.skill_taxonomy.SkillTaxonomy",
                   side_effect=Exception("no taxonomy")), \
             patch("matplotlib.pyplot.savefig"):
            fig = plot_skill_correlation_heatmap(mock_analyzer, save_path=tmp_path / "heat.png")
            assert isinstance(fig, plt.Figure)

    def test_coverage_comparison_save(self, sample_results, tmp_path):
        """Убедимся, что график действительно сохраняется."""
        save_path = tmp_path / "cov.png"
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_coverage_comparison(sample_results, save_path=save_path)
            mock_save.assert_called_once_with(save_path)

    def test_radar_save(self, tmp_path):
        save_path = tmp_path / "radar.png"
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_skill_comparison_radar(["py"], ["py", "java"], "S", save_path=save_path)
            mock_save.assert_called_once_with(save_path)

    def test_save_all_charts_with_correlation(self, sample_results, tmp_path, monkeypatch):
        """Покрытие ветки с vacancies_skills_list (строки 30-82)."""
        monkeypatch.setattr(
            "src.visualization.orchestration.load_skill_weights",
            lambda: {"python": 0.9},
        )
        with patch("src.visualization.orchestration.plot_coverage_comparison"), \
             patch("src.visualization.orchestration.plot_skill_comparison_radar"), \
             patch("src.visualization.orchestration.plot_ml_importance"), \
             patch("src.visualization.orchestration.plot_weight_distribution"), \
             patch("src.visualization.orchestration.plot_skill_correlation_heatmap") as mock_corr, \
             patch("src.visualization.orchestration.plot_skills_heatmap"), \
             patch("src.visualization.orchestration.plot_cluster_insights"), \
             patch("src.visualization.orchestration.load_ml_recommendations", return_value=[]), \
             patch("src.visualization.orchestration.SkillCorrelationAnalyzer") as mock_analyzer:
            save_all_charts(sample_results, tmp_path, vacancies_skills_list=[["python"]])
            mock_corr.assert_called()

    def test_save_all_charts_correlation_exception(self, sample_results, tmp_path, monkeypatch):
        """Строки 77-82: исключение при построении heatmap."""
        monkeypatch.setattr(
            "src.visualization.orchestration.load_skill_weights",
            lambda: {"python": 0.9},
        )
        with patch("src.visualization.orchestration.plot_coverage_comparison"), \
             patch("src.visualization.orchestration.plot_skill_comparison_radar"), \
             patch("src.visualization.orchestration.plot_ml_importance"), \
             patch("src.visualization.orchestration.plot_weight_distribution"), \
             patch("src.visualization.orchestration.plot_skill_correlation_heatmap",
                   side_effect=Exception("fail")), \
             patch("src.visualization.orchestration.plot_skills_heatmap"), \
             patch("src.visualization.orchestration.plot_cluster_insights"), \
             patch("src.visualization.orchestration.load_ml_recommendations", return_value=[]):
            # Не должно упасть
            save_all_charts(sample_results, tmp_path, vacancies_skills_list=[["python"]])

    def test_run_notebook_exception(self, tmp_path):
        """Строки 155-160: исключение при запуске."""
        with patch("subprocess.run", side_effect=Exception("OS error")):
            assert run_notebook("test.ipynb") is False

    def test_show_context_info_no_files(self, capsys, monkeypatch, tmp_path):
        """Покрытие ветвей с отсутствующими файлами."""
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.COMPETENCY_MAPPING_FILE",
                            tmp_path / "nonexistent")
        monkeypatch.setattr("src.visualization.orchestration.config.STUDENTS_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_DIR", tmp_path)
        show_context_info()
        captured = capsys.readouterr()
        assert "⚠️" in captured.out

    def test_plot_cluster_insights_with_skills(self, sample_results, tmp_path):
        """Покрытие ветки с навыками студента (строки 25, 37, 47)."""
        with patch("matplotlib.pyplot.savefig") as mock_save, patch("matplotlib.pyplot.close"):
            plot_cluster_insights(sample_results, tmp_path)
        # Должен был вызвать savefig
        mock_save.assert_called_once()
        # Проверяем, что путь содержит правильное имя
        args, _ = mock_save.call_args
        assert "cluster_insights_student1.png" in str(args[0])

    def test_save_all_charts_with_correlation(self, sample_results, tmp_path, monkeypatch):
        """Покрытие ветки с vacancies_skills_list (строки 30-82)."""
        monkeypatch.setattr(
            "src.visualization.orchestration.load_skill_weights",
            lambda: {"python": 0.9},
        )
        with patch("src.visualization.orchestration.plot_coverage_comparison"), \
            patch("src.visualization.orchestration.plot_skill_comparison_radar"), \
            patch("src.visualization.orchestration.plot_ml_importance"), \
            patch("src.visualization.orchestration.plot_weight_distribution"), \
            patch("src.visualization.orchestration.plot_skill_correlation_heatmap") as mock_corr, \
            patch("src.visualization.orchestration.plot_skills_heatmap"), \
            patch("src.visualization.orchestration.plot_cluster_insights"), \
            patch("src.visualization.orchestration.load_ml_recommendations", return_value=[]), \
            patch("src.visualization.orchestration.SkillCorrelationAnalyzer", create=True) as mock_analyzer:
            save_all_charts(sample_results, tmp_path, vacancies_skills_list=[["python"]])
            mock_corr.assert_called()

    def test_save_all_charts_correlation_exception(self, sample_results, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.visualization.orchestration.load_skill_weights",
            lambda: {"python": 0.9},
        )
        with patch("src.visualization.orchestration.plot_coverage_comparison"), \
            patch("src.visualization.orchestration.plot_skill_comparison_radar"), \
            patch("src.visualization.orchestration.plot_ml_importance"), \
            patch("src.visualization.orchestration.plot_weight_distribution"), \
            patch("src.visualization.orchestration.plot_skill_correlation_heatmap",
                side_effect=Exception("fail")), \
            patch("src.visualization.orchestration.plot_skills_heatmap"), \
            patch("src.visualization.orchestration.plot_cluster_insights"), \
            patch("src.visualization.orchestration.load_ml_recommendations", return_value=[]):
            # Не должно упасть
            save_all_charts(sample_results, tmp_path, vacancies_skills_list=[["python"]])


    def test_save_all_charts_with_ml_recs_and_deficits(self, sample_results, tmp_path, monkeypatch):
        """Покрытие ветки с ML-рекомендациями и дефицитами (строки 55-66)."""
        monkeypatch.setattr(
            "src.visualization.orchestration.load_skill_weights",
            lambda: {"python": 0.9},
        )
        recs = [("docker", 90, ""), ("k8s", 70, ""), ("java", 30, "")]
        with patch("src.visualization.orchestration.load_ml_recommendations", return_value=recs), \
            patch("src.visualization.orchestration.plot_coverage_comparison"), \
            patch("src.visualization.orchestration.plot_skill_comparison_radar"), \
            patch("src.visualization.orchestration.plot_ml_importance"), \
            patch("src.visualization.orchestration.plot_weight_distribution"), \
            patch("src.visualization.orchestration.plot_skills_heatmap"), \
            patch("src.visualization.orchestration.plot_cluster_insights"), \
            patch("matplotlib.pyplot.savefig"), \
            patch("matplotlib.pyplot.close"):
            save_all_charts(sample_results, tmp_path)
            # Проверяем, что был построен график дефицитов: barh вызван
            # Можем проверить, что файл deficits сохранен
            prof_dir = tmp_path / "student1"
            assert (prof_dir / "deficits_student1.png").exists()

    def test_save_all_charts_no_radar_if_no_market_top(self, sample_results, tmp_path, monkeypatch):
        """Если market_top пуст, радар не должен вызываться."""
        monkeypatch.setattr("src.visualization.orchestration.load_skill_weights", lambda: {})
        with patch("src.visualization.orchestration.plot_skill_comparison_radar") as mock_radar, \
            patch("src.visualization.orchestration.plot_coverage_comparison"), \
            patch("src.visualization.orchestration.plot_ml_importance"), \
            patch("src.visualization.orchestration.plot_weight_distribution"), \
            patch("src.visualization.orchestration.plot_skills_heatmap"), \
            patch("src.visualization.orchestration.plot_cluster_insights"), \
            patch("src.visualization.orchestration.load_ml_recommendations", return_value=[]):
            save_all_charts(sample_results, tmp_path)
            mock_radar.assert_not_called()

    def test_save_all_charts_with_correlation_analyzer_error(self, sample_results, tmp_path, monkeypatch):
        """Ошибка создания SkillCorrelationAnalyzer не должна крашить."""
        monkeypatch.setattr("src.visualization.orchestration.load_skill_weights", lambda: {"python": 0.9})
        with patch("src.visualization.orchestration.SkillCorrelationAnalyzer", create=True,
                side_effect=Exception("import error")), \
            patch("src.visualization.orchestration.plot_coverage_comparison"), \
            patch("src.visualization.orchestration.plot_skill_comparison_radar"), \
            patch("src.visualization.orchestration.plot_ml_importance"), \
            patch("src.visualization.orchestration.plot_weight_distribution"), \
            patch("src.visualization.orchestration.plot_skills_heatmap"), \
            patch("src.visualization.orchestration.plot_cluster_insights"), \
            patch("src.visualization.orchestration.load_ml_recommendations", return_value=[]):
            save_all_charts(sample_results, tmp_path, vacancies_skills_list=[["python"]])
            # Не должно упасть

    def test_show_context_info_with_all_data(self, capsys, monkeypatch, tmp_path):
        # Создадим нужные файлы и директории
        freq_file = tmp_path / "competency_frequency.json"
        freq_file.write_text('{"python": 10}')
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text('{"C1": "desc"}')
        students_dir = tmp_path / "students"
        students_dir.mkdir()
        (students_dir / "john_competency.json").write_text('{"навыки": ["py"]}')
        results_dir = tmp_path / "result"
        results_dir.mkdir()
        (results_dir / "john").mkdir()
        (results_dir / "john" / "full_recommendations_john.json").write_text("{}")
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.COMPETENCY_MAPPING_FILE", mapping_file)
        monkeypatch.setattr("src.visualization.orchestration.config.STUDENTS_DIR", students_dir)
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_DIR", tmp_path)
        show_context_info()
        captured = capsys.readouterr()
        assert "Рыночные навыки (частота): 1" in captured.out
        assert "Компетенций в маппинге: 1" in captured.out
        assert "john" in captured.out
        assert "1" in captured.out  # количество студентов

    def test_show_context_info_market_file_missing(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.COMPETENCY_MAPPING_FILE", tmp_path / "nonexistent")
        monkeypatch.setattr("src.visualization.orchestration.config.STUDENTS_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_DIR", tmp_path)
        show_context_info()
        captured = capsys.readouterr()
        assert "⚠️" in captured.out

    def test_show_context_info_market_file_error(self, capsys, monkeypatch, tmp_path):
        freq_file = tmp_path / "competency_frequency.json"
        freq_file.write_text("{invalid")
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.COMPETENCY_MAPPING_FILE", tmp_path / "nonexistent")
        monkeypatch.setattr("src.visualization.orchestration.config.STUDENTS_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_DIR", tmp_path)
        show_context_info()
        captured = capsys.readouterr()
        assert "Не удалось загрузить рыночные навыки" in captured.out

    def test_show_context_info_mapping_error(self, capsys, monkeypatch, tmp_path):
        freq_file = tmp_path / "competency_frequency.json"
        freq_file.write_text(json.dumps({"py": 1}))
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text("{invalid")
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.COMPETENCY_MAPPING_FILE", mapping_file)
        monkeypatch.setattr("src.visualization.orchestration.config.STUDENTS_DIR", tmp_path)
        monkeypatch.setattr("src.visualization.orchestration.config.DATA_DIR", tmp_path)
        show_context_info()
        captured = capsys.readouterr()
        assert "Не удалось загрузить маппинг компетенций" in captured.out

    def test_load_ml_recommendations_full_fallback(self, tmp_path, monkeypatch):
        """Покрытие загрузки из full_recommendations_{profile}.json (строка 80)"""
        profile = "p"
        result_dir = tmp_path / "result" / profile
        result_dir.mkdir(parents=True)
        file = result_dir / f"full_recommendations_{profile}.json"
        file.write_text(json.dumps({
            "recommendations": [{"skill": "java", "importance": 80, "explanation": "x"}]
        }))
        monkeypatch.setattr("src.visualization._utils.config.DATA_DIR", tmp_path)
        recs = load_ml_recommendations(profile)
        assert len(recs) == 1
        assert recs[0][0] == "java"
        assert recs[0][1] == 80

    def test_load_ml_recommendations_both_files_corrupted(self, tmp_path, monkeypatch):
        """Покрытие строк 82-84: оба файла повреждены -> пустой список"""
        profile = "err"
        result_dir = tmp_path / "result" / profile
        result_dir.mkdir(parents=True)
        (result_dir / f"ltr_recommendations_{profile}.json").write_text("{bad")
        (result_dir / f"full_recommendations_{profile}.json").write_text("{bad")
        monkeypatch.setattr("src.visualization._utils.config.DATA_DIR", tmp_path)
        recs = load_ml_recommendations(profile)
        assert recs == []

    # Для correlation.py строка 22-23 (ошибка taxonomy):
    def test_correlation_safe_label_taxonomy_error(self):
        from src.visualization.correlation import _safe_label
        mock_tax = MagicMock()
        mock_tax.get_category_icon.side_effect = Exception("fail")
        label = _safe_label("docker", mock_tax)
        assert label == "docker"  # возвращается исходный skill

    def test_save_all_charts_with_deficits(self, sample_results, tmp_path, monkeypatch):
        """Покрытие блока deficits (строки 55-66 в orchestration)."""
        monkeypatch.setattr(
            "src.visualization.orchestration.load_skill_weights",
            lambda: {"python": 0.9},
        )
        # делаем, чтобы load_ml_recommendations вернула данные с высокими scores
        recs = [("docker", 90, ""), ("k8s", 70, ""), ("java", 40, "")]
        with patch("src.visualization.orchestration.load_ml_recommendations", return_value=recs), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"), \
             patch("src.visualization.orchestration.plot_coverage_comparison"), \
             patch("src.visualization.orchestration.plot_skill_comparison_radar"), \
             patch("src.visualization.orchestration.plot_ml_importance"), \
             patch("src.visualization.orchestration.plot_weight_distribution"), \
             patch("src.visualization.orchestration.plot_skill_correlation_heatmap"), \
             patch("src.visualization.orchestration.plot_skills_heatmap"), \
             patch("src.visualization.orchestration.plot_cluster_insights"):
            save_all_charts(sample_results, tmp_path, use_ml=True)
        # Проверим, что файл deficits создался
        deficit_file = tmp_path / "student1" / "deficits_student1.png"
        assert deficit_file.exists()
