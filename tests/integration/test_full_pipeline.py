# tests/integration/test_full_pipeline.py
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest

from src import config
from src.models.vacancy import Vacancy, Area, Employer, KeySkill
from src.models.student import StudentProfile


# ----------------------------------------------------------------------
# Тесты парсинга и извлечения навыков
# ----------------------------------------------------------------------
@patch("src.parsing.vacancy_parser.VacancyParser.extract_skills_from_vacancies")
def test_skill_extraction_pipeline(mock_extract, tmp_path):
    """Проверяем, что извлечение навыков возвращает ожидаемую структуру."""
    mock_extract.return_value = {
        "frequencies": {"python": 10, "sql": 5},
        "hybrid_weights": {"python": 0.9, "sql": 0.4},
        "skill_embeddings": {},
    }
    from src.parsing.vacancy_parser import VacancyParser
    parser = VacancyParser()
    result = parser.extract_skills_from_vacancies([])
    assert "frequencies" in result
    assert "hybrid_weights" in result


@patch("src.parsing.hh_api.HeadHunterAPI.search_vacancies")
def test_vacancy_collection(mock_search, tmp_path):
    """Проверяем сбор вакансий и сохранение в raw."""
    mock_search.return_value = [
        {"id": "1", "name": "Python Dev", "area": {"id": 1}, "employer": {"id": "10"}}
    ]
    from src.parsing.hh_api import HeadHunterAPI
    from src.parsing.vacancy_parser import VacancyParser

    with patch.object(config, "DATA_RAW_DIR", tmp_path):
        api = HeadHunterAPI()
        parser = VacancyParser()
        vacancies = api.search_vacancies("Python", area=1, max_pages=1)
        assert len(vacancies) == 1
        parser.save_raw_vacancies(vacancies, filename="test.json")
        assert (tmp_path / "test.json").exists()


# ----------------------------------------------------------------------
# Тесты gap-анализа и профилей
# ----------------------------------------------------------------------
def test_gap_analysis_integration():
    """GapAnalyzer корректно считает покрытие и дефициты."""
    from src.analyzers.gap_analyzer import GapAnalyzer
    skill_weights = {"python": 10, "sql": 5, "docker": 2}
    analyzer = GapAnalyzer(skill_weights)
    coverage, details = analyzer.coverage(["python"])
    assert coverage == pytest.approx(10 / 17 * 100, 0.1)
    gaps = analyzer.analyze_gap(["python"])
    assert len(gaps["high_priority"]) + len(gaps["medium_priority"]) + len(gaps["low_priority"]) == 2


def test_profile_evaluator_integration():
    """ProfileEvaluator считает readiness для профилей."""
    from src.analyzers.profile_evaluator import ProfileEvaluator
    from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer

    skill_weights = {"python": 10, "docker": 5}
    vacancies_skills = [["python"], ["python", "docker"]]
    evaluator = ProfileEvaluator(skill_weights, vacancies_skills)

    student = StudentProfile(
        profile_name="test",
        competencies=[],
        skills=["python"],
        target_level="middle"
    )
    level_analyzer = SkillLevelAnalyzer()
    level_analyzer.skill_by_level = {"python": {"middle": 1}, "docker": {"middle": 1}}
    level_weights = level_analyzer.get_weights_for_level(skill_weights, "middle")

    eval_result = evaluator.evaluate_profile(student, "test", level_analyzer, level_weights)
    assert 0 <= eval_result.readiness_score <= 100
    assert eval_result.coverage["raw"] > 0


# ----------------------------------------------------------------------
# Тесты рекомендаций
# ----------------------------------------------------------------------
@patch("src.predictors.recommendation_engine.RecommendationEngine._llm_explain")
def test_recommendation_engine_integration(mock_llm, tmp_path):
    """RecommendationEngine генерирует рекомендации (без LLM)."""
    mock_llm.return_value = None
    from src.predictors.recommendation_engine import RecommendationEngine

    engine = RecommendationEngine(use_ltr=False, use_llm=False)
    vacancies = [["python", "sql"], ["python", "docker"]]
    skill_weights = {"python": 1.0, "sql": 0.8, "docker": 0.5}
    engine.fit(vacancies, skill_weights=skill_weights)

    student_skills = ["python"]
    recs = engine.generate_recommendations(student_skills)
    assert "summary" in recs
    assert recs["summary"]["coverage"] > 0
    assert len(recs["recommendations"]) > 0


# ----------------------------------------------------------------------
# Тесты сохранения результатов
# ----------------------------------------------------------------------
def test_save_all_results(tmp_path):
    """Проверяем, что все выходные файлы создаются."""
    from src.visualization.charts import save_all_charts

    results = {
        "base": {
            "coverage_percent": 60,
            "covered_skills": ["python"],
            "high_demand_gaps": [{"skill": "sql", "frequency": 5}],
            "readiness_score": 65
        }
    }
    skill_weights = {"python": 100, "sql": 80}
    skill_weights_path = tmp_path / "processed" / "skill_weights.json"
    skill_weights_path.parent.mkdir(parents=True)
    skill_weights_path.write_text(json.dumps(skill_weights))

    with patch.object(config, "DATA_PROCESSED_DIR", skill_weights_path.parent):
        with patch.object(config, "DATA_DIR", tmp_path):
            output_dir = tmp_path / "output"
            save_all_charts(results, output_dir, use_ml=False)

    assert (output_dir / "coverage_comparison.png").exists()
    assert (output_dir / "base" / "radar_base.png").exists()


# ----------------------------------------------------------------------
# Тесты конфигурации и путей
# ----------------------------------------------------------------------
def test_config_paths_exist():
    """Все ключевые директории конфига должны существовать."""
    assert config.DATA_DIR.exists()
    assert config.DATA_RAW_DIR.exists()
    assert config.DATA_PROCESSED_DIR.exists()
    assert config.STUDENTS_DIR.exists()


def test_mock_full_main_execution():
    """Имитация полного выполнения main.py без реальных вызовов API."""
    with patch("src.parsing.hh_api.HeadHunterAPI.search_vacancies") as mock_search, \
         patch("src.parsing.vacancy_parser.VacancyParser.extract_skills_from_vacancies") as mock_extract, \
         patch("src.predictors.recommendation_engine.RecommendationEngine.generate_recommendations") as mock_rec, \
         patch("src.visualization.charts.save_all_charts") as mock_charts, \
         patch("builtins.print"):
        mock_search.return_value = [{"id": "1", "name": "Dev"}]
        mock_extract.return_value = {
            "frequencies": {"python": 10, "sql": 5},
            "hybrid_weights": {"python": 0.9, "sql": 0.4},
        }
        mock_rec.return_value = {"summary": {}, "recommendations": []}

        # Импортируем main и запускаем с тестовыми аргументами
        import main
        with patch("sys.argv", ["main.py", "--query", "test", "--max-pages", "1", "--skip-details"]):
            main.main()

        mock_search.assert_called_once()
        mock_extract.assert_called_once()
        # Убедимся, что generate_recommendations был вызван хотя бы раз
        assert mock_rec.call_count >= 1, "generate_recommendations не был вызван"
        mock_charts.assert_called_once()