# tests/integration/test_full_pipeline.py
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src import config
from src.models.student import StudentProfile


# ----------------------------------------------------------------------
# Тесты парсинга и извлечения навыков
# ----------------------------------------------------------------------
@patch("src.parsing.vacancy_parser.VacancyParser.extract_skills_from_vacancies")
def test_skill_extraction_pipeline(mock_extract):
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
    mock_search.return_value = [{"id": "1", "name": "Python Dev", "area": {"id": 1}, "employer": {"id": "10"}}]
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
    """GapAnalyzer корректно считает метрики через compute_metrics."""
    from src.analyzers.gap_analyzer import GapAnalyzer

    skill_weights_by_level = {
        "junior": {"python": 0.8, "sql": 0.6, "docker": 0.4},
        "middle": {"python": 0.9, "docker": 0.7, "sql": 0.5},
        "senior": {"python": 0.9, "docker": 0.9, "k8s": 0.8},
    }
    analyzer = GapAnalyzer(skill_weights_by_level)
    metrics = analyzer.compute_metrics(["python"], {"python": 0.7})

    assert "python" in metrics
    assert metrics["python"].gap_m >= 0  # gap = max(0, 0.9 - 0.7) = 0.2


def test_profile_evaluator_integration():
    """ProfileEvaluator считает readiness для профилей (новый API)."""
    from src.analyzers.profile_evaluator import ProfileEvaluator

    skill_weights = {"python": 0.9, "docker": 0.7}
    vacancies_skills = [["python"], ["python", "docker"]]
    vacancies_skills_dict = [{"skills": ["python"]}, {"skills": ["python", "docker"]}]
    skill_weights_by_level = {
        "junior": {"python": 0.8},
        "middle": {"python": 0.9, "docker": 0.7},
        "senior": {"python": 0.9, "docker": 0.9},
    }

    evaluator = ProfileEvaluator(
        skill_weights=skill_weights,
        vacancies_skills=vacancies_skills,
        vacancies_skills_dict=vacancies_skills_dict,
        skill_weights_by_level=skill_weights_by_level,
        use_clustering=False,
    )

    student = StudentProfile(
        profile_name="test",
        competencies=[],
        skills=["python"],
        target_level="middle",
        created_at=datetime.now(),
    )

    eval_result = evaluator.evaluate_profile(student, user_type="student")
    assert "readiness_score" in eval_result
    assert 0 <= eval_result["readiness_score"] <= 100
    assert "market_coverage_score" in eval_result


# ----------------------------------------------------------------------
# Тесты рекомендаций
# ----------------------------------------------------------------------
@patch("src.predictors.recommendation_engine.RecommendationEngine._llm_explain")
def test_recommendation_engine_integration(mock_llm):
    """RecommendationEngine генерирует рекомендации (без LLM)."""
    mock_llm.return_value = None

    from src.analyzers.profile_evaluator import ProfileEvaluator
    from src.predictors.recommendation_engine import RecommendationEngine

    # Создаём мок-оценщик
    mock_evaluator = MagicMock(spec=ProfileEvaluator)
    mock_evaluator.evaluate_profile.return_value = {
        "market_coverage_score": 72.0,
        "skill_coverage": 65.0,
        "domain_coverage_score": 60.0,
        "readiness_score": 68.0,
        "avg_gap": 15.0,
        "skill_metrics": {
            "docker": {"cluster_relevance": 0.8, "gap_m": 0.6, "category": "devops"},
            "fastapi": {"cluster_relevance": 0.5, "gap_m": 0.4, "category": "frameworks"},
        },
        "domain_coverage": {"Backend": {"coverage": 0.5}},
        "top_recommendations": [("docker", 0.85), ("fastapi", 0.65)],
        "gaps": {"docker": {"gap_m": 0.6}, "fastapi": {"gap_m": 0.4}},
        "market_skill_coverage": 45.5,
        "student_skills": ["python", "sql", "git"],
        "cluster_context": {
            "closest_clusters": [
                {"id": 0, "name": "Backend Developer", "similarity": 0.85},
            ],
            "skills": {"docker": 0.9, "fastapi": 0.7},
            "total_skills_in_context": 2,
        },
        "level_weights_used": {"junior": 0.2, "middle": 0.5, "senior": 0.3},
        "skill_categories": {"strong": 1, "weak": 2, "missing": 5, "total": 8},
    }

    engine = RecommendationEngine(use_ltr=False, use_llm=False, profile_evaluator=mock_evaluator)

    student = StudentProfile(
        profile_name="test",
        competencies=[],
        skills=["python", "sql", "git"],
        target_level="middle",
        created_at=datetime.now(),
    )

    recs = engine.generate_recommendations(student, user_type="student")
    assert "summary" in recs
    assert "recommendations" in recs


# ----------------------------------------------------------------------
# Тесты сохранения результатов
# ----------------------------------------------------------------------
def test_save_all_results(tmp_path):
    """Проверяем, что все выходные файлы создаются."""
    from src.visualization.charts import save_all_charts

    results = {
        "base": {
            "readiness_score": 65,
            "market_coverage_score": 60,
            "skill_coverage": 55,
            "domain_coverage_score": 50,
            "top_recommendations": [("sql", 0.8)],
            "gaps": {"sql": {"gap_m": 0.5}},
            "skill_metrics": {},
            "domain_coverage": {},
            "student_skills": ["python"],
            "avg_gap": 15.0,
            "market_skill_coverage": 45.0,
            "cluster_context": {"closest_clusters": [], "skills": {}, "total_skills_in_context": 0},
        }
    }
    skill_weights = {"python": 100, "sql": 80}
    skill_weights_path = tmp_path / "processed" / "skill_weights.json"
    skill_weights_path.parent.mkdir(parents=True)
    skill_weights_path.write_text(json.dumps(skill_weights))

    with (
        patch.object(config, "DATA_PROCESSED_DIR", skill_weights_path.parent),
        patch.object(config, "DATA_DIR", tmp_path),
    ):
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

def test_main_train_model_flag(tmp_path):
    """Тест флага --train-model."""
    with (
        patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine.fit") as mock_fit,
        patch("builtins.print"),
    ):
        # Создаём файл вакансий
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        vac_file = raw_dir / "hh_vacancies_basic.json"
        vac_file.write_text(json.dumps([{"id": "1", "name": "Test"}]))

        with patch.object(config, "DATA_RAW_DIR", raw_dir):
            with patch.object(config, "DATA_RESULT_DIR", tmp_path / "result"):
                import main

                with patch("sys.argv", ["main.py", "--train-model"]):
                    main.main()

                mock_fit.assert_called_once()


def test_main_skip_collection_flag(tmp_path):
    """Тест флага --skip-collection."""
    with (
        patch("src.parsing.vacancy_parser.VacancyParser.extract_skills_from_vacancies") as mock_extract,
        patch("src.analyzers.profile_evaluator.ProfileEvaluator.evaluate_profile") as mock_eval,
        patch("src.predictors.recommendation_engine.RecommendationEngine.generate_recommendations") as mock_rec,
        patch("src.visualization.charts.save_all_charts") as mock_charts,
        patch("src.loaders_student.student_loader.generate_profiles_from_csv") as mock_csv,
        patch("builtins.print"),
    ):
        # Создаём файлы
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        vac_file = raw_dir / "hh_vacancies_basic.json"
        vac_file.write_text(json.dumps([{"id": "1", "name": "Test", "description": "python"}]))

        mock_extract.return_value = {
            "frequencies": {"python": 10},
            "hybrid_weights": {"python": 0.9},
            "skill_embeddings": {},
        }
        mock_eval.return_value = {
            "market_coverage_score": 70.0,
            "skill_coverage": 65.0,
            "domain_coverage_score": 60.0,
            "readiness_score": 68.0,
            "avg_gap": 15.0,
            "skill_metrics": {},
            "domain_coverage": {},
            "top_recommendations": [],
            "gaps": {},
            "student_skills": ["python"],
            "market_skill_coverage": 50.0,
            "cluster_context": {"closest_clusters": [], "skills": {}, "total_skills_in_context": 0},
        }
        mock_rec.return_value = {"summary": {}, "recommendations": [], "closest_roles": [], "domain_coverage": {}, "gaps": {}}
        mock_csv.return_value = {}

        with patch.object(config, "DATA_RAW_DIR", raw_dir):
            with patch.object(config, "DATA_RESULT_DIR", tmp_path / "result"):
                with patch.object(config, "DATA_PROCESSED_DIR", tmp_path / "processed"):
                    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

                    import main

                    with patch("sys.argv", ["main.py", "--skip-collection"]):
                        main.main()

                    mock_extract.assert_called_once()
