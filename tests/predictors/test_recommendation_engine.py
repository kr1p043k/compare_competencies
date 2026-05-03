# tests/predictors/test_recommendation_engine.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.predictors.recommendation_engine import RecommendationEngine
from src.models.student import StudentProfile
from src.analyzers.profile_evaluator import ProfileEvaluator
from src import config


@pytest.fixture
def mock_profile_evaluator():
    mock_eval = MagicMock(spec=ProfileEvaluator)
    mock_eval.evaluate_profile.return_value = {
        "market_coverage_score": 72.0,
        "skill_coverage": 65.0,
        "domain_coverage_score": 60.0,
        "readiness_score": 68.0,
        "avg_gap": 15.0,
        "skill_metrics": {
            "docker": {"cluster_relevance": 0.8, "gap_m": 0.6},
            "fastapi": {"cluster_relevance": 0.5, "gap_m": 0.4},
            "k8s": {"cluster_relevance": 0.3, "gap_m": 0.7},
        },
        "domain_coverage": {"Backend": {"coverage": 0.5}},
        "top_recommendations": [
            ("docker", 0.85),
            ("fastapi", 0.65),
            ("k8s", 0.45),
        ],
        "gaps": {
            "docker": {"gap_m": 0.6, "gap_j": 0.3, "gap_s": 0.8},
            "fastapi": {"gap_m": 0.4, "gap_j": 0.5, "gap_s": 0.6},
        },
        "market_skill_coverage": 45.5,
        "student_skills": ["python", "sql", "git"],
        "level_weights_used": {"junior": 0.2, "middle": 0.5, "senior": 0.3},
    }
    return mock_eval


@pytest.fixture
def mock_skill_filter():
    mock_sf = MagicMock()
    mock_sf.get_skill_categories.return_value = {"other": []}
    return mock_sf


@pytest.fixture
def sample_student_profile():
    return StudentProfile(
        profile_name="base",
        competencies=["SS1.1", "DL-1.3"],
        skills=["python", "sql", "git"],
        target_level="middle",
        created_at=datetime.now()
    )


class TestInit:
    def test_default_init(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine.use_ltr is True
        assert engine.use_llm is False
        assert engine.profile_evaluator is mock_profile_evaluator

    def test_init_with_llm_keys(self, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True)
        assert engine.use_llm is True

    def test_init_disable_ltr(self, mock_profile_evaluator):
        engine = RecommendationEngine(use_ltr=False, profile_evaluator=mock_profile_evaluator)
        assert engine.use_ltr is False
        assert engine.ltr_engine is None

    def test_init_without_profile_evaluator(self):
        engine = RecommendationEngine()
        assert engine.profile_evaluator is None


class TestClusterContext:
    def test_set_cluster_context(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        weights = {"docker": 0.9, "fastapi": 0.7}
        engine.set_cluster_context(weights)
        assert engine.cluster_weights == weights

    def test_clear_cluster_context(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.set_cluster_context({"docker": 0.9})
        engine.clear_cluster_context()
        assert engine.cluster_weights is None


class TestLoadTemplates:
    def test_load_templates_from_file(self, tmp_path, monkeypatch, mock_profile_evaluator):
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        templates_file = templates_dir / "recommendation_templates.json"
        templates_file.write_text(json.dumps({
            "hard_skills": {"python": "Python is great"},
            "soft_skills": {"communication": "Soft skill"},
            "hard_paths": {"python": "Learn Python"},
            "soft_paths": {"communication": "Practice"}
        }), encoding='utf-8')
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine.HARD_SKILL_TEMPLATES["python"] == "Python is great"
        assert engine.SOFT_SKILL_TEMPLATES["communication"] == "Soft skill"

    def test_default_templates(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert "python" in engine.HARD_SKILL_TEMPLATES
        assert "английский язык" in engine.SOFT_SKILL_TEMPLATES


class TestGenerateRecommendations:
    def test_generate_without_profile_evaluator(self):
        engine = RecommendationEngine()
        student = StudentProfile(profile_name="test", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now())
        with pytest.raises(AttributeError):
            engine.generate_recommendations(student)

    def test_generate_success(self, mock_profile_evaluator, sample_student_profile):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile)
        assert "summary" in result
        assert "recommendations" in result
        assert result["summary"]["readiness_score"] == 68.0
        recs = result["recommendations"]
        assert len(recs) == 3
        assert recs[0]["skill"] == "docker"
        assert recs[0]["importance_score"] == 0.85

    def test_generate_with_user_type(self, mock_profile_evaluator, sample_student_profile):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile, user_type='junior')
        mock_profile_evaluator.evaluate_profile.assert_called_with(sample_student_profile, user_type='junior')

    def test_generate_empty_recommendations(self, mock_profile_evaluator, sample_student_profile):
        mock_profile_evaluator.evaluate_profile.return_value = {
            "market_coverage_score": 90.0, "skill_coverage": 85.0, "domain_coverage_score": 80.0,
            "readiness_score": 88.0, "avg_gap": 5.0, "skill_metrics": {}, "domain_coverage": {},
            "top_recommendations": [], "gaps": {}, "market_skill_coverage": 80.0,
        }
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile)
        assert result["recommendations"] == []


class TestGenerateExplanation:
    def test_cluster_relevance_high(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        expl = engine._generate_explanation("docker", 0.85, {"skill_metrics": {"docker": {"cluster_relevance": 0.9}}})
        assert "🎯 Сильно связан" in expl

    def test_high_score(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        expl = engine._generate_explanation("fastapi", 0.75, {"skill_metrics": {"fastapi": {"cluster_relevance": 0.3}}})
        assert "🔴 Один из самых" in expl

    def test_low_score(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        expl = engine._generate_explanation("html", 0.20, {"skill_metrics": {"html": {"cluster_relevance": 0.1}}})
        assert "🟢 Полезен" in expl


class TestHardSkillDetection:
    def test_is_hard_skill_by_keyword(self, mock_profile_evaluator, mock_skill_filter):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.skill_filter = mock_skill_filter
        mock_skill_filter.get_skill_categories.return_value = {}
        assert engine._is_hard_skill("docker") is True
        assert engine._is_hard_skill("kubernetes") is True

    def test_is_soft_skill(self, mock_profile_evaluator, mock_skill_filter):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.skill_filter = mock_skill_filter
        mock_skill_filter.get_skill_categories.return_value = {}
        assert engine._is_hard_skill("коммуникация") is False


class TestTimeframes:
    def test_get_timeframe_easy(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine._get_timeframe("git") == "1-2 недели"

    def test_get_timeframe_medium(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine._get_timeframe("python") == "1-2 месяца"

    def test_get_timeframe_hard(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine._get_timeframe("java") == "2-6 месяцев"


class TestLearningPaths:
    def test_get_learning_path_junior(self, mock_profile_evaluator, sample_student_profile):
        sample_student_profile.target_level = "junior"
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("python", False, sample_student_profile)
        assert "Сфокусируйтесь на основах" in path

    def test_get_learning_path_senior(self, mock_profile_evaluator):
        profile = StudentProfile(profile_name="senior_dev", competencies=[], skills=["python"], target_level="senior", created_at=datetime.now())
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("docker", False, profile)
        assert "Углублённое изучение" in path


class TestPriority:
    def test_get_priority_high(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine._get_priority(0.7) == "HIGH"

    def test_get_priority_medium(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine._get_priority(0.4) == "MEDIUM"

    def test_get_priority_low(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine._get_priority(0.1) == "LOW"


class TestWhyImportant:
    def test_why_important_high_priority(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine._why_important("docker", 0.8, "HIGH")
        assert "🔴 ВЫСОКИЙ приоритет" in result

    def test_why_important_with_ltr(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine._why_important("docker", 0.8, "HIGH", ltr_explanation="LTR says important")
        assert "🎯 Модель" in result


class TestLLM:
    def test_llm_disabled(self, mock_profile_evaluator):
        engine = RecommendationEngine(use_llm=False, profile_evaluator=mock_profile_evaluator)
        result = engine._llm_explain("docker", 0.8, "HIGH", ["python"], 70.0)
        assert result is None

    def test_llm_explain_with_retry_disabled(self, mock_profile_evaluator):
        engine = RecommendationEngine(use_llm=False, profile_evaluator=mock_profile_evaluator)
        result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0)
        assert result is None

class TestFullCoverageRecommendationEngine:
    """Дополнительные тесты для покрытия пропущенных строк"""

    def test_init_with_ltr_model_loaded(self, tmp_path, monkeypatch, mock_profile_evaluator):
        """Строки 46-48: загрузка LTR модели при инициализации"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        import joblib
        model_path = models_dir / "ltr_ranker_xgb_regressor.joblib"
        joblib.dump({"model": None, "feature_names": [], "skill_metadata": {}, "skill_embeddings": {}, "total_vacancies": 0}, model_path)

        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        engine = RecommendationEngine(use_ltr=True, profile_evaluator=mock_profile_evaluator)
        assert engine.ltr_engine is not None
        assert engine.ltr_engine.is_fitted

    def test_init_ltr_load_exception(self, tmp_path, monkeypatch, mock_profile_evaluator):
        """Строки 46-48: исключение при загрузке LTR"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        model_path = models_dir / "ltr_ranker_xgb_regressor.joblib"
        model_path.write_text("corrupted data")

        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        engine = RecommendationEngine(use_ltr=True, profile_evaluator=mock_profile_evaluator)
        assert engine.ltr_engine is None

    def test_load_templates_exception(self, tmp_path, monkeypatch, mock_profile_evaluator):
        """Строки 59, 77-78: ошибка загрузки шаблонов"""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        templates_file = templates_dir / "recommendation_templates.json"
        templates_file.write_text("{invalid json")

        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert "python" in engine.HARD_SKILL_TEMPLATES

    def test_fit_with_vacancies(self, mock_profile_evaluator):
        """Строки 99-110: fit с данными"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.fit([["python", "sql"], ["java", "spring"]], skill_weights={"python": 0.9, "sql": 0.7, "java": 0.6})
        assert engine.is_fitted

    def test_fit_without_skill_weights(self, mock_profile_evaluator):
        """Строки 113-135: fit без skill_weights"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        with pytest.raises(ValueError, match="skill_weights обязательны"):
            engine.fit([["python"]], skill_weights=None)

    def test_fit_with_empty_vacancies(self, mock_profile_evaluator):
        """Строка 154: fit с пустыми вакансиями"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.fit([], skill_weights={"python": 0.9})
        assert not engine.is_fitted

    def test_generate_recommendations_coverage_summary(self, mock_profile_evaluator, sample_student_profile):
        """Строки 231-234: summary с coverage_details"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile)
        summary = result["summary"]
        assert "covered_skills_count" in summary["coverage_details"]
        assert "total_market_skills" in summary["coverage_details"]
        assert "market_skill_coverage" in summary

    def test_is_hard_skill_with_categories(self, mock_profile_evaluator, mock_skill_filter):
        """Строки 279-285: определение hardskill через категории"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.skill_filter = mock_skill_filter
        mock_skill_filter.get_skill_categories.return_value = {"programming_languages": ["python"]}
        assert engine._is_hard_skill("python") is True

    def test_get_suggestion_fallback(self, mock_profile_evaluator):
        """Строки 296-340: шаблоны для неизвестных навыков"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        suggestion = engine._get_suggestion("unknown_skill", False)
        assert "unknown_skill" in suggestion
        suggestion_soft = engine._get_suggestion("unknown_soft", True)
        assert "soft skill" in suggestion_soft.lower()

    def test_why_important_with_shap_values(self, mock_profile_evaluator):
        """Строки 355-367: объяснение с SHAP значениями"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.1, 0.9]])
        X = pd.DataFrame([[0.5, 0.95]], columns=["level_encoded", "cosine_sim"])
        result = engine._why_important(
            "docker", 0.8, "HIGH",
            shap_values=shap_vals, X=X, idx=0,
            feature_names=["level_encoded", "cosine_sim"]
        )
        assert "🎯 Навык" in result

    def test_get_learning_path_default(self, mock_profile_evaluator):
        """Строки 378-380: дефолтный путь обучения"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("unknown_skill", False, None)
        assert "Изучите документацию" in path

    def test_get_expected_outcome_with_role(self, mock_profile_evaluator):
        """Строки 383-390: expected outcome с ролью"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        outcome = engine._get_expected_outcome("docker", None)
        assert "docker" in outcome
        assert "вашей целевой роли" in outcome

    def test_llm_with_retry_success(self, mock_profile_evaluator, monkeypatch):
        """Строки 396-399: LLM успешный ответ"""
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"alternatives": [{"message": {"text": "LLM response"}}]}}

        with patch("requests.post", return_value=mock_resp):
            result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0)
            assert result == "LLM response"

    def test_llm_with_retry_429_then_success(self, mock_profile_evaluator, monkeypatch):
        """Строки 396-399: LLM с 429 и последующим успехом"""
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)

        mock_fail = MagicMock()
        mock_fail.status_code = 429
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"result": {"alternatives": [{"message": {"text": "Success after retry"}}]}}

        with patch("requests.post", side_effect=[mock_fail, mock_success]):
            with patch("time.sleep", return_value=None):
                result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0, max_retries=1)
                assert result == "Success after retry"

    def test_llm_with_retry_error_status(self, mock_profile_evaluator, monkeypatch):
        """Строки 396-399: LLM с ошибкой"""
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_resp):
            result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0)
            assert result is None

    def test_llm_with_retry_exception(self, mock_profile_evaluator, monkeypatch):
        """Строки 396-399: LLM с исключением"""
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)

        with patch("requests.post", side_effect=Exception("Network error")):
            with patch("time.sleep", return_value=None):
                result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0)
                assert result is None

    def test_why_important_llm_enabled(self, mock_profile_evaluator, monkeypatch):
        """Строки 355-367: why_important с LLM"""
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"alternatives": [{"message": {"text": "LLM explanation"}}]}}

        with patch("requests.post", return_value=mock_resp):
            result = engine._why_important("docker", 0.8, "HIGH", student_skills=["python"], coverage=70.0)
            assert "🤖 LLM explanation" in result

    def test_why_important_low_priority(self, mock_profile_evaluator):
        """Строка 431-432: низкий приоритет"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine._why_important("html", 0.2, "LOW")
        assert "🟢 НИЗКИЙ приоритет" in result

    def test_generate_recommendations_ranking(self, mock_profile_evaluator, sample_student_profile):
        """Проверка сортировки рекомендаций"""
        mock_profile_evaluator.evaluate_profile.return_value = {
            "market_coverage_score": 50.0, "skill_coverage": 40.0,
            "domain_coverage_score": 30.0, "readiness_score": 45.0,
            "avg_gap": 25.0, "skill_metrics": {},
            "domain_coverage": {},
            "top_recommendations": [
                ("k8s", 0.45),
                ("docker", 0.85),
                ("fastapi", 0.65),
            ],
            "gaps": {}, "market_skill_coverage": 30.0,
        }
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile)
        recs = result["recommendations"]
        # Должны быть отсортированы по убыванию importance_score
        assert recs[0]["skill"] == "docker"
        assert recs[0]["rank"] == 1
        assert recs[1]["rank"] == 2
        assert recs[2]["rank"] == 3

    def test_shap_explain_level_junior(self, mock_profile_evaluator):
        """SHAP объяснение для junior уровня"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.9, 0.1]])
        X = pd.DataFrame([[1, 0.5]], columns=["level_encoded", "cosine_sim"])
        result = engine._shap_explain("python", shap_vals, 0, X, ["level_encoded", "cosine_sim"])
        assert "junior" in result

    def test_shap_explain_category(self, mock_profile_evaluator):
        """SHAP объяснение для категории"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.1, 0.9]])
        X = pd.DataFrame([[2, 0.5]], columns=["level_encoded", "category_encoded"])
        result = engine._shap_explain("python", shap_vals, 0, X, ["level_encoded", "category_encoded"])
        assert "категории" in result

    def test_load_templates_file_error(self, tmp_path, monkeypatch, mock_profile_evaluator):
        """Строка 59: ошибка при загрузке файла шаблонов"""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        templates_file = templates_dir / "recommendation_templates.json"
        templates_file.write_text("{broken")

        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert "python" in engine.HARD_SKILL_TEMPLATES  # fallback defaults

    def test_analyze_with_cluster_context(self, mock_profile_evaluator):
        """Строки 113-135: analyze с кластерным контекстом"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.fit([["python", "sql"]], skill_weights={"python": 0.9, "sql": 0.7})
        engine.set_cluster_context({"docker": 0.8})
        assert engine.cluster_weights == {"docker": 0.8}

    def test_generate_skill_recommendation_full(self, mock_profile_evaluator, sample_student_profile):
        """Строки 231-234: полный метод _generate_skill_recommendation"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        rec = engine._generate_skill_recommendation(
            "docker", 0.85, "HIGH", 1,
            student_profile=sample_student_profile,
            student_skills=["python"],
            coverage=70.0
        )
        assert rec["skill"] == "docker"
        assert rec["importance_score"] == 0.85
        assert "why_important" in rec
        assert "how_to_learn" in rec

    def test_get_suggestion_template_missing(self, mock_profile_evaluator):
        """Строка 340: шаблон отсутствует — fallback"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        suggestion = engine._get_suggestion("nonexistent_skill", False)
        assert "nonexistent_skill" in suggestion

    def test_why_important_llm_fallback_to_shap(self, mock_profile_evaluator, monkeypatch):
        """Строки 356, 367: LLM возвращает None → SHAP"""
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)

        import numpy as np
        import pandas as pd
        shap_vals = np.array([[0.9, 0.1]])
        X = pd.DataFrame([[3, 0.95]], columns=["level_encoded", "cosine_sim"])

        with patch.object(engine, "_llm_explain", return_value=None):
            result = engine._why_important(
                "docker", 0.8, "HIGH",
                student_skills=["python"], coverage=70.0,
                shap_values=shap_vals, X=X, idx=0,
                feature_names=["level_encoded", "cosine_sim"]
            )
            assert "🎯 Навык" in result

    def test_get_expected_outcome_no_role(self, mock_profile_evaluator):
        """Строки 388-389: expected outcome без роли"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        outcome = engine._get_expected_outcome("docker", None)
        assert "docker" in outcome

    def test_llm_with_retry_all_fail(self, mock_profile_evaluator, monkeypatch):
        """Строки 397, 406: LLM все попытки неудачны"""
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Error"

        with patch("requests.post", return_value=mock_resp):
            result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0, max_retries=0)
            assert result is None

    def test_fit_with_empty_vacancies(self, mock_profile_evaluator):
        """Строка 57: fit с пустыми вакансиями"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.fit([], skill_weights={"python": 0.9})
        assert not engine.is_fitted

    def test_get_suggestion_soft_skill_fallback(self, mock_profile_evaluator):
        """Строка 305: soft skill без шаблона"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        suggestion = engine._get_suggestion("unknown_soft", True)
        assert "soft skill" in suggestion.lower()

    def test_why_important_shap_level_junior(self, mock_profile_evaluator):
        """Строка 321: SHAP с уровнем junior"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.9, 0.1]])
        X = pd.DataFrame([[1, 0.5]], columns=["level_encoded", "cosine_sim"])
        result = engine._shap_explain("python", shap_vals, 0, X, ["level_encoded", "cosine_sim"])
        assert "junior" in result

    def test_why_important_shap_unknown_feature(self, mock_profile_evaluator):
        """Строка 332: SHAP с неизвестной фичей"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.9, 0.1]])
        X = pd.DataFrame([[2, 0.5]], columns=["unknown_feat", "cosine_sim"])
        result = engine._shap_explain("python", shap_vals, 0, X, ["unknown_feat", "cosine_sim"])
        assert result is None

    def test_why_important_medium_priority_shap(self, mock_profile_evaluator):
        """Строки 353-354: MEDIUM приоритет с SHAP"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.1, 0.9]])
        X = pd.DataFrame([[2, 0.95]], columns=["level_encoded", "cosine_sim"])
        result = engine._why_important(
            "docker", 0.5, "MEDIUM",
            shap_values=shap_vals, X=X, idx=0,
            feature_names=["level_encoded", "cosine_sim"]
        )
        assert "Его освоение повысит" in result

    def test_get_learning_path_middle(self, mock_profile_evaluator):
        """Строка 362: middle уровень"""
        profile = StudentProfile(
            profile_name="mid_dev", competencies=[], skills=["python"],
            target_level="middle", created_at=datetime.now()
        )
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("python", False, profile)
        assert "Основы Python" in path or "Изучите документацию" in path

    def test_get_learning_path_soft_skill(self, mock_profile_evaluator):
        """Строка 371: soft skill путь обучения"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("английский язык", True, None)
        assert "Занимайтесь ежедневно" in path

    def test_fit_without_skill_weights_raises(self, mock_profile_evaluator):
        """Строка 119: fit без skill_weights"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        with pytest.raises(ValueError, match="skill_weights обязательны"):
            engine.fit([["python"]], skill_weights=None)

    def test_fit_empty_vacancies(self, mock_profile_evaluator):
        """Строка 57: fit с пустыми вакансиями"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.fit([], skill_weights={"python": 0.9})
        assert not engine.is_fitted

    def test_fit_missing_skill_weights(self, mock_profile_evaluator):
        """Строка 119: fit без skill_weights"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        with pytest.raises(ValueError, match="skill_weights обязательны"):
            engine.fit([["python"]], skill_weights=None)

    def test_get_suggestion_soft_fallback(self, mock_profile_evaluator):
        """Строка 305: soft skill без шаблона"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine._get_suggestion("unknown_skill", True)
        assert "soft skill" in result.lower()

    def test_shap_explain_level_junior(self, mock_profile_evaluator):
        """Строка 321: SHAP объяснение для junior"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.9, 0.1]])
        X = pd.DataFrame([[1, 0.5]], columns=["level_encoded", "cosine_sim"])
        result = engine._shap_explain("python", shap_vals, 0, X, ["level_encoded", "cosine_sim"])
        assert "junior" in result

    def test_get_learning_path_middle_level(self, mock_profile_evaluator):
        """Строка 362: путь обучения для middle"""
        profile = StudentProfile(
            profile_name="mid", competencies=[], skills=["python"],
            target_level="middle", created_at=datetime.now()
        )
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("python", False, profile)
        assert len(path) > 0

# tests/predictors/test_recommendation_engine.py — добавить в TestFullCoverageRecommendationEngine:

    def test_fit_with_skill_weights(self, mock_profile_evaluator):
        """Строка 57: успешный fit с данными"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.fit([["python", "sql"]], skill_weights={"python": 0.9, "sql": 0.7})
        assert engine.is_fitted is True

    def test_fit_without_skill_weights_error(self, mock_profile_evaluator):
        """Строка 119: fit с None вместо skill_weights"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        with pytest.raises(ValueError, match="skill_weights обязательны"):
            engine.fit([["python"]], skill_weights=None)

    def test_get_suggestion_soft_missing_template(self, mock_profile_evaluator):
        """Строка 305: soft skill без шаблона"""
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        suggestion = engine._get_suggestion("unknown_skill", True)
        assert "soft skill" in suggestion.lower()

    def test_shap_explain_junior_level(self, mock_profile_evaluator):
        """Строка 321: SHAP с junior (level_encoded=1)"""
        import numpy as np
        import pandas as pd
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        shap_vals = np.array([[0.9]])
        X = pd.DataFrame([[1]], columns=["level_encoded"])
        result = engine._shap_explain("python", shap_vals, 0, X, ["level_encoded"])
        assert "junior" in result

    def test_get_learning_path_middle_default(self, mock_profile_evaluator):
        """Строка 362: путь обучения для middle уровня"""
        profile = StudentProfile(
            profile_name="mid", competencies=[], skills=["python"],
            target_level="middle", created_at=datetime.now()
        )
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("python", False, profile)
        assert len(path) > 0