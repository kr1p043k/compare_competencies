# tests/predictors/test_recommendation_engine.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.predictors.recommendation_engine import RecommendationEngine
from src.models.student import StudentProfile
from src import config


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Мокируем все внешние компоненты RecommendationEngine."""
    # Comparator
    mock_comparator_cls = MagicMock()
    mock_comparator = MagicMock()
    mock_comparator.compare.return_value = (0.75, 0.9)
    mock_comparator.get_skill_weights.return_value = {"python": 0.9}
    mock_comparator_cls.return_value = mock_comparator
    monkeypatch.setattr("src.predictors.recommendation_engine.CompetencyComparator", mock_comparator_cls)

    # GapAnalyzer
    mock_gap_cls = MagicMock()
    mock_gap = MagicMock()
    mock_gap.analyze_gap.return_value = {
        "high_priority": [{"skill": "docker", "importance": 0.8}],
        "medium_priority": [],
        "low_priority": [],
        "total_gaps": 1,
        "stats": {}
    }
    mock_gap.coverage.return_value = (65.0, {"covered_skills_count": 2, "total_market_skills": 10})
    mock_gap.top_market_skills.return_value = []
    mock_gap_cls.return_value = mock_gap
    monkeypatch.setattr("src.predictors.recommendation_engine.GapAnalyzer", mock_gap_cls)

    # SkillFilter
    mock_filter_cls = MagicMock()
    mock_filter = MagicMock()
    mock_filter.validate_skills.return_value = ["python", "sql"]
    mock_filter.get_skill_categories.return_value = {"programming_languages": ["python"]}
    mock_filter_cls.return_value = mock_filter
    monkeypatch.setattr("src.predictors.recommendation_engine.SkillFilter", mock_filter_cls)

    # LTR
    mock_ltr_cls = MagicMock()
    mock_ltr = MagicMock()
    mock_ltr.is_fitted = False
    mock_ltr.load_model.return_value = None
    mock_ltr_cls.return_value = mock_ltr
    monkeypatch.setattr("src.predictors.recommendation_engine.LTRRecommendationEngine", mock_ltr_cls)

    # LLM – отключаем реальные запросы
    monkeypatch.setattr("src.predictors.recommendation_engine.RecommendationEngine._llm_explain", MagicMock(return_value=None))
    monkeypatch.setattr("src.predictors.recommendation_engine.RecommendationEngine._llm_explain_with_retry", MagicMock(return_value=None))

    # Конфиг
    monkeypatch.setattr(config, "YC_API_KEY", None)
    monkeypatch.setattr(config, "YC_FOLDER_ID", None)
    monkeypatch.setattr(config, "YANDEXGPT_MODEL", "yandexgpt-lite")
    monkeypatch.setattr(config, "MODELS_DIR", Path("/tmp/models"))

    return {
        "comparator": mock_comparator,
        "comparator_cls": mock_comparator_cls,
        "gap_analyzer": mock_gap,
        "gap_analyzer_cls": mock_gap_cls,
        "skill_filter": mock_filter,
        "ltr_engine": mock_ltr,
    }


@pytest.fixture
def sample_student_profile():
    return StudentProfile(
        profile_name="base",
        competencies=["SS1.1", "DL-1.3"],
        skills=["python", "sql", "git"],
        target_level="junior"
    )


@pytest.fixture
def sample_vacancies_skills():
    return [
        ["python", "sql", "pandas"],
        ["python", "docker", "fastapi"],
        ["java", "spring", "sql"],
        ["python", "machine learning", "pytorch"],
        ["javascript", "react", "html"],
    ]


@pytest.fixture
def sample_skill_weights():
    return {"python": 1.0, "sql": 0.8, "docker": 0.6, "java": 0.7, "fastapi": 0.5}


class TestInit:
    def test_default_init(self, mock_dependencies):
        engine = RecommendationEngine()
        assert engine.use_ltr is True
        assert engine.use_llm is False
        assert engine.is_fitted is False

    def test_init_with_llm_keys(self, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", "test-key")
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True)
        assert engine.use_llm is True

    def test_init_disable_ltr(self, mock_dependencies):
        engine = RecommendationEngine(use_ltr=False)
        assert engine.use_ltr is False
        assert engine.ltr_engine is None


class TestFit:
    def test_fit_success(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine()
        engine.fit(sample_vacancies_skills, skill_weights=sample_skill_weights)

        mock_dependencies["comparator"].fit_market.assert_called_once_with(sample_vacancies_skills)
        mock_dependencies["gap_analyzer_cls"].assert_called_once_with(sample_skill_weights)
        assert engine.is_fitted is True

    def test_fit_empty_vacancies(self, mock_dependencies):
        engine = RecommendationEngine()
        engine.fit([])
        assert engine.is_fitted is False
        mock_dependencies["comparator"].fit_market.assert_not_called()

    def test_fit_without_skill_weights(self, mock_dependencies, sample_vacancies_skills):
        mock_dependencies["comparator"].get_skill_weights.return_value = {"python": 0.9}
        engine = RecommendationEngine()
        engine.fit(sample_vacancies_skills)
        mock_dependencies["gap_analyzer_cls"].assert_called_once_with({"python": 0.9})


class TestAnalyze:
    def test_analyze_not_fitted(self, mock_dependencies):
        engine = RecommendationEngine()
        result = engine.analyze(["python"])
        assert result == {}

    def test_analyze_success(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine()
        engine.fit(sample_vacancies_skills, sample_skill_weights)

        result = engine.analyze(["python", "sql"])

        assert result["match_score"] == 0.75
        assert result["confidence"] == 0.9
        assert result["coverage"] == 65.0
        assert "gaps" in result


class TestGenerateRecommendations:
    def test_generate_not_fitted(self, mock_dependencies):
        engine = RecommendationEngine()
        result = engine.generate_recommendations(["python"])
        assert result == {}

    def test_generate_success_no_ltr_no_llm(
        self, mock_dependencies, sample_vacancies_skills, sample_skill_weights, sample_student_profile
    ):
        engine = RecommendationEngine(use_ltr=False, use_llm=False)
        engine.fit(sample_vacancies_skills, sample_skill_weights)

        result = engine.generate_recommendations(["python"], student_profile=sample_student_profile)

        assert "summary" in result
        assert result["summary"]["coverage"] == 65.0
        assert len(result["recommendations"]) == 1
        rec = result["recommendations"][0]
        assert rec["skill"] == "docker"
        assert rec["priority"] == "HIGH"
        assert "🔴" in rec["why_important"]

    def test_generate_with_ltr_fallback(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine(use_ltr=True, use_llm=False)
        engine.fit(sample_vacancies_skills, sample_skill_weights)

        mock_ltr = mock_dependencies["ltr_engine"]
        mock_ltr.is_fitted = False
        engine.ltr_engine = mock_ltr

        result = engine.generate_recommendations(["python"])
        assert result["recommendations"][0]["importance_score"] == 0.8

    def test_generate_with_ltr_success(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine(use_ltr=True, use_llm=False)
        engine.fit(sample_vacancies_skills, sample_skill_weights)

        mock_ltr = mock_dependencies["ltr_engine"]
        mock_ltr.is_fitted = True
        mock_ltr.predict_skill_impact.return_value = [
            ("docker", 85.0, "высокий вес"),
        ]
        engine.ltr_engine = mock_ltr

        result = engine.generate_recommendations(["python"])

        rec_docker = result["recommendations"][0]
        assert rec_docker["importance_score"] == 0.85
        assert "🎯 Модель: высокий вес" in rec_docker["why_important"]

    def test_generate_with_llm_explanation(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine(use_ltr=False, use_llm=True)
        engine.fit(sample_vacancies_skills, sample_skill_weights)

        with patch.object(engine, "_llm_explain", return_value="Живое объяснение от LLM"):
            result = engine.generate_recommendations(["python"])
            rec = result["recommendations"][0]
            assert "🤖 Живое объяснение от LLM" in rec["why_important"]


class TestSkillClassification:
    def test_is_hard_skill_true(self, mock_dependencies):
        engine = RecommendationEngine()
        mock_dependencies["skill_filter"].get_skill_categories.return_value = {"programming_languages": ["python"]}
        assert engine._is_hard_skill("python") is True

    def test_is_hard_skill_false(self, mock_dependencies):
        engine = RecommendationEngine()
        mock_dependencies["skill_filter"].get_skill_categories.return_value = {}
        assert engine._is_hard_skill("английский язык") is False

    def test_is_hard_skill_keyword_fallback(self, mock_dependencies):
        engine = RecommendationEngine()
        mock_dependencies["skill_filter"].get_skill_categories.return_value = {}
        assert engine._is_hard_skill("docker") is True
        assert engine._is_hard_skill("communication") is False


class TestTemplates:
    def test_get_suggestion_hard(self, mock_dependencies):
        engine = RecommendationEngine()
        suggestion = engine._get_suggestion("python", is_soft=False)
        assert "Python — основной язык" in suggestion

    def test_get_suggestion_soft(self, mock_dependencies):
        engine = RecommendationEngine()
        suggestion = engine._get_suggestion("английский язык", is_soft=True)
        assert "Английский язык B2+" in suggestion

    def test_get_learning_path_junior(self, mock_dependencies, sample_student_profile):
        engine = RecommendationEngine()
        path = engine._get_learning_path("python", is_soft=False, student_profile=sample_student_profile)
        assert "Сфокусируйтесь на основах" in path

    def test_get_learning_path_senior(self, mock_dependencies):
        profile = StudentProfile(
            profile_name="top_dc", competencies=[], skills=[], target_level="senior"
        )
        engine = RecommendationEngine()
        path = engine._get_learning_path("docker", is_soft=False, student_profile=profile)
        assert "Углублённое изучение" in path
        assert "архитектурные паттерны" in path

    def test_get_timeframe(self, mock_dependencies):
        engine = RecommendationEngine()
        assert engine._get_timeframe("git") == "1-2 недели"
        assert engine._get_timeframe("python") == "1-2 месяца"
        assert engine._get_timeframe("java") == "2-6 месяцев"
        assert engine._get_timeframe("unknown") == "1-3 месяца"

    def test_load_templates_from_file(self, tmp_path, monkeypatch, mock_dependencies):
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
        engine = RecommendationEngine()
        assert engine.HARD_SKILL_TEMPLATES["python"] == "Python is great"
        assert engine.SOFT_SKILL_TEMPLATES["communication"] == "Soft skill"

    def test_why_important_llm_override(self, mock_dependencies):
        engine = RecommendationEngine(use_llm=False)
        with patch.object(engine, "_llm_explain", return_value="LLM says"):
            result = engine._why_important("docker", 0.5, "HIGH", student_skills=["python"])
            assert "🤖 LLM says" in result

    def test_why_important_ltr_fallback(self, mock_dependencies):
        engine = RecommendationEngine(use_llm=False)
        result = engine._why_important("docker", 0.5, "HIGH", ltr_explanation="LTR says")
        assert "🎯 Модель: LTR says" in result

    def test_why_important_priority_only(self, mock_dependencies):
        engine = RecommendationEngine(use_llm=False)
        result = engine._why_important("docker", 0.5, "MEDIUM")
        assert "🟡 СРЕДНИЙ приоритет" in result


class TestEdgeCases:
    def test_analyze_with_empty_skills(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine()
        engine.fit(sample_vacancies_skills, sample_skill_weights)
        result = engine.analyze([])
        assert result["match_score"] == 0.75
        assert result["coverage"] == 65.0

    def test_generate_recommendations_no_gaps(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine(use_ltr=False, use_llm=False)
        engine.fit(sample_vacancies_skills, sample_skill_weights)
        mock_dependencies["gap_analyzer"].analyze_gap.return_value = {
            "high_priority": [], "medium_priority": [], "low_priority": [], "total_gaps": 0
        }
        result = engine.generate_recommendations(["python"])
        assert result["recommendations"] == []

    def test_generate_recommendations_ltr_exception(self, mock_dependencies, sample_vacancies_skills, sample_skill_weights):
        engine = RecommendationEngine(use_ltr=True, use_llm=False)
        engine.fit(sample_vacancies_skills, sample_skill_weights)
        mock_ltr = mock_dependencies["ltr_engine"]
        mock_ltr.is_fitted = True
        mock_ltr.predict_skill_impact.side_effect = Exception("LTR failed")
        engine.ltr_engine = mock_ltr

        result = engine.generate_recommendations(["python"])
        assert result["recommendations"][0]["importance_score"] == 0.8