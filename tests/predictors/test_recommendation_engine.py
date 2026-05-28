# tests/predictors/test_recommendation_engine.py
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import sys
sys.modules['shap'] = MagicMock()
sys.modules['cv2'] = MagicMock()

import numpy as np
import pytest
from pydantic import SecretStr

from src import Err, Ok, config
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.models.student import StudentProfile
from src.predictors.recommendation_engine import RecommendationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
from src import Ok, Result


def _unwrap(res: Result):
    match res:
        case Ok(d):
            return d
        case _:
            raise AssertionError(f"Expected Ok, got {res}")


@pytest.fixture
def mock_profile_evaluator():
    evaluator = MagicMock(spec=ProfileEvaluator)
    evaluator.evaluate_profile.return_value = Ok({
        "market_coverage_score": 72.0,
        "skill_coverage": 65.0,
        "domain_coverage_score": 60.0,
        "readiness_score": 68.0,
        "avg_gap": 15.0,
        "skill_metrics": {
            "docker": {"cluster_relevance": 0.8, "gap_m": 0.6, "category": "missing"},
            "fastapi": {"cluster_relevance": 0.5, "gap_m": 0.4, "category": "weak"},
            "k8s": {"cluster_relevance": 0.3, "gap_m": 0.7, "category": "missing"},
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
        "cluster_context": {
            "closest_clusters": [
                {"id": 0, "name": "Backend Developer", "similarity": 0.85},
            ],
            "skills": {"docker": 0.9, "fastapi": 0.7},
        },
    })
    return evaluator


@pytest.fixture
def mock_skill_filter():
    sf = MagicMock()
    sf.get_skill_categories.return_value = {}
    return sf


@pytest.fixture
def sample_student_profile():
    return StudentProfile(
        profile_name="test_student",
        competencies=[],
        skills=["python", "sql", "git"],
        target_level="middle",
        created_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_default_init(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine.use_ltr is True
        assert engine.use_llm is False
        assert engine.profile_evaluator is mock_profile_evaluator

    def test_init_with_llm_keys(self, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", SecretStr("test-key"))
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


# ---------------------------------------------------------------------------
# Tests for cluster context
# ---------------------------------------------------------------------------
class TestClusterContext:
    def test_set_cluster_context(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.set_cluster_context({"docker": 0.9})
        assert engine.cluster_weights == {"docker": 0.9}

    def test_clear_cluster_context(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.set_cluster_context({"docker": 0.9})
        engine.clear_cluster_context()
        assert engine.cluster_weights is None


# ---------------------------------------------------------------------------
# Tests for template loading
# ---------------------------------------------------------------------------
class TestLoadTemplates:
    def test_load_templates_from_file(self, tmp_path, monkeypatch, mock_profile_evaluator):
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        templates_file = templates_dir / "recommendation_templates.json"
        templates_file.write_text(json.dumps({
            "hard_paths": {"python": "Learn Python"},
            "soft_paths": {"communication": "Practice"},
        }), encoding="utf-8")
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine.HARD_LEARNING_PATHS["python"] == "Learn Python"

    def test_default_templates(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert "python" in engine.HARD_LEARNING_PATHS
        assert "английский язык" in engine.SOFT_LEARNING_PATHS

    def test_load_templates_corrupted(self, tmp_path, monkeypatch, mock_profile_evaluator):
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        templates_file = templates_dir / "recommendation_templates.json"
        templates_file.write_text("{invalid")
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert "python" in engine.HARD_LEARNING_PATHS


# ---------------------------------------------------------------------------
# Tests for fit()
# ---------------------------------------------------------------------------
class TestFit:

    def test_fit_without_skill_weights(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.fit([["python"]], skill_weights=None)
        assert result.is_err()

    def test_fit_empty_vacancies(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.fit([], skill_weights={"python": 0.9})
        assert result.is_err()
        assert not engine.is_fitted


# ---------------------------------------------------------------------------
# Tests for generate_recommendations()
# ---------------------------------------------------------------------------
class TestGenerateRecommendations:
    def test_generate_without_profile_evaluator(self):
        engine = RecommendationEngine()
        student = StudentProfile(profile_name="t", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now())
        result = engine.generate_recommendations(student)
        assert result.is_err()

    def test_generate_success(self, mock_profile_evaluator, sample_student_profile):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        match engine.generate_recommendations(sample_student_profile):
            case Ok(result):
                assert result.summary.readiness_score == 68.0
                recs = result.recommendations
                assert len(recs) == 3
                assert recs[0].skill == "docker"
                assert recs[0].importance_score > 0

    def test_generate_with_user_type(self, mock_profile_evaluator, sample_student_profile):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        match engine.generate_recommendations(sample_student_profile, user_type="junior"):
            case Ok(_):
                mock_profile_evaluator.evaluate_profile.assert_called_with(sample_student_profile, user_type="junior", target_domains=None, taxonomy=None)

    def test_generate_empty_recommendations(self, mock_profile_evaluator, sample_student_profile):
        mock_profile_evaluator.evaluate_profile.return_value = Ok({
            "market_coverage_score": 90.0,
            "skill_coverage": 85.0,
            "domain_coverage_score": 80.0,
            "readiness_score": 88.0,
            "avg_gap": 5.0,
            "skill_metrics": {},
            "domain_coverage": {},
            "top_recommendations": [],
            "gaps": {},
            "market_skill_coverage": 80.0,
        })
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        match engine.generate_recommendations(sample_student_profile):
            case Ok(result):
                assert result.recommendations == []

    def test_generate_eval_result_none(self, mock_profile_evaluator, sample_student_profile):
        mock_profile_evaluator.evaluate_profile.return_value = None
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile)
        assert result.is_err()


# ---------------------------------------------------------------------------
# Tests for _generate_explanation (private, tested via generate or direct)
# ---------------------------------------------------------------------------
class TestGenerateExplanation:
    def test_explanation_high_cluster_relevance(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        expl = engine._generate_explanation("docker", 0.85, _unwrap(mock_profile_evaluator.evaluate_profile.return_value))
        assert "🎯" in expl

    def test_explanation_high_score_no_cluster(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        eval_copy = _unwrap(mock_profile_evaluator.evaluate_profile.return_value).copy()
        eval_copy["cluster_context"] = None
        expl = engine._generate_explanation("docker", 0.85, eval_copy)
        assert "🔴" in expl

    def test_explanation_medium_score(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        expl = engine._generate_explanation("fastapi", 0.45, _unwrap(mock_profile_evaluator.evaluate_profile.return_value))
        assert "🟡" in expl

    def test_explanation_low_score(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        expl = engine._generate_explanation("html", 0.2, _unwrap(mock_profile_evaluator.evaluate_profile.return_value))
        assert "🟢" in expl


# ---------------------------------------------------------------------------
# Tests for _is_hard_skill
# ---------------------------------------------------------------------------
class TestHardSkillDetection:
    def test_is_hard_skill_true(self, mock_profile_evaluator, mock_skill_filter):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.skill_filter = mock_skill_filter
        # docker – обычно в taxonomy hard, но у нас замокана пустая, поэтому проверка по _hard_keywords
        engine._hard_keywords = {"docker"}
        assert engine._is_hard_skill("docker") is True

    def test_is_soft_skill(self, mock_profile_evaluator, mock_skill_filter):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        engine.skill_filter = mock_skill_filter
        engine._hard_keywords = set()
        assert engine._is_hard_skill("communication") is False


# ---------------------------------------------------------------------------
# Tests for _get_timeframe
# ---------------------------------------------------------------------------
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

    def test_get_timeframe_default(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert engine._get_timeframe("unknown") == "1-3 месяца"


# ---------------------------------------------------------------------------
# Tests for _get_learning_path
# ---------------------------------------------------------------------------
class TestLearningPaths:
    def test_get_learning_path_junior(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        profile = StudentProfile(profile_name="j", competencies=[], skills=["python"], target_level="junior", created_at=datetime.now())
        path = engine._get_learning_path("python", False, profile)
        assert "Сфокусируйтесь на основах" in path

    def test_get_learning_path_senior(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        profile = StudentProfile(profile_name="s", competencies=[], skills=["python"], target_level="senior", created_at=datetime.now())
        path = engine._get_learning_path("docker", False, profile)
        assert "Углублённое изучение" in path

    def test_get_learning_path_default(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        path = engine._get_learning_path("unknown", False, None)
        assert "Изучите документацию" in path


# ---------------------------------------------------------------------------
# Tests for _build_closest_roles
# ---------------------------------------------------------------------------
class TestClosestRoles:
    def test_build_closest_roles_empty(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        roles = engine._build_closest_roles([], {}, set())
        assert roles == []

    def test_build_closest_roles_with_data(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        # замокаем clusterer, чтобы избежать обращения к реальному объекту
        mock_profile_evaluator.clusterer = None
        closest_clusters = [{"id": 0, "name": "Backend", "similarity": 0.8}]
        cluster_skills_map = {"python": 1.0, "docker": 0.9}
        student_set = {"python"}
        roles = engine._build_closest_roles(closest_clusters, cluster_skills_map, student_set)
        assert len(roles) == 1
        assert roles[0]["role"] == "Backend"


# ---------------------------------------------------------------------------
# Tests for _diversify_recommendations
# ---------------------------------------------------------------------------
class TestDiversify:
    def test_diversify(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        recs = [
            {"skill": "a", "category": "missing", "importance_score": 0.9},
            {"skill": "b", "category": "missing", "importance_score": 0.8},
            {"skill": "c", "category": "missing", "importance_score": 0.7},
            {"skill": "d", "category": "weak", "importance_score": 0.6},
        ]
        result = engine._diversify_recommendations(recs, max_per_category=2)
        # first 2 missing + 1 weak + then leftover
        assert result[0]["skill"] == "a"
        assert result[1]["skill"] == "b"
        assert result[2]["skill"] == "d"  # weak
        assert result[3]["skill"] == "c"  # leftover


# ---------------------------------------------------------------------------
# Tests for _get_role_outcome
# ---------------------------------------------------------------------------
class TestRoleOutcome:
    def test_get_role_outcome_with_roles(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        roles = [{
            "role": "Backend",
            "semantic_similarity": 80.0,
            "coverage_percent": 50.0,
            "skills_covered": "3/6",
        }]
        outcome = engine._get_role_outcome("docker", roles)
        assert "docker" in outcome
        assert "Backend" in outcome

    def test_get_role_outcome_no_roles(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        outcome = engine._get_role_outcome("docker", [])
        assert "расширит ваш технический кругозор" in outcome


# ---------------------------------------------------------------------------
# Tests for LLM
# ---------------------------------------------------------------------------
class TestLLM:
    def test_llm_disabled(self, mock_profile_evaluator):
        engine = RecommendationEngine(use_llm=False, profile_evaluator=mock_profile_evaluator)
        assert engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0) is None

    def test_llm_success(self, mock_profile_evaluator, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", SecretStr("test-key"))
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"alternatives": [{"message": {"text": "Ответ от LLM"}}]}}
        with patch("requests.post", return_value=mock_resp):
            result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0)
            assert result == "Ответ от LLM"

    def test_llm_429_retry(self, mock_profile_evaluator, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", SecretStr("test-key"))
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)
        resp1 = MagicMock(status_code=429)
        resp2 = MagicMock(status_code=200)
        resp2.json.return_value = {"result": {"alternatives": [{"message": {"text": "После retry"}}]}}
        with patch("requests.post", side_effect=[resp1, resp2]), patch("time.sleep", return_value=None):
            result = engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0)
            assert result == "После retry"

    def test_llm_error_status(self, mock_profile_evaluator, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", SecretStr("test-key"))
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)
        resp = MagicMock(status_code=500, text="Error")
        with patch("requests.post", return_value=resp):
            assert engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0) is None

    def test_llm_exception(self, mock_profile_evaluator, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", SecretStr("test-key"))
        monkeypatch.setattr(config, "YC_FOLDER_ID", "test-folder")
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)
        with patch("requests.post", side_effect=Exception("Сеть")):
            assert engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0) is None


# ---------------------------------------------------------------------------
# Additional coverage for init file loads
# ---------------------------------------------------------------------------
class TestInitFileLoads:
    def test_load_hard_keywords(self, tmp_path, monkeypatch, mock_profile_evaluator):
        hard_file = tmp_path / "hard_skills.json"
        hard_file.write_text(json.dumps(["docker", "kubernetes"]))
        monkeypatch.setattr(config, "HARD_SKILLS_PATH", hard_file)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator, use_ltr=False)
        assert engine._hard_keywords == {"docker", "kubernetes"}

    def test_load_hot_skills(self, tmp_path, monkeypatch, mock_profile_evaluator):
        hot_file = tmp_path / "hot_skills.json"
        hot_file.write_text(json.dumps(["python", "sql"]))
        monkeypatch.setattr(config, "TREND_HOT_SKILLS_PATH", hot_file)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator, use_ltr=False)
        assert engine._always_hot == {"python", "sql"}

    def test_load_timeframe_groups(self, tmp_path, monkeypatch, mock_profile_evaluator):
        tf_file = tmp_path / "timeframe.json"
        tf_file.write_text(json.dumps({"easy": ["git"], "medium": ["python"], "hard": ["java"]}))
        monkeypatch.setattr(config, "TIMEFRAME_GROUPS_PATH", tf_file)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator, use_ltr=False)
        assert engine._timeframe_easy == {"git"}

    def test_load_hard_keywords_missing(self, monkeypatch, mock_profile_evaluator):
        monkeypatch.setattr(config, "HARD_SKILLS_PATH", Path("/nonexistent.json"))
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator, use_ltr=False)
        assert engine._hard_keywords == set()

class TestRecommendationEngineExtended:
    """Тесты для активации пропущенных веток в RecommendationEngine."""

    def test_generate_with_ltr_active(self, mock_profile_evaluator, sample_student_profile, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        engine = RecommendationEngine(use_ltr=True, profile_evaluator=mock_profile_evaluator)
        engine._always_hot = set()      # убираем реальные горячие навыки
        mock_ltr = MagicMock()
        mock_ltr.is_fitted = True
        mock_ltr.skill_metadata = {"docker": {}, "fastapi": {}, "k8s": {}}
        mock_ltr.predict_impact.return_value = [
            __import__("src.predictors.models", fromlist=["SkillImpact"]).SkillImpact(skill="docker", score=90, explanation=""),
            __import__("src.predictors.models", fromlist=["SkillImpact"]).SkillImpact(skill="fastapi", score=70, explanation=""),
        ]
        engine.ltr_engine = mock_ltr

        match engine.generate_recommendations(sample_student_profile):
            case Ok(result):
                assert len(result.recommendations) > 0
                assert result.trend_bonuses_count == 0

    def test_generate_with_trends(self, mock_profile_evaluator, sample_student_profile, tmp_path):
        mock_trends = MagicMock()
        mock_trends.get_trending_skills.return_value = {
            "rising": [{"skill": "docker", "change_pct": 20.0}]
        }
        mock_trends.save_trends.return_value = None
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator, trend_analyzer=mock_trends)
        engine._always_hot = set()
        match engine.generate_recommendations(sample_student_profile):
            case Ok(result):
                assert result.trend_bonuses_count == 1

    def test_generate_with_domain_bonus(self, mock_profile_evaluator, sample_student_profile, tmp_path):
        """Доменный бонус: добавляем domain_analyzer с мапой навыков."""
        mock_profile_evaluator.domain_analyzer = MagicMock()
        mock_profile_evaluator.domain_analyzer.domain_map = {"Backend": ["docker", "fastapi"]}
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        match engine.generate_recommendations(sample_student_profile):
            case Ok(result):
                recs = result.recommendations
                assert any("🔗" in r.why_important for r in recs)

    def test_generate_eval_result_none_returns_err(self, mock_profile_evaluator, sample_student_profile):
        mock_profile_evaluator.evaluate_profile.return_value = None
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile)
        assert result.is_err()

    def test_generate_crash_returns_err(self, mock_profile_evaluator, sample_student_profile):
        mock_profile_evaluator.evaluate_profile.side_effect = Exception("unexpected error")
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        result = engine.generate_recommendations(sample_student_profile)
        assert result.is_err()

    def test_diversify_with_many_categories(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        recs = [
            {"skill": f"skill_{i}", "category": f"cat_{i%4}", "importance_score": 1.0 - i*0.1}
            for i in range(20)
        ]
        result = engine._diversify_recommendations(recs, max_per_category=2)
        assert len(result) == len(recs)
        # Проверяем, что в приоритетной части (первые 8 элементов) каждая категория ≤ 2
        priority_part = result[:8]
        cats_in_priority = [r["category"] for r in priority_part]
        for cat in set(cats_in_priority):
            assert cats_in_priority.count(cat) <= 2, f"Category {cat} appears too many times in priority"

    def test_get_role_outcome_no_roles(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        outcome = engine._get_role_outcome("docker", [])
        assert "расширит" in outcome

    def test_get_role_outcome_with_roles(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        roles = [{"role": "Backend", "semantic_similarity": 80.0, "coverage_percent": 50.0,
                  "skills_covered": "3/6"}]
        outcome = engine._get_role_outcome("docker", roles)
        assert "Backend" in outcome

    def test_llm_explain_with_retry_without_credentials(self, mock_profile_evaluator, monkeypatch):
        monkeypatch.setattr(config, "YC_API_KEY", None)
        engine = RecommendationEngine(use_llm=True, profile_evaluator=mock_profile_evaluator)
        assert engine._llm_explain_with_retry("docker", 0.8, "HIGH", ["python"], 70.0) is None

    def test_load_templates_corrupted_file(self, tmp_path, monkeypatch, mock_profile_evaluator):
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "recommendation_templates.json").write_text("{invalid")
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        assert "python" in engine.HARD_LEARNING_PATHS

    def test_empty_recommendations_structure(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        empty = engine._empty_recommendations()
        assert empty["summary"]["readiness_score"] == 0
        assert empty["recommendations"] == []

    def test_generate_saves_ltr_debug_file(self, mock_profile_evaluator, sample_student_profile, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        match engine.generate_recommendations(sample_student_profile):
            case Ok(_):
                ltr_file = tmp_path / "result" / sample_student_profile.profile_name / f"ltr_recommendations_{sample_student_profile.profile_name}.json"
                assert ltr_file.exists()
        with open(ltr_file, encoding="utf-8") as f:   # <-- добавлен encoding
            data = json.load(f)
        assert data["profile"] == sample_student_profile.profile_name

    def test_generate_ltr_save_exception(self, mock_profile_evaluator, sample_student_profile, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        with patch("src.utils.atomic_write_json", side_effect=Exception("disk full")):
            match engine.generate_recommendations(sample_student_profile):
                case Ok(result):
                    assert len(result.recommendations) > 0

    def test_explanation_taxonomy_exception(self, mock_profile_evaluator):
        engine = RecommendationEngine(profile_evaluator=mock_profile_evaluator)
        eval_copy = _unwrap(mock_profile_evaluator.evaluate_profile.return_value).copy()
        eval_copy["cluster_context"] = None               # убираем кластер
        eval_copy["skill_metrics"] = {"docker": {"cluster_relevance": 0, "category": "missing"}}
        engine._taxonomy.get_category_label = MagicMock(side_effect=Exception("taxonomy error"))
        expl = engine._generate_explanation("docker", 0.8, eval_copy)
        assert "технический" in expl
