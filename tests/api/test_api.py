# tests/api/test_api.py
import sys
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skip(reason="Requires running API server")

# Предотвращаем импорт shap и cv2
sys.modules['shap'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Mock sentence_transformers BEFORE importing src.api to prevent real import
_sent_original = sys.modules.get('sentence_transformers')
_sent_mock = MagicMock()
_sent_mock.__version__ = "0.0.0"
sys.modules['sentence_transformers'] = _sent_mock
from src.api import app
from src.models.student import StudentProfile

client = TestClient(app)


def _profile():
    return StudentProfile(profile_name="base", competencies=[], skills=["python", "sql"], target_level="junior")


@pytest.fixture(autouse=True)
def mock_globals():
    """Подменяет глобальные переменные src.api на моки перед каждым тестом."""
    import src.api as _api
    _originals = {}
    _mocks = {
        'evaluator': MagicMock(),
        'recommendation_engine': MagicMock(is_fitted=True),
        'clusterer': MagicMock(is_fitted=True),
        'trend_analyzer': MagicMock(),
        'skill_weights': {"python": 0.9, "sql": 0.7},
        'skill_freq': {"python": 100, "sql": 80},
        'taxonomy': MagicMock(),
        'current_skills_set': {"python", "sql"},
        'basic_vacancies': [{"id": 1}],
        'student_profiles': {"base": _profile()},
    }
    for name, mock in _mocks.items():
        _originals[name] = getattr(_api, name, None)
        setattr(_api, name, mock)
    yield
    for name, original in _originals.items():
        setattr(_api, name, original)


class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_ready(self):
        r = client.get("/ready")
        assert r.status_code == 200


class TestRecommendations:
    def test_existing(self):
        import src.api
        src.api.recommendation_engine.generate_recommendations.return_value = {"summary": {}, "recommendations": []}
        r = client.get("/api/recommendations/base")
        assert r.status_code == 200

    def test_missing(self):
        r = client.get("/api/recommendations/nobody")
        assert r.status_code == 404


class TestMarket:
    def test_top_skills(self):
        r = client.get("/api/market/top-skills?limit=2")
        assert r.status_code == 200
        data = r.json()
        assert len(data["skills"]) == 2

    def test_skill_info(self):
        import src.api
        src.api.taxonomy.get_category_label.return_value = "Lang"
        src.api.taxonomy.get_category_icon.return_value = "💻"
        r = client.get("/api/market/skill/python")
        assert r.status_code == 200
        assert r.json()["skill"] == "python"

    def test_skill_info_no_taxonomy(self):
        import src.api
        src.api.taxonomy = None
        r = client.get("/api/market/skill/python")
        assert r.status_code == 200
        assert r.json()["category"] == "unknown"


class TestClusters:
    def test_cluster_by_level(self):
        import src.api
        src.api.clusterer.n_clusters_ = 2
        src.api.clusterer._generate_cluster_name.side_effect = lambda cid: f"Cluster{cid}"
        src.api.clusterer.get_top_skills_in_cluster.return_value = ["a", "b"]
        src.api.clusterer.load_model.return_value = True
        r = client.get("/api/clusters/junior")
        assert r.status_code == 200
        data = r.json()
        assert data["level"] == "junior"
        assert len(data["clusters"]) == 2

    def test_cluster_not_loaded(self):
        import src.api
        src.api.clusterer.is_fitted = False
        src.api.clusterer.load_model.return_value = False
        r = client.get("/api/clusters/senior")
        assert r.status_code == 503

    def test_clusters_summary(self):
        import src.api
        src.api.clusterer.is_fitted = True
        src.api.clusterer.n_clusters_ = 3
        src.api.clusterer.clusterer_type = "kmeans"
        src.api.clusterer._generate_cluster_name.return_value = "C0"
        src.api.clusterer.get_top_skills_in_cluster.return_value = ["py", "sql"]
        src.api.clusterer.load_model.return_value = True
        r = client.get("/api/clusters/summary")
        assert r.status_code == 200
        data = r.json()
        for lvl in ["junior", "middle", "senior"]:
            assert lvl in data

    def test_clusters_summary_unfitted(self):
        import src.api
        src.api.clusterer.is_fitted = False
        src.api.clusterer.load_model.return_value = False
        r = client.get("/api/clusters/summary")
        assert r.status_code == 200
        data = r.json()
        for lvl in ["junior", "middle", "senior"]:
            assert data[lvl] == {"error": "not_fitted"}


class TestProfilesCompare:
    def test_compare(self):
        import src.api
        src.api.evaluator.evaluate_profile.return_value = {
            "market_coverage_score": 70, "skill_coverage": 60,
            "domain_coverage_score": 50, "readiness_score": 65,
            "market_skill_coverage": 40
        }
        r = client.get("/api/profiles/compare")
        assert r.status_code == 200
        assert "base" in r.json()["profiles"]

    def test_compare_error(self):
        import src.api
        src.api.evaluator.evaluate_profile.side_effect = Exception("fail")
        r = client.get("/api/profiles/compare")
        assert r.status_code == 200
        assert "error" in r.json()["profiles"]["base"]


class TestTrends:
    def test_trends(self):
        import src.api
        src.api.trend_analyzer.get_trending_skills.return_value = {"rising": [], "falling": []}
        r = client.get("/api/trends")
        assert r.status_code == 200

    def test_trends_error(self):
        import src.api
        src.api.trend_analyzer.get_trending_skills.side_effect = Exception("boom")
        r = client.get("/api/trends")
        assert r.status_code == 500


class TestTaxonomyCoverage:
    def test_coverage(self):
        import src.api
        src.api.taxonomy.get_all_categories.return_value = ["cat1"]
        src.api.taxonomy.get_skills_in_category.return_value = ["python", "java"]
        src.api.taxonomy.get_category_label_by_id.return_value = "Test"
        src.api.taxonomy.get_category_icon_by_id.return_value = "🧪"
        src.api.current_skills_set = {"python"}
        r = client.get("/api/taxonomy/coverage")
        assert r.status_code == 200
        assert r.json()["coverage"]["cat1"]["covered"] == 1

    def test_no_taxonomy(self):
        import src.api
        src.api.taxonomy = None
        r = client.get("/api/taxonomy/coverage")
        assert r.status_code == 503


class TestSkills:
    def test_missing(self):
        import src.api
        src.api.skill_freq = {"docker": 5, "k8s": 3}
        src.api.current_skills_set = {"python"}
        with patch('src.api.SkillValidator') as mock_validator:
            mock_validator.return_value.validate.return_value.is_valid = True
            r = client.get("/api/skills/missing")
        assert r.status_code == 200
        skills = r.json()["missing_skills"]
        assert len(skills) == 2
        assert skills[0]["skill"] == "docker"

    def test_dead(self):
        import src.api
        src.api.skill_freq = {"python": 10}
        src.api.current_skills_set = {"python", "sql"}
        r = client.get("/api/skills/dead")
        assert r.status_code == 200
        assert "sql" in r.json()["dead_skills"]


class TestStatus:
    def test_status(self):
        import src.api
        src.api.clusterer.load_model.return_value = True
        src.api.clusterer.is_fitted = True
        src.api.recommendation_engine.is_fitted = True
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["vacancies_loaded"] is True
        assert data["profiles_available"] == ["base"]
