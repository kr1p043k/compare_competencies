# tests/api/test_api.py
"""Smoke-тесты публичных GET-эндпоинтов через dependency_overrides.

Паттерн: моки подсовываются через app.dependency_overrides[deps.get_*],
а не через глобалы src.api_pkg (роутеры читают зависимости через deps).
"""
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Предотвращаем импорт shap и cv2
sys.modules["shap"] = MagicMock()
sys.modules["cv2"] = MagicMock()

# NOTE: sentence_transformers is mocked session-wide by tests/conftest.py
# (configured mock with working encode). Do NOT overwrite sys.modules here:
# a bare MagicMock breaks every other test file importing embeddings
# (proven: 15 comparator failures). shap/cv2 mocks below are load-bearing
# (packages not installed) and harmless (additive, nothing overwritten).
from src import Err, Ok
from src.api_pkg import app
from src.api_pkg import deps
from src.models.student import StudentProfile

client = TestClient(app)


def _profile():
    return StudentProfile(
        profile_name="base", competencies=[], skills=["python", "sql"],
        target_level="junior",
    )


@pytest.fixture(autouse=True)
def clean_overrides():
    """Чистые dependency_overrides + живой current_skills_set на каждый тест."""
    from tests.conftest import open_all_gates
    app.dependency_overrides.clear()
    open_all_gates(app)
    old_skills = deps.current_skills_set
    yield
    app.dependency_overrides.clear()
    deps.current_skills_set = old_skills


class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_ready(self):
        r = client.get("/ready")
        assert r.status_code == 200

    def test_api_health(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestRecommendations:
    def test_existing(self):
        rec = MagicMock()
        rec.model_dump.return_value = {"summary": {}, "recommendations": []}
        engine = MagicMock()
        engine.generate_recommendations.return_value = Ok(rec)
        evaluator = MagicMock()
        evaluator.evaluate_profile.return_value = Ok({"market_coverage_score": 70})
        app.dependency_overrides[deps.get_recommendation_engine] = lambda: engine
        app.dependency_overrides[deps.get_evaluator] = lambda: evaluator
        app.dependency_overrides[deps.get_student_profiles] = lambda: {"base": _profile()}
        r = client.get("/api/recommendations/base")
        assert r.status_code == 200

    def test_missing(self):
        engine = MagicMock()
        app.dependency_overrides[deps.get_recommendation_engine] = lambda: engine
        app.dependency_overrides[deps.get_evaluator] = lambda: MagicMock()
        app.dependency_overrides[deps.get_student_profiles] = lambda: {"base": _profile()}
        r = client.get("/api/recommendations/nobody")
        assert r.status_code == 404


class TestMarket:
    def test_top_skills(self):
        app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 0.9, "sql": 0.7}
        app.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 100, "sql": 80}
        r = client.get("/api/market/top-skills?limit=2")
        assert r.status_code == 200
        data = r.json()
        assert len(data["skills"]) == 2
        assert data["skills"][0]["skill"] == "python"

    def test_skill_info(self):
        taxonomy = MagicMock()
        taxonomy.get_category_label.return_value = "Lang"
        taxonomy.get_category_icon.return_value = "icon"
        app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 0.9}
        app.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 100}
        app.dependency_overrides[deps.get_taxonomy] = lambda: taxonomy
        r = client.get("/api/market/skill/python")
        assert r.status_code == 200
        assert r.json()["skill"] == "python"
        assert r.json()["category"] == "Lang"

    def test_skill_info_no_taxonomy(self):
        app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 0.9}
        app.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 100}
        app.dependency_overrides[deps.get_taxonomy] = lambda: None
        r = client.get("/api/market/skill/python")
        assert r.status_code == 200
        assert r.json()["category"] == "unknown"


def _clusterer_mock(*, fitted=True, n=2):
    inst = MagicMock()
    inst.load_model.return_value = True
    inst.is_fitted = fitted
    inst.n_clusters_ = n
    inst.clusterer_type = "kmeans"
    inst._generate_cluster_name.side_effect = lambda cid: f"Cluster{cid}"
    inst.get_top_skills_in_cluster.return_value = ["a", "b"]
    return inst


class TestClusters:
    def test_cluster_by_level(self):
        with patch("src.api_pkg.routers.clusters.VacancyClusterer",
                   return_value=_clusterer_mock(fitted=True, n=2)):
            r = client.get("/api/clusters/junior")
        assert r.status_code == 200
        data = r.json()
        assert data["level"] == "junior"
        assert len(data["clusters"]) == 2

    def test_cluster_not_loaded(self):
        with patch("src.api_pkg.routers.clusters.VacancyClusterer",
                   return_value=_clusterer_mock(fitted=False)):
            r = client.get("/api/clusters/senior")
        assert r.status_code == 503

    def test_clusters_summary(self):
        with patch("src.api_pkg.routers.clusters.VacancyClusterer",
                   return_value=_clusterer_mock(fitted=True, n=3)):
            r = client.get("/api/clusters/summary")
        assert r.status_code == 200
        data = r.json()
        for lvl in ["junior", "middle", "senior"]:
            assert lvl in data
            assert len(data[lvl]["top_clusters"]) == 3

    def test_clusters_summary_unfitted(self):
        with patch("src.api_pkg.routers.clusters.VacancyClusterer",
                   return_value=_clusterer_mock(fitted=False)):
            r = client.get("/api/clusters/summary")
        assert r.status_code == 200
        data = r.json()
        for lvl in ["junior", "middle", "senior"]:
            assert data[lvl] == {"error": "not_fitted"}


class TestProfilesCompare:
    def test_compare(self):
        evaluator = MagicMock()
        evaluator.evaluate_profile.return_value = Ok({
            "market_coverage_score": 70, "skill_coverage": 60,
            "domain_coverage_score": 50, "readiness_score": 65,
            "market_skill_coverage": 40,
        })
        app.dependency_overrides[deps.get_evaluator] = lambda: evaluator
        app.dependency_overrides[deps.get_student_profiles] = lambda: {"base": _profile()}
        r = client.get("/api/profiles/compare")
        assert r.status_code == 200
        body = r.json()["profiles"]
        assert "base" in body
        assert body["base"]["readiness_score"] == 65
        assert body["base"]["real_coverage"] == 40

    def test_compare_error(self):
        evaluator = MagicMock()
        evaluator.evaluate_profile.return_value = Err("fail")
        app.dependency_overrides[deps.get_evaluator] = lambda: evaluator
        app.dependency_overrides[deps.get_student_profiles] = lambda: {"base": _profile()}
        r = client.get("/api/profiles/compare")
        assert r.status_code == 200
        assert "error" in r.json()["profiles"]["base"]


class TestTrends:
    def test_trends(self):
        analyzer = MagicMock()
        analyzer.get_trending_skills.return_value = Ok({"rising": [], "falling": []})
        app.dependency_overrides[deps.get_trend_analyzer] = lambda: analyzer
        r = client.get("/api/trends")
        assert r.status_code == 200
        assert r.json()["trends"] == {"rising": [], "falling": []}

    def test_trends_error(self):
        analyzer = MagicMock()
        analyzer.get_trending_skills.side_effect = Exception("boom")
        app.dependency_overrides[deps.get_trend_analyzer] = lambda: analyzer
        r = client.get("/api/trends")
        assert r.status_code == 500


class TestTaxonomyCoverage:
    def test_coverage(self):
        taxonomy = MagicMock()
        taxonomy.get_all_categories.return_value = Ok(["cat1"])
        taxonomy.get_skills_in_category.return_value = Ok(["python", "java"])
        taxonomy.get_category_label_by_id.return_value = "Test"
        taxonomy.get_category_icon_by_id.return_value = "icon"
        app.dependency_overrides[deps.get_taxonomy] = lambda: taxonomy
        deps.current_skills_set = {"python"}
        r = client.get("/api/taxonomy/coverage")
        assert r.status_code == 200
        assert r.json()["coverage"]["cat1"]["covered"] == 1

    def test_no_taxonomy(self):
        app.dependency_overrides[deps.get_taxonomy] = lambda: None
        r = client.get("/api/taxonomy/coverage")
        assert r.status_code == 503


class TestSkills:
    def test_missing(self):
        app.dependency_overrides[deps.get_skill_freq] = lambda: {"docker": 5, "k8s": 3}
        deps.current_skills_set = {"python"}
        with patch("src.api_pkg.routers.profiles.SkillValidator") as mock_validator:
            mock_validator.return_value.validate.return_value = Ok(
                MagicMock(is_valid=True)
            )
            r = client.get("/api/skills/missing")
        assert r.status_code == 200
        skills = r.json()["missing_skills"]
        assert len(skills) == 2
        assert skills[0]["skill"] == "docker"

    def test_dead(self):
        app.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 10}
        deps.current_skills_set = {"python", "sql"}
        r = client.get("/api/skills/dead")
        assert r.status_code == 200
        assert "sql" in r.json()["dead_skills"]
