"""Тесты для src.api_pkg.deps — глобальное состояние и фабрики."""

import pytest
from fastapi import HTTPException

from src.api_pkg import deps
from src.analyzers.clustering.vacancy_clustering import VacancyClusterer
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
from src.analyzers.skills.trends import TrendAnalyzer
from src.models.student import StudentProfile
from src.predictors.recommendation_engine import RecommendationEngine


class TestModuleLevelVariables:
    def test_evaluator_is_none_by_default(self):
        assert deps.evaluator is None

    def test_recommendation_engine_is_none_by_default(self):
        assert deps.recommendation_engine is None

    def test_clusterer_is_instance(self):
        assert isinstance(deps.clusterer, VacancyClusterer)

    def test_trend_analyzer_is_none_by_default(self):
        assert deps.trend_analyzer is None

    def test_student_profiles_is_dict(self):
        assert isinstance(deps.student_profiles, dict)

    def test_skill_weights_is_dict(self):
        assert isinstance(deps.skill_weights, dict)

    def test_hybrid_weights_is_dict(self):
        assert isinstance(deps.hybrid_weights, dict)

    def test_competency_mapping_is_dict(self):
        assert isinstance(deps.competency_mapping, dict)

    def test_skill_freq_is_dict(self):
        assert isinstance(deps.skill_freq, dict)

    def test_taxonomy_is_none_by_default(self):
        assert deps.taxonomy is None

    def test_current_skills_set_is_set(self):
        assert isinstance(deps.current_skills_set, set)

    def test_basic_vacancies_is_list(self):
        assert isinstance(deps.basic_vacancies, list)

    def test_vacancy_load_error_is_none(self):
        assert deps.vacancy_load_error is None

    def test_is_ready_is_false(self):
        assert deps.is_ready is False

    def test_regions_cache_is_list(self):
        assert isinstance(deps._regions_cache, list)

    def test_regions_cache_time_is_numeric(self):
        assert isinstance(deps._regions_cache_time, (int, float))


class TestGetEvaluator:
    def test_raises_when_none(self, monkeypatch):
        monkeypatch.setattr(deps, "evaluator", None)
        with pytest.raises(HTTPException) as exc:
            deps.get_evaluator()
        assert exc.value.status_code == 503
        assert "not initialized" in exc.value.detail

    def test_returns_evaluator(self, monkeypatch):
        mock = ProfileEvaluator(
            skill_weights={"python": 10},
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
        )
        monkeypatch.setattr(deps, "evaluator", mock)
        assert deps.get_evaluator() is mock


class TestGetRecommendationEngine:
    def test_raises_when_none(self, monkeypatch):
        monkeypatch.setattr(deps, "recommendation_engine", None)
        with pytest.raises(HTTPException) as exc:
            deps.get_recommendation_engine()
        assert exc.value.status_code == 503
        assert "not initialized" in exc.value.detail

    def test_returns_engine(self, monkeypatch):
        mock = RecommendationEngine(use_ltr=True, use_llm=False)
        monkeypatch.setattr(deps, "recommendation_engine", mock)
        assert deps.get_recommendation_engine() is mock


class TestGetClusterer:
    def test_returns_clusterer(self):
        assert deps.get_clusterer() is deps.clusterer


class TestGetTrendAnalyzer:
    def test_raises_when_none(self, monkeypatch):
        monkeypatch.setattr(deps, "trend_analyzer", None)
        with pytest.raises(HTTPException) as exc:
            deps.get_trend_analyzer()
        assert exc.value.status_code == 503
        assert "not initialized" in exc.value.detail

    def test_returns_analyzer(self, monkeypatch):
        mock = TrendAnalyzer({})
        monkeypatch.setattr(deps, "trend_analyzer", mock)
        assert deps.get_trend_analyzer() is mock


class TestGetTaxonomy:
    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.setattr(deps, "taxonomy", None)
        assert deps.get_taxonomy() is None

    def test_returns_taxonomy(self, monkeypatch):
        mock = SkillTaxonomy()
        monkeypatch.setattr(deps, "taxonomy", mock)
        assert deps.get_taxonomy() is mock


class TestGetBasicVacancies:
    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setattr(deps, "basic_vacancies", [])
        with pytest.raises(HTTPException) as exc:
            deps.get_basic_vacancies()
        assert exc.value.status_code == 503
        assert "not loaded" in exc.value.detail

    def test_returns_vacancies(self, monkeypatch):
        monkeypatch.setattr(deps, "basic_vacancies", [{"id": "1"}])
        assert deps.get_basic_vacancies() == [{"id": "1"}]


class TestGetStudentProfiles:
    def test_returns_student_profiles(self, monkeypatch):
        monkeypatch.setattr(deps, "student_profiles", {"base": None})
        assert deps.get_student_profiles() == {"base": None}


class TestGetSkillWeights:
    def test_returns_skill_weights(self, monkeypatch):
        monkeypatch.setattr(deps, "skill_weights", {"python": 10.0})
        assert deps.get_skill_weights() == {"python": 10.0}


class TestGetSkillFreq:
    def test_returns_skill_freq(self, monkeypatch):
        monkeypatch.setattr(deps, "skill_freq", {"python": 100})
        assert deps.get_skill_freq() == {"python": 100}


class TestGetHybridWeights:
    def test_returns_hybrid_weights(self, monkeypatch):
        monkeypatch.setattr(deps, "hybrid_weights", {"python": 0.9})
        assert deps.get_hybrid_weights() == {"python": 0.9}


class TestValidateRegions:
    def test_raises_when_cache_empty(self, monkeypatch):
        monkeypatch.setattr(deps, "_regions_cache", [])
        with pytest.raises(HTTPException) as exc:
            deps.validate_regions(["Москва"])
        assert exc.value.status_code == 503

    def test_returns_all_regions_when_vse_regiony(self, monkeypatch):
        monkeypatch.setattr(deps, "_regions_cache", ["Москва"])
        result = deps.validate_regions(["Все регионы", "Москва"])
        assert result == ["Все регионы", "Москва"]

    def test_raises_for_invalid_regions(self, monkeypatch):
        monkeypatch.setattr(deps, "_regions_cache", ["Москва", "СПб"])
        with pytest.raises(HTTPException) as exc:
            deps.validate_regions(["Москва", "Казань"])
        assert exc.value.status_code == 400
        assert "Казань" in exc.value.detail

    def test_returns_valid_regions(self, monkeypatch):
        monkeypatch.setattr(deps, "_regions_cache", ["Москва", "СПб"])
        result = deps.validate_regions(["Москва"])
        assert result == ["Москва"]
