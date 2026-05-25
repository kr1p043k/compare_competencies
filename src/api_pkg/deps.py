"""Dependency injection — глобальное состояние и фабрики для роутеров."""

from fastapi import HTTPException
from src.analyzers.clustering.vacancy_clustering import VacancyClusterer
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
from src.analyzers.skills.trends import TrendAnalyzer
from src.models.student import StudentProfile
from src.predictors.recommendation_engine import RecommendationEngine

evaluator: ProfileEvaluator | None = None
recommendation_engine: RecommendationEngine | None = None
clusterer: VacancyClusterer = VacancyClusterer()
trend_analyzer: TrendAnalyzer | None = None
student_profiles: dict[str, StudentProfile] = {}
skill_weights: dict[str, float] = {}
hybrid_weights: dict[str, float] = {}
competency_mapping: dict[str, list[str]] = {}
skill_freq: dict[str, int] = {}
taxonomy: SkillTaxonomy | None = None
current_skills_set: set[str] = set()
basic_vacancies: list = []
vacancy_load_error: str | None = None
is_ready: bool = False
_regions_cache: list[str] = []
_regions_cache_time: float = 0


def get_evaluator() -> ProfileEvaluator:
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Profile evaluator not initialized")
    return evaluator


def get_recommendation_engine() -> RecommendationEngine:
    if recommendation_engine is None:
        raise HTTPException(
            status_code=503, detail="Recommendation engine not initialized"
        )
    return recommendation_engine


def get_clusterer() -> VacancyClusterer:
    return clusterer


def get_trend_analyzer() -> TrendAnalyzer:
    if trend_analyzer is None:
        raise HTTPException(status_code=503, detail="Trend analyzer not initialized")
    return trend_analyzer


def get_taxonomy() -> SkillTaxonomy | None:
    return taxonomy


def get_basic_vacancies() -> list:
    if not basic_vacancies:
        raise HTTPException(status_code=503, detail="Vacancy data not loaded yet")
    return basic_vacancies


def get_student_profiles() -> dict[str, StudentProfile]:
    return student_profiles


def get_skill_weights() -> dict[str, float]:
    return skill_weights


def get_skill_freq() -> dict[str, int]:
    return skill_freq


def get_hybrid_weights() -> dict[str, float]:
    return hybrid_weights


def validate_regions(regions: list[str]) -> list[str]:
    global _regions_cache
    if not _regions_cache:
        raise HTTPException(status_code=503, detail="Regions data not loaded yet")
    if "Все регионы" in regions:
        return regions
    invalid_regions = [r for r in regions if r not in _regions_cache]
    if invalid_regions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regions: {', '.join(invalid_regions)}. Use /api/regions to get valid regions.",
        )
    return regions
