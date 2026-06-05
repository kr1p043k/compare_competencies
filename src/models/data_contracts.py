"""
Строгие Pydantic-модели для контрактов между этапами пайплайна.
Bounded‑context модели: CollectionContext / AnalysisContext / RecommendationContext.
"""

from typing import Any

from pydantic import BaseModel, Field


class WeightedSkill(BaseModel):
    name: str
    weight: float
    category: str = "other"


class LevelVacancy(BaseModel):
    skills: list[str]
    description: str = ""
    experience: str = "middle"


# ─── Collection (сбор и очистка вакансий) ──────────────────────────────
class CollectionContext(BaseModel):
    """Результаты сбора и фильтрации вакансий."""

    raw_vacancies: list[Any] = Field(default_factory=list)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    skip_collection: bool = False

    class Config:
        arbitrary_types_allowed = True


# ─── Analysis (извлечение навыков, веса, уровни, кластеры, модель) ─────
class AnalysisContext(BaseModel):
    """Извлечённые навыки, веса, уровни, кластеры, LTR-модель."""

    skill_freq: dict[str, int] = Field(default_factory=dict)
    hybrid_weights: dict[str, float] = Field(default_factory=dict)
    vacancies_skills: list[list[str]] = Field(default_factory=list)
    level_vacancies_data: list[LevelVacancy] = Field(default_factory=list)
    trend_analyzer: Any = None
    clusters_trained: bool = False
    model_trained: bool = False

    class Config:
        arbitrary_types_allowed = True


# ─── Recommendation (gap-анализ + рекомендации) ───────────────────────
class RecommendationContext(BaseModel):
    """Профили, оценка, пробелы, рекомендации."""

    skill_freq: dict[str, int] = Field(default_factory=dict)
    hybrid_weights: dict[str, float] = Field(default_factory=dict)
    vacancies_skills: list[list[str]] = Field(default_factory=list)
    level_vacancies_data: list[LevelVacancy] = Field(default_factory=list)
    trend_analyzer: Any = None
    request_id: str = ""

    class Config:
        arbitrary_types_allowed = True


# ─── PipelineContext (обёртка для обратной совместимости) ─────────────
class PipelineContext(RecommendationContext):
    """Legacy-обёртка — унаследована от RecommendationContext.
    Используется в GapRunner / RecommendationRunner для обратной совместимости.
    """
    pass


class SkillExtractionResult(BaseModel):
    """Результат извлечения навыков из вакансий (плоский словарь)."""

    frequencies: dict[str, int]
    hybrid_weights: dict[str, float]
    skill_embeddings: dict[str, list[float]]

    class Config:
        arbitrary_types_allowed = True


class ProfileEvaluationResult(BaseModel):
    """Результат оценки профиля студента."""

    market_coverage_score: float
    skill_coverage: float
    domain_coverage_score: float
    readiness_score: float
    avg_gap: float = 0.0
    skill_metrics: dict[str, Any] = Field(default_factory=dict)
    domain_coverage: dict[str, Any] = Field(default_factory=dict)
    cluster_context: dict[str, Any] | None = None
    top_recommendations: list[tuple[str, float]] = Field(default_factory=list)
    gaps: dict[str, Any] = Field(default_factory=dict)
    level_weights_used: dict[str, float] | None = None
    student_skills: list[str] = Field(default_factory=list)
    market_skill_coverage: float = 0.0
    skill_categories: dict[str, int] = Field(default_factory=dict)
    profession_coverage: float = 0.0
    profession_coverage_detail: dict[str, float] = Field(default_factory=dict)
    domain_skill_count: int = 0
    target_profession: str = ""
    target_domains: list[str] = Field(default_factory=list)
    krm_coverage: dict[str, dict[str, float]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class RecommendationItem(BaseModel):
    skill: str = ""
    action: str = ""
    impact: float = 0.0
    priority: str = "medium"


class DomainCoverage(BaseModel):
    score: float = 0.0
    skills: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    """Ответ движка рекомендаций."""

    summary: dict[str, Any] = Field(default_factory=dict)
    closest_roles: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    domain_coverage: dict[str, Any] = Field(default_factory=dict)
    gaps: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
