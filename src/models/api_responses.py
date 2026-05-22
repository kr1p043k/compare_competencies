"""Pydantic-модели ответов API для валидации всех эндпоинтов."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, RootModel


class HealthResponse(BaseModel):
    status: str
    version: str
    evaluator: bool
    recommendation_engine: bool


class ReadyResponse(BaseModel):
    status: str
    components: dict[str, bool]


class StatusResponse(BaseModel):
    vacancies_loaded: bool
    skill_weights_count: int
    taxonomy_loaded: bool
    whitelist_size: int
    profiles_available: list[str]
    clusters: dict[str, bool]
    trends_available: bool
    recommendation_engine_ready: bool


class ProfileShort(BaseModel):
    profile_name: str
    target_level: str
    skills_count: int
    skills: list[str] = Field(default_factory=list)
    competencies_count: int = 0
    competencies: list[str] = Field(default_factory=list)


class ProfiledEval(BaseModel):
    market_coverage_score: float | None = None
    skill_coverage: float | None = None
    domain_coverage_score: float | None = None
    readiness_score: float | None = None
    real_coverage: float | None = None
    error: str | None = None


class ProfilesCompareResponse(BaseModel):
    profiles: dict[str, ProfiledEval]


class SkillItem(BaseModel):
    skill: str
    weight: float


class TopSkillsResponse(BaseModel):
    skills: list[SkillItem]


class SkillInfoResponse(BaseModel):
    skill: str
    frequency: int
    weight: float
    category: str
    icon: str


class MarketCompetenciesResponse(BaseModel):
    skills: list[dict[str, Any]]
    total: int


class ClusterSummaryItem(BaseModel):
    id: int
    name: str
    top_skills: list[str]


class LevelClusters(BaseModel):
    clusters: int | None = None
    type: str | None = None
    top_clusters: list[ClusterSummaryItem] | None = None
    error: str | None = None


class ClusterSummaryResponse(RootModel):
    root: dict[str, LevelClusters]


class ClustersByLevelResponse(BaseModel):
    level: str
    clusters: list[ClusterSummaryItem]


class TrendsResponse(BaseModel):
    trends: dict[str, list[dict[str, Any]]]


class CategoryCoverage(BaseModel):
    label: str
    icon: str
    total: int
    covered: int
    percent: float


class TaxonomyCoverageResponse(BaseModel):
    coverage: dict[str, CategoryCoverage]


class ProfessionItem(BaseModel):
    name: str
    domains: list[str]
    competency_codes: list[str]
    hh_queries: list[str]
    aliases: list[str]


class ProfessionsResponse(BaseModel):
    professions: list[ProfessionItem]
    total: int


class KRMCompetency(BaseModel):
    skill_count: int
    skills: list[str]


class ProfessionDetailResponse(BaseModel):
    name: str
    domains: list[str]
    skill_count: int
    skills: list[str]
    competency_codes: list[str]
    krm_competencies: dict[str, KRMCompetency]


class KRMExpertiseItem(BaseModel):
    coverage: float
    total_required: int
    covered_skills: list[str]
    missing_skills: list[str]


class KRMCoverageResponse(BaseModel):
    profession: str
    user_skills: list[str]
    competency_coverage: dict[str, KRMExpertiseItem]
    avg_coverage: float


class ProfessionEvalResponse(BaseModel):
    profile: str
    target_profession: str
    target_domains: list[str]
    profession_coverage: float
    krm_coverage: dict[str, Any]
    readiness_score: float
    skill_coverage: float
    domain_coverage_score: float


class MissingSkillItem(BaseModel):
    skill: str
    frequency: int


class MissingSkillsResponse(BaseModel):
    missing_skills: list[MissingSkillItem]


class DeadSkillsResponse(BaseModel):
    dead_skills: list[str]


class PipelineTaskStatus(BaseModel):
    task_id: str
    status: str
    message: str
    started_at: float | None = None
    completed_at: float | None = None
    output: str | None = None


class PipelineTaskListResponse(BaseModel):
    tasks: list[PipelineTaskStatus]
    total: int


class PipelineStatusResponse(BaseModel):
    clusters: dict[str, bool]
    clusters_all_ready: bool
    ltr_model: bool
    recommendations: dict[str, bool]
    recommendations_all_ready: bool
    skill_weights: bool
    scripts: dict[str, bool]


class PipelineSimpleResponse(BaseModel):
    status: str
    message: str
    task_id: str | None = None


class CacheRefreshResponse(BaseModel):
    status: str
    message: str
    removed: list[str]
    next_step: str


class VacancyItem(BaseModel):
    id: Any = None
    name: str
    experience: str
    salary_from: float | None = None
    salary_to: float | None = None
    salary_currency: str = "RUR"
    employer_name: str
    employer_logo: str | None = None
    area: str
    published_at: str | None = None
    alternate_url: str | None = None
    skills: list[str]
    snippet: dict[str, Any] = Field(default_factory=dict)


class VacanciesResponse(BaseModel):
    items: list[VacancyItem]
    total: int
    limit: int
    offset: int
    has_more: bool


class VacancyDetailResponse(BaseModel):
    id: Any = None
    name: str | None = None
    description: str = ""
    experience: Any = None
    salary: Any = None
    employer: Any = None
    area: Any = None
    published_at: str | None = None
    alternate_url: str | None = None
    skills: list[str] = Field(default_factory=list)
    schedule: Any = None
    employment: Any = None
    key_skills: list[Any] = Field(default_factory=list)
    snippet: Any = None


class VacancyStatsResponse(BaseModel):
    total: int
    by_experience: dict[str, int]
    salary: dict[str, float]


class RegionsResponse(BaseModel):
    regions: list[str]
    total: int
    default: str = "Все регионы"


class VacanciesByRegionResponse(BaseModel):
    region: str
    count: int
    limit: int
    vacancies: list[dict[str, Any]]
