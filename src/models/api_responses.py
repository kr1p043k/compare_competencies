"""Pydantic-модели ответов API для валидации всех эндпоинтов."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, RootModel


class HealthResponse(BaseModel):
    """Ответ health-check."""
    status: str
    version: str
    evaluator: bool
    recommendation_engine: bool


class ReadyResponse(BaseModel):
    """Готовность сервиса и компонентов."""
    status: str
    components: dict[str, bool]


class StatusResponse(BaseModel):
    """Статус загрузки данных."""
    vacancies_loaded: bool
    skill_weights_count: int
    taxonomy_loaded: bool
    whitelist_size: int
    profiles_available: list[str]
    clusters: dict[str, bool]
    trends_available: bool
    recommendation_engine_ready: bool


class ProfileShort(BaseModel):
    """Краткий профиль студента."""
    profile_name: str
    target_level: str
    skills_count: int
    skills: list[str] = Field(default_factory=list)
    competencies_count: int = 0
    competencies: list[str] = Field(default_factory=list)


class ProfiledEval(BaseModel):
    """Оценки профиля."""
    market_coverage_score: float | None = None
    skill_coverage: float | None = None
    domain_coverage_score: float | None = None
    readiness_score: float | None = None
    real_coverage: float | None = None
    error: str | None = None


class ProfilesCompareResponse(BaseModel):
    """Сравнение профилей."""
    profiles: dict[str, ProfiledEval]


class SkillItem(BaseModel):
    """Навык с весом."""
    skill: str
    weight: float


class TopSkillsResponse(BaseModel):
    """Топ навыков."""
    skills: list[SkillItem]


class SkillInfoResponse(BaseModel):
    """Информация о навыке."""
    skill: str
    frequency: int
    weight: float
    category: str
    icon: str


class MarketCompetenciesResponse(BaseModel):
    """Компетенции рынка."""
    skills: list[dict[str, Any]]
    total: int


class ClusterSummaryItem(BaseModel):
    """Сводка кластера."""
    id: int
    name: str
    top_skills: list[str]


class LevelClusters(BaseModel):
    """Кластеры уровня."""
    clusters: int | None = None
    type: str | None = None
    top_clusters: list[ClusterSummaryItem] | None = None
    error: str | None = None


class ClusterSummaryResponse(RootModel):
    """Сводка кластеризации."""
    root: dict[str, LevelClusters]


class ClustersByLevelResponse(BaseModel):
    """Кластеры по уровню."""
    level: str
    clusters: list[ClusterSummaryItem]


class TrendsResponse(BaseModel):
    """Тренды навыков."""
    trends: dict[str, list[dict[str, Any]]]


class CategoryCoverage(BaseModel):
    """Покрытие категории."""
    label: str
    icon: str
    total: int
    covered: int
    percent: float


class TaxonomyCoverageResponse(BaseModel):
    """Покрытие таксономии."""
    coverage: dict[str, CategoryCoverage]


class ProfessionItem(BaseModel):
    """Профессия."""
    name: str
    domains: list[str]
    competency_codes: list[str]
    hh_queries: list[str]
    aliases: list[str]


class ProfessionsResponse(BaseModel):
    """Список профессий."""
    professions: list[ProfessionItem]
    total: int


class KRMCompetency(BaseModel):
    """Компетенция KRM с навыками."""
    skill_count: int
    skills: list[str]


class ProfessionDetailResponse(BaseModel):
    """Детали профессии."""
    name: str
    domains: list[str]
    skill_count: int
    skills: list[str]
    competency_codes: list[str]
    krm_competencies: dict[str, KRMCompetency]


class KRMExpertiseItem(BaseModel):
    """Экспертиза KRM."""
    coverage: float
    total_required: int
    covered_skills: list[str]
    missing_skills: list[str]


class KRMCoverageResponse(BaseModel):
    """KRM-покрытие."""
    profession: str
    user_skills: list[str]
    competency_coverage: dict[str, KRMExpertiseItem]
    avg_coverage: float


class ProfessionEvalResponse(BaseModel):
    """Оценка профиля под профессию."""
    profile: str
    target_profession: str
    target_domains: list[str]
    profession_coverage: float
    krm_coverage: dict[str, Any]
    readiness_score: float
    skill_coverage: float
    domain_coverage_score: float


class MissingSkillItem(BaseModel):
    """Отсутствующий навык."""
    skill: str
    frequency: int


class MissingSkillsResponse(BaseModel):
    """Отсутствующие навыки."""
    missing_skills: list[MissingSkillItem]


class DeadSkillsResponse(BaseModel):
    """Мёртвые навыки."""
    dead_skills: list[str]


class PipelineTaskStatus(BaseModel):
    """Статус задачи пайплайна."""
    task_id: str
    status: str
    message: str
    started_at: float | None = None
    completed_at: float | None = None
    output: str | None = None
    step: int = 0
    sub_progress: int | None = None
    logs: list[str] = []


class GapProgressResponse(BaseModel):
    """Прогресс gap-анализа."""
    pct: float = 0.0
    message: str = ""
    stage: str = ""
    exists: bool = False


class PipelineTaskListResponse(BaseModel):
    """Список задач."""
    tasks: list[PipelineTaskStatus]
    total: int


class PipelineStatusResponse(BaseModel):
    """Статус пайплайна."""
    clusters: dict[str, bool]
    clusters_all_ready: bool
    ltr_model: bool
    recommendations: dict[str, bool]
    recommendations_all_ready: bool
    skill_weights: bool
    scripts: dict[str, bool]


class PipelineSimpleResponse(BaseModel):
    """Простой ответ пайплайна."""
    status: str
    message: str
    task_id: str | None = None


class CacheRefreshResponse(BaseModel):
    """Результат обновления кэша."""
    status: str
    message: str
    removed: list[str]
    next_step: str


class VacancyItem(BaseModel):
    """Краткая вакансия."""
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
    """Список вакансий с пагинацией."""
    items: list[VacancyItem]
    total: int
    limit: int
    offset: int
    has_more: bool


class VacancyDetailResponse(BaseModel):
    """Детали вакансии."""
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
    """Статистика вакансий."""
    total: int
    by_experience: dict[str, int]
    salary: dict[str, float]


class RegionsResponse(BaseModel):
    """Регионы поиска."""
    regions: list[str]
    total: int
    default: str = "Все регионы"


class VacanciesByRegionResponse(BaseModel):
    """Вакансии региона."""
    region: str
    count: int
    limit: int
    vacancies: list[dict[str, Any]]
