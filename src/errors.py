"""Domain-specific error types for explicit error handling with Result[T, E]."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainError(Exception):
    """Базовая доменная ошибка."""
    message: str
    detail: str = ""

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class VacancyError(DomainError):
    """Ошибка вакансии."""
    vacancy_id: str = ""


@dataclass(frozen=True)
class VacancyNotFoundError(VacancyError):
    """Вакансия не найдена."""
    pass


@dataclass(frozen=True)
class VacancyApiError(VacancyError):
    """Ошибка API вакансий."""
    status_code: int = 0


@dataclass(frozen=True)
class ParseError(DomainError):
    """Ошибка парсинга."""
    source: str = ""


@dataclass(frozen=True)
class SkillParseError(ParseError):
    """Ошибка парсинга навыка."""
    skill_name: str = ""


@dataclass(frozen=True)
class ApiError(DomainError):
    """Ошибка API."""
    status_code: int = 0
    endpoint: str = ""


@dataclass(frozen=True)
class RateLimitError(ApiError):
    """Превышен лимит запросов."""
    retry_after: float = 0.0


@dataclass(frozen=True)
class ModelError(DomainError):
    """Ошибка ML-модели."""
    model_name: str = ""


@dataclass(frozen=True)
class ModelNotFoundError(ModelError):
    """Модель не найдена."""
    path: str = ""


@dataclass(frozen=True)
class ModelTrainingError(ModelError):
    """Ошибка обучения модели."""
    n_samples: int = 0


@dataclass(frozen=True)
class ScorerError(DomainError):
    """Ошибка скорера."""
    pass


@dataclass(frozen=True)
class ConfigError(DomainError):
    """Ошибка конфигурации."""
    key: str = ""


@dataclass(frozen=True)
class PipelineError(DomainError):
    """Ошибка пайплайна."""
    stage: str = ""


@dataclass(frozen=True)
class SkillExtractionError(PipelineError):
    """Ошибка извлечения навыков."""
    vacancies_count: int = 0


@dataclass(frozen=True)
class LevelBuildError(PipelineError):
    """Ошибка построения уровней."""
    vacancies_count: int = 0


@dataclass(frozen=True)
class WeightCleanError(PipelineError):
    """Ошибка чистки весов."""
    skills_count: int = 0


@dataclass(frozen=True)
class GapAnalysisError(PipelineError):
    """Ошибка gap-анализа."""
    profiles_count: int = 0


@dataclass(frozen=True)
class RecommendationError(DomainError):
    """Ошибка рекомендаций."""
    profile: str = ""


@dataclass(frozen=True)
class DataSourceError(DomainError):
    """Ошибка источника данных."""
    source: str = ""


@dataclass(frozen=True)
class CacheError(DomainError):
    """Ошибка кэша."""
    cache_path: str = ""


@dataclass(frozen=True)
class ManifestError(DomainError):
    """Ошибка манифеста артефакта."""
    artifact_path: str = ""


@dataclass(frozen=True)
class NormalizerError(DomainError):
    """Ошибка нормализации."""
    skill_name: str = ""


@dataclass(frozen=True)
class TeacherAnalysisError(DomainError):
    """Ошибка teacher analysis."""
    pass


@dataclass(frozen=True)
class AnalysisDataError(TeacherAnalysisError):
    """Ошибка данных анализа."""
    source: str = ""


@dataclass(frozen=True)
class MatchingError(TeacherAnalysisError):
    """Ошибка matching."""
    skill_name: str = ""


@dataclass(frozen=True)
class CoverageError(TeacherAnalysisError):
    """Ошибка расчёта покрытия."""
    discipline_id: str = ""


@dataclass(frozen=True)
class TrendError(TeacherAnalysisError):
    """Ошибка трендов."""
    reason: str = ""


@dataclass(frozen=True)
class AnalysisRunnerError(TeacherAnalysisError):
    """Ошибка запуска анализа."""
    stage: str = ""
