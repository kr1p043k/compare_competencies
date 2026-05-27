"""Domain-specific error types for explicit error handling with Result[T, E]."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainError:
    message: str
    detail: str = ""


@dataclass
class VacancyError(DomainError):
    vacancy_id: str = ""


@dataclass
class VacancyNotFoundError(VacancyError):
    pass


@dataclass
class VacancyApiError(VacancyError):
    status_code: int = 0


@dataclass
class ParseError(DomainError):
    source: str = ""


@dataclass
class SkillParseError(ParseError):
    skill_name: str = ""


@dataclass
class ApiError(DomainError):
    status_code: int = 0
    endpoint: str = ""


@dataclass
class RateLimitError(ApiError):
    retry_after: float = 0.0


@dataclass
class ModelError(DomainError):
    model_name: str = ""


@dataclass
class ModelNotFoundError(ModelError):
    path: str = ""


@dataclass
class ModelTrainingError(ModelError):
    n_samples: int = 0


@dataclass
class ScorerError(DomainError):
    pass


@dataclass
class ConfigError(DomainError):
    key: str = ""


@dataclass
class PipelineError(DomainError):
    stage: str = ""


@dataclass
class SkillExtractionError(PipelineError):
    vacancies_count: int = 0


@dataclass
class LevelBuildError(PipelineError):
    vacancies_count: int = 0


@dataclass
class WeightCleanError(PipelineError):
    skills_count: int = 0


@dataclass
class GapAnalysisError(PipelineError):
    profiles_count: int = 0


@dataclass
class RecommendationError(DomainError):
    profile: str = ""


@dataclass
class DataSourceError(DomainError):
    source: str = ""


@dataclass
class CacheError(DomainError):
    cache_path: str = ""


@dataclass
class ManifestError(DomainError):
    artifact_path: str = ""


@dataclass
class NormalizerError(DomainError):
    skill_name: str = ""
