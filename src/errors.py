"""Domain-specific error types for explicit error handling with Result[T, E]."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainError(Exception):
    message: str
    detail: str = ""

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class VacancyError(DomainError):
    vacancy_id: str = ""


@dataclass(frozen=True)
class VacancyNotFoundError(VacancyError):
    pass


@dataclass(frozen=True)
class VacancyApiError(VacancyError):
    status_code: int = 0


@dataclass(frozen=True)
class ParseError(DomainError):
    source: str = ""


@dataclass(frozen=True)
class SkillParseError(ParseError):
    skill_name: str = ""


@dataclass(frozen=True)
class ApiError(DomainError):
    status_code: int = 0
    endpoint: str = ""


@dataclass(frozen=True)
class RateLimitError(ApiError):
    retry_after: float = 0.0


@dataclass(frozen=True)
class ModelError(DomainError):
    model_name: str = ""


@dataclass(frozen=True)
class ModelNotFoundError(ModelError):
    path: str = ""


@dataclass(frozen=True)
class ModelTrainingError(ModelError):
    n_samples: int = 0


@dataclass(frozen=True)
class ScorerError(DomainError):
    pass


@dataclass(frozen=True)
class ConfigError(DomainError):
    key: str = ""


@dataclass(frozen=True)
class PipelineError(DomainError):
    stage: str = ""


@dataclass(frozen=True)
class SkillExtractionError(PipelineError):
    vacancies_count: int = 0


@dataclass(frozen=True)
class LevelBuildError(PipelineError):
    vacancies_count: int = 0


@dataclass(frozen=True)
class WeightCleanError(PipelineError):
    skills_count: int = 0


@dataclass(frozen=True)
class GapAnalysisError(PipelineError):
    profiles_count: int = 0


@dataclass(frozen=True)
class RecommendationError(DomainError):
    profile: str = ""


@dataclass(frozen=True)
class DataSourceError(DomainError):
    source: str = ""


@dataclass(frozen=True)
class CacheError(DomainError):
    cache_path: str = ""


@dataclass(frozen=True)
class ManifestError(DomainError):
    artifact_path: str = ""


@dataclass(frozen=True)
class NormalizerError(DomainError):
    skill_name: str = ""


@dataclass(frozen=True)
class TeacherAnalysisError(DomainError):
    pass


@dataclass(frozen=True)
class AnalysisDataError(TeacherAnalysisError):
    source: str = ""


@dataclass(frozen=True)
class MatchingError(TeacherAnalysisError):
    skill_name: str = ""


@dataclass(frozen=True)
class CoverageError(TeacherAnalysisError):
    discipline_id: str = ""


@dataclass(frozen=True)
class TrendError(TeacherAnalysisError):
    reason: str = ""


@dataclass(frozen=True)
class AnalysisRunnerError(TeacherAnalysisError):
    stage: str = ""
