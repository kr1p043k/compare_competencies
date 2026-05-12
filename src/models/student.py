"""
Модель данных студента с поддержкой уровней опыта, оценками и эмбеддингами.
"""

from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from .enums import ExperienceLevel

logger = structlog.get_logger(__name__)


class StudentProfile(BaseModel):
    """Профиль студента"""

    profile_name: str
    competencies: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    target_level: ExperienceLevel = ExperienceLevel.MIDDLE
    created_at: datetime = Field(default_factory=datetime.now)

    # Для работы кластерного контекста
    embedding: Any | None = Field(default=None, exclude=True)  # np.ndarray или list[float]
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def __repr__(self):
        emb_info = f", emb={len(self.embedding) if self.embedding is not None else None}"
        return f"StudentProfile({self.profile_name}, {self.target_level}, skills={len(self.skills)}{emb_info})"

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(
            "student_profile_created",
            profile=self.profile_name,
            target_level=self.target_level,
            skills_count=len(self.skills),
            competencies_count=len(self.competencies),
            has_embedding=self.embedding is not None,
        )


class ProfileEvaluation(BaseModel):
    """Результат оценки одного профиля (новая версия)"""

    profile_name: str
    student: StudentProfile
    level: ExperienceLevel

    market_coverage_score: float
    skill_coverage: float
    domain_coverage_score: float
    readiness_score: float
    avg_gap: float = 0.0

    recommendation: str = ""
    gaps: dict[str, Any] = Field(default_factory=dict)
    cluster_context: dict | None = None
    top_recommendations: list[dict] = Field(default_factory=list)

    evaluated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"ProfileEvaluation({self.profile_name}@{self.level}, readiness={self.readiness_score:.1f}%)"

    def __init__(self, **data):
        super().__init__(**data)
        logger.info(
            "profile_evaluation_created",
            profile=self.profile_name,
            level=self.level,
            readiness=round(self.readiness_score, 2),
            market_coverage=round(self.market_coverage_score, 2),
            skill_coverage=round(self.skill_coverage, 2),
            domain_coverage=round(self.domain_coverage_score, 2),
            avg_gap=round(self.avg_gap, 3),
            recommendations_count=len(self.top_recommendations),
            gaps_count=len(self.gaps),
        )


class ProfileComparison(BaseModel):
    """Сравнение нескольких профилей"""

    evaluations: list[ProfileEvaluation] = Field(default_factory=list)
    best_evaluation: ProfileEvaluation | None = None
    average_readiness: float = 0.0
    average_market_coverage: float = 0.0
    average_skill_coverage: float = 0.0
    average_domain_coverage: float = 0.0
    compared_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True

    def compute_aggregates(self):
        if not self.evaluations:
            logger.debug("compute_aggregates_skipped", reason="no_evaluations")
            return

        n = len(self.evaluations)
        logger.debug("computing_aggregates", profiles_count=n)

        self.average_readiness = sum(e.readiness_score for e in self.evaluations) / n
        self.average_market_coverage = sum(e.market_coverage_score for e in self.evaluations) / n
        self.average_skill_coverage = sum(e.skill_coverage for e in self.evaluations) / n
        self.average_domain_coverage = sum(e.domain_coverage_score for e in self.evaluations) / n

        if self.evaluations:
            self.best_evaluation = max(self.evaluations, key=lambda e: e.readiness_score)
            logger.info(
                "aggregates_computed",
                profiles=n,
                avg_readiness=round(self.average_readiness, 2),
                avg_market_coverage=round(self.average_market_coverage, 2),
                avg_skill_coverage=round(self.average_skill_coverage, 2),
                avg_domain_coverage=round(self.average_domain_coverage, 2),
                best_profile=self.best_evaluation.profile_name,
                best_readiness=round(self.best_evaluation.readiness_score, 2),
            )

    def to_dict_for_json(self) -> dict:
        self.compute_aggregates()
        result = {
            "timestamp": str(self.compared_at),
            "total_profiles": len(self.evaluations),
            "average_readiness": round(self.average_readiness, 2),
            "average_market_coverage": round(self.average_market_coverage, 2),
            "average_skill_coverage": round(self.average_skill_coverage, 2),
            "average_domain_coverage": round(self.average_domain_coverage, 2),
            "best_profile": {
                "profile_name": (self.best_evaluation.profile_name if self.best_evaluation else None),
                "readiness_score": (round(self.best_evaluation.readiness_score, 2) if self.best_evaluation else None),
                "market_coverage": (
                    round(self.best_evaluation.market_coverage_score, 2) if self.best_evaluation else None
                ),
            },
            "profiles": [
                {
                    "profile_name": e.profile_name,
                    "target_level": str(e.level),
                    "readiness_score": round(e.readiness_score, 2),
                    "market_coverage_score": round(e.market_coverage_score, 2),
                    "skill_coverage": round(e.skill_coverage, 2),
                    "domain_coverage_score": round(e.domain_coverage_score, 2),
                    "avg_gap": round(e.avg_gap, 3),
                    "num_skills": len(e.student.skills),
                }
                for e in self.evaluations
            ],
        }
        logger.debug("profile_comparison_serialized_to_json", profiles=len(self.evaluations))
        return result

    def __repr__(self):
        return (
            f"ProfileComparison(profiles={len(self.evaluations)}, "
            f"avg_readiness={self.average_readiness:.1f}%, "
            f"best={self.best_evaluation.profile_name if self.best_evaluation else 'N/A'})"
        )


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================


def merge_skills_hierarchically(top_skills: list[str], middle_skills: list[str], base_skills: list[str]) -> list[str]:
    """
    Объединяет навыки трёх уровней (top_dc, dc, base) в один список,
    сохраняя порядок приоритета и исключая дубликаты.

    Используется для профиля 'top_dc', чтобы гарантировать наличие
    всех навыков из более низких уровней.
    """
    seen = set()
    merged = []
    duplicates = 0

    # Сначала топ-навыки
    for skill in top_skills:
        if skill not in seen:
            merged.append(skill)
            seen.add(skill)
        else:
            duplicates += 1

    # Потом middle
    for skill in middle_skills:
        if skill not in seen:
            merged.append(skill)
            seen.add(skill)
        else:
            duplicates += 1

    # Потом base
    for skill in base_skills:
        if skill not in seen:
            merged.append(skill)
            seen.add(skill)
        else:
            duplicates += 1

    logger.debug(
        "skills_merged_hierarchically",
        top_count=len(top_skills),
        middle_count=len(middle_skills),
        base_count=len(base_skills),
        merged_count=len(merged),
        duplicates_removed=duplicates,
    )

    return merged
