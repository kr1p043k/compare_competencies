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
