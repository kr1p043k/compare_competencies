from pydantic import BaseModel, Field
from typing import Any


class SkillImpact(BaseModel):
    skill: str
    score: float = Field(ge=0.0, le=100.0)
    explanation: str = ""


class Recommendation(BaseModel):
    rank: int = 0
    skill: str
    importance_score: float = Field(ge=0.0)
    # R3: разложение итога — база (blend ev/LTR + reranker) и дельта бонусов.
    importance_base: float = 0.0
    importance_bonus: float = 0.0
    priority: str = "medium"
    category: str = "missing"
    why_important: str = ""
    how_to_learn: str = ""
    expected_timeframe: str = ""
    expected_outcome: str = ""
    is_soft_skill: bool = False
    market_frequency_percent: float = 0.0


class RecommendationSummary(BaseModel):
    match_score: float = 0.0
    confidence: float = 0.0
    market_coverage_score: float = 0.0
    skill_coverage: float = 0.0
    domain_coverage_score: float = 0.0
    readiness_score: float = 0.0
    profession_coverage: float = 0.0
    avg_gap: float = 0.0
    coverage: float = 0.0
    coverage_details: dict[str, int] = {}
    market_skill_coverage: float = 0.0
    # R2: явная пара покрытий + область строгого ("market" | "profession").
    coverage_strict: float = 0.0
    coverage_weighted: float = 0.0
    coverage_strict_scope: str = "market"


class ClosestRole(BaseModel):
    role: str
    semantic_similarity: float = 0.0
    similarity_explanation: str = ""
    skills_covered: str = ""
    coverage_percent: float = 0.0
    coverage_explanation: str = ""
    cluster_skills: list[str] = []
    cluster_core_skills: list[str] = []
    # L2: прозрачность ранжирования ролей.
    rank_score: float = 0.0
    target_overlap: float = 0.0
    target_profession: str = ""
    dominant_category: str = ""
    cluster_level: str = ""


class RecommendationResult(BaseModel):
    summary: RecommendationSummary
    profession_coverage_detail: dict[str, Any] = {}
    closest_roles: list[ClosestRole] = []
    recommendations: list[Recommendation] = []
    domain_coverage: dict[str, Any] = {}
    gaps: dict[str, Any] = {}
    trend_bonuses_count: int = 0
    dominant_domain_name: str | None = None
    target_profession: str = ""
