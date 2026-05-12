"""Метрики рынка для gap-анализа."""

from dataclasses import dataclass

import structlog

from .enums import SkillCategory

logger = structlog.get_logger(__name__)


@dataclass
class SkillMetrics:
    skill: str
    gap_j: float = 0.0
    gap_m: float = 0.0
    gap_s: float = 0.0
    demand_j: float = 0.0
    demand_m: float = 0.0
    demand_s: float = 0.0
    cluster_relevance: float = 0.0
    user_level: float = 0.0
    importance: float = 0.0  # новое: нормализованная важность (0-1)
    category: SkillCategory = SkillCategory.MISSING  # новое: missing / weak / strong

    def score(self, level_weights: dict[str, float], domain_bonus: float = 0.0) -> float:
        alpha, beta, gamma = 0.5, 0.3, 0.2
        domain_factor = 1.0 + 0.15 * domain_bonus

        def norm(x: float) -> float:
            return max(0.0, min(1.0, x))

        score_j = alpha * norm(self.gap_j) + beta * norm(self.demand_j) + gamma * self.cluster_relevance
        score_m = alpha * norm(self.gap_m) + beta * norm(self.demand_m) + gamma * self.cluster_relevance
        score_s = alpha * norm(self.gap_s) + beta * norm(self.demand_s) + gamma * self.cluster_relevance

        base_score = (
            level_weights.get("junior", 0) * score_j
            + level_weights.get("middle", 0) * score_m
            + level_weights.get("senior", 0) * score_s
        )

        final_score = base_score * domain_factor

        logger.debug(
            "skill_scored",
            skill=self.skill,
            base_score=round(base_score, 4),
            domain_bonus=round(domain_bonus, 4),
            final_score=round(final_score, 4),
            scores=dict(j=round(score_j, 3), m=round(score_m, 3), s=round(score_s, 3)),
        )

        return final_score

    def __post_init__(self):
        if not self.category:
            max_gap = max(self.gap_j, self.gap_m, self.gap_s)
            if max_gap < 0.2:
                self.category = "strong"
            elif max_gap < 0.6:
                self.category = "weak"
            else:
                self.category = "missing"

            logger.debug(
                "skill_category_assigned",
                skill=self.skill,
                max_gap=round(max_gap, 3),
                category=self.category,
            )


@dataclass
class DomainMetrics:
    domain: str
    required_skills: list[str]
    user_has: int = 0
    total_required: int = 0
    coverage: float = 0.0
    importance: float = 1.0

    def compute_coverage(self, user_skills: set):
        self.user_has = len(set(s.lower() for s in self.required_skills) & user_skills)
        self.total_required = len(self.required_skills)
        self.coverage = self.user_has / self.total_required if self.total_required > 0 else 0.0

        logger.debug(
            "domain_coverage_computed",
            domain=self.domain,
            user_has=self.user_has,
            total=self.total_required,
            coverage=round(self.coverage, 4),
        )

        return self.coverage
