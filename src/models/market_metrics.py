from dataclasses import dataclass
from typing import Dict, List

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

    def score(self, level_weights: Dict[str, float], domain_bonus: float = 0.0) -> float:
        """Единая формула RecommendationScore(skill) с возможным бонусом от домена"""
        alpha, beta, gamma = 0.5, 0.3, 0.2
        domain_factor = 1.0 + 0.15 * domain_bonus   # до +15% при полном покрытии домена

        def norm(x: float) -> float:
            return max(0.0, min(1.0, x))

        score_j = (alpha * norm(self.gap_j) +
                   beta * norm(self.demand_j) +
                   gamma * self.cluster_relevance)
        score_m = (alpha * norm(self.gap_m) +
                   beta * norm(self.demand_m) +
                   gamma * self.cluster_relevance)
        score_s = (alpha * norm(self.gap_s) +
                   beta * norm(self.demand_s) +
                   gamma * self.cluster_relevance)

        base_score = (level_weights.get('junior', 0) * score_j +
                      level_weights.get('middle', 0) * score_m +
                      level_weights.get('senior', 0) * score_s)

        return base_score * domain_factor


@dataclass
class DomainMetrics:
    domain: str
    required_skills: List[str]
    user_has: int = 0
    total_required: int = 0
    coverage: float = 0.0
    importance: float = 1.0

    def compute_coverage(self, user_skills: set):
        self.user_has = len(set(s.lower() for s in self.required_skills) & user_skills)
        self.total_required = len(self.required_skills)
        self.coverage = self.user_has / self.total_required if self.total_required > 0 else 0.0
        return self.coverage