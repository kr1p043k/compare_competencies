"""
Анализатор пробелов (gaps) с использованием модели SkillMetrics.
"""

import numpy as np

from src.models.market_metrics import SkillMetrics


class GapAnalyzer:
    def __init__(self, skill_weights_by_level: dict[str, dict[str, float]]):
        self.skill_weights = skill_weights_by_level

    def compute_metrics(self, user_skills: list[str], user_levels: dict[str, float]) -> dict[str, SkillMetrics]:
        metrics: dict[str, SkillMetrics] = {}
        all_weights = {}
        for level_data in self.skill_weights.values():
            all_weights.update(level_data)
        max_weight = max(all_weights.values()) if all_weights else 1.0

        for level in ["junior", "middle", "senior"]:
            level_key = level[0]
            for skill, market_weight in self.skill_weights.get(level, {}).items():
                if skill not in metrics:
                    user_lvl = user_levels.get(skill, 0.0)
                    metrics[skill] = SkillMetrics(
                        skill=skill, user_level=user_lvl, importance=round(market_weight / max_weight, 4)
                    )
                gap = max(0.0, market_weight - user_levels.get(skill, 0.0))
                demand = np.log1p(market_weight)
                setattr(metrics[skill], f"gap_{level_key}", gap)
                setattr(metrics[skill], f"demand_{level_key}", demand)

        return metrics

    def set_weights_by_level(self, weights_by_level: dict[str, dict[str, float]]):
        self.skill_weights_by_level = weights_by_level
