"""
Анализатор пробелов (gaps) с использованием модели SkillMetrics.
"""
from typing import Dict, List
import numpy as np
from src.models.market_metrics import SkillMetrics

class GapAnalyzer:
    def __init__(self, skill_weights_by_level: Dict[str, Dict[str, float]]):
        """
        skill_weights_by_level: {'junior': {skill: weight}, 'middle': {...}, 'senior': {...}}
        """
        self.skill_weights = skill_weights_by_level

    def compute_metrics(self, user_skills: List[str], user_levels: Dict[str, float]) -> Dict[str, SkillMetrics]:
        metrics: Dict[str, SkillMetrics] = {}
        for level in ['junior', 'middle', 'senior']:
            level_key = level[0]
            for skill, market_weight in self.skill_weights.get(level, {}).items():
                if skill not in metrics:
                    metrics[skill] = SkillMetrics(
                        skill=skill,
                        user_level=user_levels.get(skill, 0.0)
                    )
                gap = max(0.0, market_weight - user_levels.get(skill, 0.0))
                demand = np.log1p(market_weight)
                setattr(metrics[skill], f'gap_{level_key}', gap)
                setattr(metrics[skill], f'demand_{level_key}', demand)
        return metrics
    def set_weights_by_level(self, weights_by_level: Dict[str, Dict[str, float]]):
        self.skill_weights_by_level = weights_by_level