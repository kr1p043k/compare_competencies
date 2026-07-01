"""
Анализатор пробелов (gaps) с использованием модели SkillMetrics.
"""

import structlog

from src import config, Result, Ok, Err
from src.errors import DomainError
from src.models.enums import ExperienceLevel
from src.models.market_metrics import SkillMetrics

logger = structlog.get_logger(__name__)


class GapAnalyzer:
    def __init__(self, skill_weights_by_level: dict[str, dict[str, float]] | dict[str, float]):
        if skill_weights_by_level and not isinstance(next(iter(skill_weights_by_level.values())), dict):
            skill_weights_by_level = {"all": skill_weights_by_level}
        self.skill_weights_by_level = skill_weights_by_level

    _LEVEL_KEY: dict[ExperienceLevel, str] = {
        ExperienceLevel.JUNIOR: "j",
        ExperienceLevel.MIDDLE: "m",
        ExperienceLevel.SENIOR: "s",
    }

    def compute_metrics(self, user_skills: list[str], user_levels: dict[str, float]) -> Result[dict[str, SkillMetrics], DomainError]:
        metrics: dict[str, SkillMetrics] = {}
        all_weights = {}
        for level_data in self.skill_weights_by_level.values():
            all_weights.update(level_data)
        max_weight = max(all_weights.values()) if all_weights else 1.0

        for level in ExperienceLevel:
            level_key = self._LEVEL_KEY[level]
            for skill, market_weight in self.skill_weights_by_level.get(level, {}).items():
                if skill not in metrics:
                    user_lvl = user_levels.get(skill, 0.0)
                    metrics[skill] = SkillMetrics(
                        skill=skill, user_level=user_lvl, importance=round(market_weight / max_weight, 4)
                    )
                gap = max(0.0, market_weight - user_levels.get(skill, 0.0))
                demand = market_weight
                setattr(metrics[skill], f"gap_{level_key}", gap)
                setattr(metrics[skill], f"demand_{level_key}", demand)

        # Compute category AFTER gap values are set
        from src.models.market_metrics import SkillCategory
        for m in metrics.values():
            max_gap = max(m.gap_j, m.gap_m, m.gap_s)
            if max_gap < config.SKILL_STRONG_GAP_THRESHOLD:
                m.category = SkillCategory.STRONG
            elif max_gap < config.SKILL_WEAK_GAP_THRESHOLD:
                m.category = SkillCategory.WEAK
            else:
                m.category = SkillCategory.MISSING

        total_gaps = sum(max(m.gap_j, m.gap_m, m.gap_s) for m in metrics.values())
        avg_gap = total_gaps / len(metrics) if metrics else 0

        logger.info(
            "metrics_computed",
            total_skills=len(metrics),
            avg_gap=round(avg_gap, 4),
            levels_available=list(self.skill_weights_by_level.keys()),
        )

        top_gaps = sorted(metrics.items(), key=lambda x: max(x[1].gap_j, x[1].gap_m, x[1].gap_s), reverse=True)[:5]
        logger.debug(
            "top_5_gaps",
            top_gaps=[
                {
                    "skill": skill,
                    "max_gap": round(max(m.gap_j, m.gap_m, m.gap_s), 4),
                    "importance": m.importance,
                }
                for skill, m in top_gaps
            ],
        )

        return Ok(metrics)

    def set_weights_by_level(self, weights_by_level: dict[str, dict[str, float]]):
        self.skill_weights_by_level = weights_by_level

    def get_weights_by_level(self) -> Result[dict[str, dict[str, float]], DomainError]:
        return Ok(self.skill_weights_by_level)
