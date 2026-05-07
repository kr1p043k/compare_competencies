"""
Анализатор пробелов (gaps) с использованием модели SkillMetrics.
"""

import structlog

from src.models.market_metrics import SkillMetrics

logger = structlog.get_logger(__name__)


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
                # market_weight уже нормализован в [0, 1], log1p не даёт смыслового выигрыша
                demand = market_weight
                setattr(metrics[skill], f"gap_{level_key}", gap)
                setattr(metrics[skill], f"demand_{level_key}", demand)

        # Сводная статистика
        total_gaps = sum(max(m.gap_j, m.gap_m, m.gap_s) for m in metrics.values())
        avg_gap = total_gaps / len(metrics) if metrics else 0

        logger.info(
            "metrics_computed",
            total_skills=len(metrics),
            avg_gap=round(avg_gap, 4),
            levels_available=list(self.skill_weights.keys()),
        )

        # Детализация по топ-5 навыкам с наибольшим gap
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

        return metrics

    def set_weights_by_level(self, weights_by_level: dict[str, dict[str, float]]):
        self.skill_weights_by_level = weights_by_level
