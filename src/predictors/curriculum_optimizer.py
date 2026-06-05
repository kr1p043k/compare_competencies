"""Curriculum optimizer: suggests structural program changes."""
from __future__ import annotations

import structlog

from src.result import Ok, Err, Result
from src.errors import RecommendationError
from src.models.teacher_analysis import Recommendation, DirectionSummary

logger = structlog.get_logger(__name__)


class CurriculumOptimizer:
    def __init__(self):
        pass

    def optimize(self, summary: DirectionSummary) -> Result[list[Recommendation], RecommendationError]:
        if not summary:
            logger.error("summary_none")
            return Err(RecommendationError(message="DirectionSummary is required"))

        if not summary.disciplines:
            logger.warning("summary_no_disciplines")
            return Err(RecommendationError(message="No disciplines in summary"))

        recs = []

        low_coverage = [d for d in summary.disciplines if d.get("coverage_level") == "low"]
        if low_coverage:
            names = [d["name"][:40] for d in low_coverage[:5]]
            recs.append(Recommendation(
                type="major_revision",
                priority="high",
                message=(
                    f"{len(low_coverage)} дисциплин имеют низкое покрытие рынка. "
                    f"Рекомендовано к пересмотру: {', '.join(names)}"
                ),
            ))

        if summary.top_emerging:
            recs.append(Recommendation(
                type="add_new_content",
                priority="high",
                message=(
                    f"Добавить новые дисциплины или модули по: "
                    f"{', '.join(e['skill'] for e in summary.top_emerging[:8])}"
                ),
            ))

        redundant = [d for d in summary.disciplines if d.get("gaps", 0) > 100]
        if redundant:
            recs.append(Recommendation(
                type="update_content",
                priority="medium",
                message=(
                    f"{len(redundant)} дисциплин содержат >100 навыков, "
                    f"не востребованных на рынке. Рекомендуется оптимизация."
                ),
            ))

        logger.info("optimization_recs_generated", count=len(recs))
        return Ok(recs)
