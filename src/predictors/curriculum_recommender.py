"""Curriculum recommender: generates per-discipline curriculum recommendations."""
from __future__ import annotations

import structlog

from src.result import Ok, Err, Result
from src.errors import RecommendationError
from src.models.teacher_analysis import Recommendation, DisciplineCoverage

logger = structlog.get_logger(__name__)


class CurriculumRecommender:
    def __init__(self):
        pass

    def generate(self, coverage: DisciplineCoverage) -> Result[list[Recommendation], RecommendationError]:
        if not coverage:
            logger.error("coverage_none")
            return Err(RecommendationError(message="Coverage data is required"))

        recs = []

        if coverage.gaps > 0:
            priority = "high" if coverage.gaps > 5 else "medium"
            sample = coverage.gaps_list[:5]
            recs.append(Recommendation(
                type="update_content",
                priority=priority,
                message=(
                    f"Обновить содержание: {coverage.gaps} навыков из РПД не найдены на рынке."
                    f" Рекомендуется пересмотреть актуальность: {', '.join(sample)}"
                ),
            ))

        if coverage.emerging:
            top_emerging = [e.skill_name for e in coverage.emerging[:5]]
            priority = "medium" if coverage.emerging[0].frequency > 10 else "low"
            recs.append(Recommendation(
                type="add_new_content",
                priority=priority,
                message=(
                    f"Добавить в программу востребованные навыки рынка: {', '.join(top_emerging)}"
                ),
            ))

        if coverage.coverage_ratio < 0.3:
            recs.append(Recommendation(
                type="major_revision",
                priority="high",
                message=(
                    f"Низкое покрытие рынка ({coverage.coverage_ratio * 100:.1f}%)."
                    f" Требуется существенный пересмотр дисциплины."
                ),
            ))

        logger.info("recommendations_generated",
                     discipline=coverage.discipline_name, count=len(recs))
        return Ok(recs)

    def generate_summary_recommendations(
        self, all_coverages: list[DisciplineCoverage],
        avg_coverage: float, total_gaps: int, top_emerging: list[dict],
    ) -> Result[list[Recommendation], RecommendationError]:
        if not all_coverages:
            logger.warning("no_coverages_for_summary_recs")
            return Err(RecommendationError(message="No coverage data provided"))

        recs = []
        if avg_coverage < 0.3:
            low_count = sum(1 for c in all_coverages if c.coverage_ratio < 0.2)
            recs.append(Recommendation(
                type="major_revision",
                priority="high",
                message=(
                    f"Среднее покрытие рынка по направлению {avg_coverage * 100:.1f}%. "
                    f"Рекомендуется обновить {low_count} "
                    f"дисциплин с низким покрытием."
                ),
            ))
        if top_emerging:
            skills = [e["skill"] for e in top_emerging[:10]]
            recs.append(Recommendation(
                type="add_new_content",
                priority="high",
                message=f"Ключевые навыки рынка для внедрения: {', '.join(skills)}",
            ))

        logger.info("summary_recommendations_generated", count=len(recs))
        return Ok(recs)
