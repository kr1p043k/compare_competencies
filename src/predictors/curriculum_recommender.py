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

        # Gaps: RPD skills not found on market — flag for human review
        if coverage.gaps > 0:
            sample = coverage.gaps_list[:5]
            recs.append(Recommendation(
                type="review_content",
                priority="medium",
                message=(
                    f"Навыки из РПД не обнаружены в рыночных данных: "
                    f"{', '.join(sample)}. Рекомендуется пересмотреть их актуальность."
                ),
            ))

        # Truly missing: market skills not in ANY discipline — suggest consideration
        if coverage.truly_missing:
            top = coverage.truly_missing[:5]
            skills_str = ", ".join(f"«{m.skill_name}» ({m.frequency})" for m in top)
            recs.append(Recommendation(
                type="add_new_content",
                priority="medium",
                message=(
                    f"Рассмотрите возможность включения востребованных навыков, "
                    f"отсутствующих в программе: {skills_str}."
                ),
            ))

        # Cross-references: skills taught in other disciplines
        if coverage.cross_references:
            seen: set[str] = set()
            for cr in coverage.cross_references:
                if cr.skill_name in seen:
                    continue
                seen.add(cr.skill_name)
                recs.append(Recommendation(
                    type="cross_reference",
                    priority="low",
                    message=(
                        f"Навык «{cr.skill_name}» ({cr.frequency}) преподаётся "
                        f"в дисциплине «{cr.discipline}» — в рамках текущей дисциплины "
                        f"достаточно упомянуть или сослаться."
                    ),
                ))

        # Low coverage warning
        if coverage.coverage_ratio < 0.3:
            low_comps = [c.code for c in coverage.competencies if c.coverage < 0.3 and c.total_skills > 0]
            comp_msg = f" Низкое покрытие компетенций: {', '.join(low_comps)}." if low_comps else ""
            recs.append(Recommendation(
                type="major_revision",
                priority="high",
                message=(
                    f"Низкое покрытие рынка ({coverage.coverage_ratio * 100:.1f}%)."
                    f"{comp_msg} Требуется существенный пересмотр дисциплины."
                ),
            ))

        # Zero-coverage competencies
        zero_comps = [c.code for c in coverage.competencies if c.coverage == 0 and c.total_skills > 0]
        if zero_comps:
            recs.append(Recommendation(
                type="review_content",
                priority="medium",
                message=(
                    f"Компетенции {', '.join(zero_comps)} имеют 0% покрытие рынком"
                    f" — рекомендуется наполнить их востребованными навыками."
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
