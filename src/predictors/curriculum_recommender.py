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

        # Gaps: RPD skills not on market — suggest replacement
        if coverage.gaps > 0:
            for g in coverage.gaps_list:
                replacement = self._suggest_replacement(g, coverage.truly_missing, coverage.cross_references)
                if replacement:
                    recs.append(Recommendation(
                        type="update_content",
                        priority="medium",
                        message=(
                            f"Замените «{g}» на «{replacement}» (частота на рынке: "
                            f"{next((m.frequency for m in coverage.truly_missing if m.skill_name == replacement), '?')})"
                        ),
                    ))
                else:
                    recs.append(Recommendation(
                        type="update_content",
                        priority="medium",
                        message=f"Исключите «{g}» — навык не востребован на рынке",
                    ))

        # Truly missing: market skills not in ANY discipline
        if coverage.truly_missing:
            for m in coverage.truly_missing[:5]:
                recs.append(Recommendation(
                    type="add_new_content",
                    priority="high" if m.frequency > 100 else "medium",
                    message=(
                        f"Добавьте «{m.skill_name}» (частота {m.frequency}) "
                        f"— не покрыт ни в одной дисциплине направления"
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
                    type="add_new_content",
                    priority="low",
                    message=(
                        f"Навык «{cr.skill_name}» (частота {cr.frequency}) уже преподаётся "
                        f"в дисциплине «{cr.discipline}» — обеспечьте междисциплинарную связь"
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
                type="update_content",
                priority="medium",
                message=(
                    f"Компетенции {', '.join(zero_comps)} имеют 0% покрытие рынком"
                    f" — добавьте востребованные навыки"
                ),
            ))

        logger.info("recommendations_generated",
                     discipline=coverage.discipline_name, count=len(recs))
        return Ok(recs)

    @staticmethod
    def _suggest_replacement(gap: str, candidates: list, cross_refs: list | None = None) -> str | None:
        gap_low = gap.lower()
        replacements = {
            "программная документация": "git",
            "встроенные системы": "docker",
            "информационно-коммуникационные технологии": "rest api",
            "математические модели": "ml",
            "анализ данных": "pandas",
        }
        all_candidates = list(candidates)
        if cross_refs:
            all_candidates += cross_refs
        for g, r in replacements.items():
            if g in gap_low or gap_low in g:
                if any(c.skill_name == r for c in all_candidates):
                    return r
        return None

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
