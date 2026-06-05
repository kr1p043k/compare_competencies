"""Curriculum recommender: generates per-discipline curriculum recommendations."""
from __future__ import annotations

import json
from pathlib import Path

import structlog

from src.result import Ok, Err, Result
from src.errors import RecommendationError
from src.models.teacher_analysis import Recommendation, DisciplineCoverage

logger = structlog.get_logger(__name__)

SKILL_TYPES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "skill_types.json"


def _load_skill_types() -> dict[str, list[str]]:
    if not SKILL_TYPES_PATH.exists():
        logger.warning("skill_types_file_not_found", path=str(SKILL_TYPES_PATH))
        return {"academic": [], "professional": []}
    with open(SKILL_TYPES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _classify_skill(skill: str, types: dict[str, list[str]]) -> str:
    sl = skill.lower().strip()
    for cat in ("academic", "professional"):
        for ref in types.get(cat, []):
            if ref in sl or sl in ref:
                return cat
    return "generic"


class CurriculumRecommender:
    def __init__(self):
        self.skill_types = _load_skill_types()

    def generate(self, coverage: DisciplineCoverage) -> Result[list[Recommendation], RecommendationError]:
        if not coverage:
            logger.error("coverage_none")
            return Err(RecommendationError(message="Coverage data is required"))

        recs = []

        # Gaps: RPD skills not found on market
        if coverage.gaps > 0:
            acad_gaps = []
            generic_gaps = []
            for s in coverage.gaps_list:
                cls = _classify_skill(s, self.skill_types)
                if cls == "academic":
                    acad_gaps.append(s)
                else:
                    generic_gaps.append(s)

            if acad_gaps:
                recs.append(Recommendation(
                    type="foundational",
                    priority="low",
                    message=(
                        f"Фундаментальные навыки, не обнаруженные на рынке: "
                        f"{', '.join(acad_gaps[:5])}. "
                        f"Носят общепрофессиональный характер — не требуют замены."
                    ),
                ))
            if generic_gaps:
                recs.append(Recommendation(
                    type="review_content",
                    priority="medium",
                    message=(
                        f"Навыки из РПД не обнаружены в рыночных данных: "
                        f"{', '.join(generic_gaps[:5])}. "
                        f"Рекомендуется пересмотреть их актуальность."
                    ),
                ))

        # Truly missing: market skills not in ANY discipline — suggest consideration
        if coverage.truly_missing:
            relevant = self._filter_relevant(coverage.truly_missing, coverage.discipline_name)
            if relevant:
                skills_str = ", ".join(f"«{m.skill_name}» ({m.frequency})" for m in relevant[:5])
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

    @staticmethod
    def _filter_relevant(
        skills: list, discipline_name: str
    ) -> list:
        """Filter truly_missing skills to those plausibly relevant to discipline."""
        dn = discipline_name.lower().strip()
        # Simple heuristic: match discipline name keywords against skill name
        discipline_tokens = set(dn.replace("(", " ").replace(")", " ").replace(",", " ").split())
        # Remove very generic tokens
        stop = {"и", "в", "на", "с", "по", "для", "их", "средства", "технологии", "системы",
                "программного", "обеспечения", "информационных"}
        discipline_tokens -= stop
        if not discipline_tokens:
            return list(skills)

        result = []
        for m in skills:
            sn = m.skill_name.lower().strip()
            if any(t in sn or sn.startswith(t) for t in discipline_tokens if len(t) > 2):
                result.append(m)
        # If filter eliminated everything, return original (conservative — show all)
        return result if result else list(skills)

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
