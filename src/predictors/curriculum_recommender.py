"""Curriculum recommender: generates per-discipline curriculum recommendations."""
from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from src.result import Ok, Err, Result
from src.errors import RecommendationError
from src.models.teacher_analysis import Recommendation, DisciplineCoverage

logger = structlog.get_logger(__name__)

SKILL_TYPES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "skill_types.json"
TAXONOMY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "skill_taxonomy.json"


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
        self._taxo_map: dict[str, set[str]] = {}
        try:
            if TAXONOMY_PATH.exists():
                import json as _json
                _tax = _json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
                _cats = _tax.get("categories", {})
                _items = _cats.values() if isinstance(_cats, dict) else _cats
                for _c in _items:
                    _label = _c.get("label", "") if isinstance(_c, dict) else ""
                    for _s in (_c.get("skills", []) if isinstance(_c, dict) else []):
                        self._taxo_map.setdefault(str(_s).lower(), set()).add(_label)
        except Exception as exc:
            logger.warning("taxonomy_load_failed", error=str(exc))

    def _cats_of(self, skill_name: str) -> set[str]:
        """Области таксономии навыка: короткие эталоны — только по границам слов.

        Иначе «erp» матчится в «enterprise», «go» — в «django», и глобальный
        топ пролезает в чужие дисциплины.
        """
        sn = (skill_name or "").lower().strip()
        if not sn:
            return set()
        words = set(re.findall(r"\w+", sn, flags=re.UNICODE))
        cats: set[str] = set()
        for ref, labels in self._taxo_map.items():
            if not ref:
                continue
            if ref == sn or ref in words:
                cats.update(labels)
            elif len(ref) > 4 and ref in sn:
                cats.update(labels)
            elif len(sn) > 4 and sn in ref:
                cats.update(labels)
        return cats

    def generate(self, coverage: DisciplineCoverage) -> Result[list[Recommendation], RecommendationError]:
        if not coverage:
            logger.error("coverage_none")
            return Err(RecommendationError(message="Coverage data is required"))

        recs = []

        # Gaps: RPD skills not found on market — one per skill
        for s in coverage.gaps_list:
            cls = _classify_skill(s, self.skill_types)
            if cls == "academic":
                recs.append(Recommendation(
                    type="foundational", priority="low", skill_name=s,
                    message=f"«{s}» — фундаментальный навык, не обнаружен на рынке. Не требует замены.",
                ))
            else:
                recs.append(Recommendation(
                    type="review_content", priority="medium", skill_name=s,
                    message=f"«{s}» — навык из РПД не обнаружен в рыночных данных. Рекомендуется пересмотреть его актуальность.",
                ))

        # Truly missing: market skills not in ANY discipline — только из тех же
        # областей таксономии, что уже покрытые навыки дисциплины (семантика
        # через области, а не глобальный топ: linux в БЖД сюда не попадает).
        if coverage.truly_missing:
            matched_names = [m.skill_name for m in (coverage.top_matched or [])]
            matched_cats: set[str] = set()
            for _nm in matched_names:
                matched_cats.update(self._cats_of(_nm))
            for m in coverage.truly_missing:
                if not matched_cats or not (self._cats_of(m.skill_name) & matched_cats):
                    continue
                recs.append(Recommendation(
                    type="add_new_content", priority="medium", skill_name=m.skill_name,
                    message=(
                        f"Рассмотрите возможность включения навыка «{m.skill_name}» "
                        f"(частота на рынке: {m.frequency})."
                    ),
                ))
                if sum(1 for r in recs if r.type == "add_new_content") >= 5:
                    break

        # Cross-references: skills taught in other disciplines
        if coverage.cross_references:
            seen: set[str] = set()
            for cr in coverage.cross_references:
                if cr.skill_name in seen:
                    continue
                seen.add(cr.skill_name)
                recs.append(Recommendation(
                    type="cross_reference", priority="low", skill_name=cr.skill_name,
                    message=(
                        f"Навык «{cr.skill_name}» ({cr.frequency}) преподаётся "
                        f"в дисциплине «{cr.discipline}» — в рамках текущей дисциплины "
                        f"достаточно упомянуть или сослаться."
                    ),
                ))

        # Low coverage warning
        if coverage.coverage_ratio < 0.3:
            recs.append(Recommendation(
                type="major_revision",
                priority="high",
                message=(
                    f"Низкое покрытие рынка ({coverage.coverage_ratio * 100:.1f}%)."
                    f" Требуется существенный пересмотр дисциплины."
                ),
            ))

        # Zero-coverage competencies — one per competency
        for cc in coverage.competencies:
            if cc.coverage == 0 and cc.total_skills > 0:
                recs.append(Recommendation(
                    type="review_content",
                    priority="medium",
                    message=(
                        f"Компетенция «{cc.code}» имеет 0% покрытие рынком"
                        f" — рекомендуется наполнить её востребованными навыками."
                    ),
                ))

        # Дедупликация: убрать повторяющиеся сообщения
        seen: set[str] = set()
        recs = [r for r in recs if not (r.message in seen or seen.add(r.message))]

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
            for e in top_emerging[:10]:
                recs.append(Recommendation(
                    type="add_new_content",
                    priority="high",
                    message=(
                        f"Рассмотрите внедрение навыка «{e['skill']}» "
                        f"(частота на рынке: {e['frequency']})."
                    ),
                ))

        logger.info("summary_recommendations_generated", count=len(recs))
        return Ok(recs)
