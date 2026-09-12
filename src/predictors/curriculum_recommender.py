"""Curriculum recommender: generates per-discipline curriculum recommendations."""
from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from src.result import Ok, Err, Result
from src.errors import RecommendationError
from src.models.teacher_analysis import Recommendation, DisciplineCoverage
from src.feature_flags import weak_comp_recs_enabled

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
    if _is_knowledge_description(skill):
        return "academic"
    return "generic"


# Categories where market-alignment overhaul orders make no sense (v29).
# A philosophy/math/management course is not rebuilt because vacancies ignore it.
NON_IT_CATS = frozenset({
    "Soft Skills и рефлексия",
    "Математика и статистика",
    "Управление и менеджмент",
})


_KNOWLEDGE_WORDS = (
    "особенност", "поняти", "теори", "фундаментал", "сущност",
    "представлени", "закономерност", "концепци", "методологи",
    "определени", "характеристик", "свойств", "структур",
    "принцип", "основ", "знани",
)


def _is_knowledge_description(skill: str) -> bool:
    """Long RPD knowledge formulation, not a market skill (v9)."""
    s = (skill or "").lower().strip()
    if len(s) < 60:
        return False
    return any(w in s for w in _KNOWLEDGE_WORDS)


_CYRILLIC = re.compile(r"[\u0430-\u044f\u0451]", re.IGNORECASE)


_DANGLING_END = re.compile(
    r"\s(и|в|на|с|к|о|а|но|не|ни|или|для|от|по|из|у|же|ли|бы|как)$",
    re.IGNORECASE,
)


_PUNCT_WS = re.compile("[\\s\"\'«»„“”().,:;!?—–-]")


def _vocab_refs(phrases, vocab) -> set:
    """Market-vocab skills evidenced by discipline phrases (v11)."""
    refs: set = set()
    if not vocab:
        return refs
    for ph in phrases or []:
        words = re.findall(r"\w+", (ph or "").lower(), flags=re.UNICODE)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if gram in vocab:
                    refs.add(gram)
    return refs


def _norm_msg(msg: str) -> str:
    """Declension-insensitive key: lemmas when available, else fold (v10)."""
    try:
        from src.text.ru_morph import lemma_key

        key = lemma_key(msg)
        if key:
            return key
    except Exception:
        pass
    return _PUNCT_WS.sub(" ", (msg or "").casefold()).strip()


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

    def _is_fragment(self, skill: str) -> bool:
        """RPD parser shreds (v9.1): dangling conjunction, or tiny lowercase
        token unknown to taxonomy (sql/erp stay, shreds like 'ческие' go)."""
        s = (skill or "").strip()
        if not s:
            return True
        if _DANGLING_END.search(s):
            return True
        toks = s.split()
        if len(toks) == 1:
            tok = toks[0].strip(chr(0xAB) + chr(0xBB) + chr(34) + chr(39) + chr(40) + chr(41)
                            + ".,;:-")
            if len(tok) <= 6 and tok[:1].islower() and _CYRILLIC.search(tok) \
                    and tok.lower() not in self._taxo_map:
                return True
        return False


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

    def generate(self, coverage: DisciplineCoverage, cooc=None) -> Result[list[Recommendation], RecommendationError]:
        if not coverage:
            logger.error("coverage_none")
            return Err(RecommendationError(message="Coverage data is required"))

        recs = []
        # Discipline profile for relevance gates (v22: shared by add_new + cross_ref).
        matched_names = [m.skill_name for m in (coverage.top_matched or [])]
        matched_cats: set[str] = set()
        for _nm in matched_names:
            matched_cats.update(self._cats_of(_nm))

        recs = []
        # phrase -> competency codes owning it (for actionable messages, v29).
        comp_of: dict[str, list[str]] = {}
        for cc in (coverage.competencies or []):
            for g in (cc.gap_skills or []):
                key = (g or "").lower().strip()
                if key and cc.code not in comp_of.setdefault(key, []):
                    comp_of[key].append(cc.code)

        def _codes(skill: str) -> str:
            codes = comp_of.get((skill or "").lower().strip(), [])[:3]
            return (" (" + ", ".join(codes) + ")") if codes else ""

        # Gaps: RPD skills not found on market — one per skill
        for s in coverage.gaps_list:
            if self._is_fragment(s):
                continue
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

        # v29: attach owning competency codes to gap messages (actionability).
        for r in recs:
            if r.type in ("review_content", "foundational") and r.skill_name:
                suffix = _codes(r.skill_name)
                if suffix and suffix not in r.message:
                    r.message = r.message.rstrip() + suffix

        # Truly missing: market skills not in ANY discipline — только из тех же
        # областей таксономии, что уже покрытые навыки дисциплины (семантика
        # через области, а не глобальный топ: linux в БЖД сюда не попадает).
        if coverage.truly_missing:
            refs_all = _vocab_refs(matched_names, getattr(cooc, "vocab", None)) if cooc is not None else set()
            ranked: list = []
            for m in coverage.truly_missing:
                cats = self._cats_of(m.skill_name)
                shared = cats & matched_cats
                self_hit = (m.skill_name or "").strip().lower() in refs_all
                if not matched_cats or (not shared and not self_hit):
                    continue
                ranked.append((len(shared), getattr(m, "frequency", 0) or 0, m, shared))
            ranked.sort(key=lambda z: (z[0], z[1]), reverse=True)
            # relevance-ranked: shared taxonomy cats, then market frequency (v9).
            # runner keeps final top-5 by embedding relevance; 25-cap below stays.
            for n_shared, freq, m, shared in ranked:
                pri = "high" if n_shared >= 2 and freq >= 1000 else "medium"
                reason = ", ".join(sorted(shared))
                cooc_obj = cooc if cooc is not None else None
                partners = cooc_obj.top_partners(m.skill_name, 3) if cooc_obj is not None else []
                if partners:
                    reason = reason + "; в вакансиях рядом: " + ", ".join(partners)
                cand_low = (m.skill_name or "").strip().lower()
                if refs_all and cand_low not in refs_all:
                    lk = cooc.link(m.skill_name, refs_all)
                    if freq >= 20 and lk < 0.03:
                        rescued = any(
                            cooc.cond(m.skill_name, r) >= 0.03 and cooc.freq.get(r, 0) >= 5
                            for r in refs_all
                        )
                        if not rescued:
                            logger.info("rec_dropped_weaklink", discipline=coverage.discipline_name,
                                        skill=m.skill_name, link=round(lk, 4))
                            continue
                recs.append(Recommendation(
                    type="add_new_content", priority=pri, skill_name=m.skill_name,
                    message=(
                        f"Рассмотрите возможность включения навыка «{m.skill_name}» "
                        f"(частота на рынке: {m.frequency})."
                        f" (смежно: {reason})."
                    ),
                ))
                # Кандидатов больше, чем покажем: финальный топ-5 по релевантности
                # режет раннер (персонализация под дисциплину).
                if sum(1 for r in recs if r.type == "add_new_content") >= 25:
                    break

        # Cross-references: skills taught in other disciplines
        if coverage.cross_references:
            seen: set[str] = set()
            for cr in coverage.cross_references:
                if len((cr.skill_name or "").strip()) < 2:
                    continue
                cr_cats = self._cats_of(cr.skill_name)
                if not matched_cats or not (cr_cats & matched_cats):
                    logger.info("rec_dropped_crossref_uncategorized", discipline=coverage.discipline_name,
                                skill=cr.skill_name)
                    continue
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

        # Low coverage warning: only where the market has authority (v29).
        # Without IT-domain evidence this would spam non-IT disciplines.
        it_evidence = bool(matched_cats - NON_IT_CATS)
        if coverage.coverage_ratio < 0.3 and it_evidence:
            recs.append(Recommendation(
                type="major_revision",
                priority="high",
                message=(
                    f"Низкое покрытие рынка ({coverage.coverage_ratio * 100:.1f}%)."
                    f" Требуется существенный пересмотр дисциплины."
                ),
            ))

        # v29: empty competencies (no parsed skills at all) get an explicit rec.
        for cc in coverage.competencies:
            if cc.coverage == 0 and cc.total_skills == 0 and cc.code:
                recs.append(Recommendation(
                    type="review_content",
                    priority="medium",
                    skill_name=cc.code,
                    message=(
                        f"Компетенция «{cc.code}» пуста: в РПД нет извлечённых пунктов. "
                        f"Добавьте знания/умения/навыки кнопкой +ЗУН в карточке компетенции — "
                        f"иначе её нечем сопоставлять с рынком."
                    ),
                ))

        # Zero-coverage competencies — one per competency
        for cc in coverage.competencies:
            if cc.coverage == 0 and cc.total_skills > 0:
                recs.append(Recommendation(
                    type="review_content",
                    priority="medium",
                    skill_name=cc.code,
                    message=(
                        f"Компетенция «{cc.code}» имеет 0% покрытие рынком"
                        f" — рекомендуется наполнить её востребованными навыками."
                    ),
                ))

        # Дедупликация: убрать повторяющиеся сообщения
        # Weak (0 < cov < 0.5) competencies: one targeted add_new per competency (v32, flag-gated).
        if weak_comp_recs_enabled():
            for cc in coverage.competencies:
                if 0 < cc.coverage < 0.5 and cc.gap_skills:
                    _top = cc.gap_skills[:3]
                    _rest = (
                        f" (и ещё {len(cc.gap_skills) - len(_top)} вне топ-3)"
                        if len(cc.gap_skills) > len(_top)
                        else ""
                    )
                    recs.append(Recommendation(
                        type="add_new_content",
                        priority="medium",
                        skill_name=cc.code,
                        message=(
                            f"Компетенция <{cc.code}>: покрытие {cc.coverage:.0%} — "
                            f"точечно добрать: {', '.join(_top)}{_rest}."
                        ),
                    ))
        seen: set[str] = set()
        seen = set()
        drops: dict = {}
        deduped: list = []
        for r in recs:
            key = _norm_msg(r.message)
            if key in seen:
                drops[key] = drops.get(key, 0) + 1
                continue
            seen.add(key)
            deduped.append(r)
        recs = deduped
        # fold near-duplicate review/foundational sharing first 7 words (v24).
        groups: dict = {}
        order: list = []
        for r in recs:
            if r.type in ("review_content", "foundational"):
                gkey = ("fold", r.type, tuple(_norm_msg(r.message).split()[:7]))
            else:
                gkey = ("single", id(r))
            if gkey not in groups:
                groups[gkey] = []
                order.append(gkey)
            groups[gkey].append(r)
        folded: list = []
        for gkey in order:
            g = groups[gkey]
            extra = len(g) - 1 + sum(drops.get(_norm_msg(m.message), 0) for m in g)
            if extra > 0:
                first = g[0]
                folded.append(Recommendation(
                    type=first.type, priority=first.priority,
                    skill_name=first.skill_name,
                    message=first.message + f" (и ещё {extra} похожих формулировок из РПД).",
                ))
            else:
                folded.append(g[0])
        recs = folded
        recs = self.validate(recs, coverage)

        logger.info("recommendations_generated",
                     discipline=coverage.discipline_name, count=len(recs))
        return Ok(recs)

    def validate(self, recs: list, coverage: DisciplineCoverage) -> list:
        """Hard invariants: noise never reaches teachers even if upstream shifts (v10)."""
        matched = {str(m.skill_name or "").strip().lower() for m in (coverage.top_matched or [])}
        out: list = []
        for r in recs:
            sk = (r.skill_name or "").strip()
            if not r.type or not r.message:
                logger.info("rec_invalid_dropped", reason="empty_type_or_message")
                continue
            if r.type in ("add_new_content", "cross_reference", "review_content", "foundational"):
                if sk and len(sk) < 2:
                    logger.info("rec_invalid_dropped", reason="short_skill", skill=sk)
                    continue
            if r.type == "add_new_content" and sk.lower() in matched:
                logger.info("rec_invalid_dropped", reason="already_covered", skill=sk)
                continue
            out.append(r)
        return out


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
