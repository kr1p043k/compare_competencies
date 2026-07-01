"""Coverage analyzer: discipline vs market coverage."""
from __future__ import annotations

from collections import Counter

import numpy as np
import structlog

from src.result import Ok, Err, Result
from src.errors import CoverageError
from src.models.teacher_analysis import CompetencyCoverage, CrossReference, DisciplineCoverage, SkillMatch
from src.analyzers.skill_matcher import SkillMatcher, coverage_level, normalize as normalize_skill

logger = structlog.get_logger(__name__)


class CoverageAnalyzer:
    def __init__(self, matcher: SkillMatcher, discipline_scorer=None):
        self.matcher = matcher
        self._discipline_scorer = discipline_scorer

    def analyze_discipline(
        self,
        discipline_id: str,
        discipline_name: str,
        competencies: dict[str, list[str]],
        direction_rpd_norm: set[str] | None = None,
        discipline_skill_map: dict[str, set[str]] | None = None,
    ) -> Result[DisciplineCoverage, CoverageError]:
        if not discipline_id:
            logger.error("missing_discipline_id")
            return Err(CoverageError(message="discipline_id is required", discipline_id=""))
        if not competencies:
            logger.warning("discipline_no_competencies", discipline=discipline_name)
            return Err(CoverageError(
                message=f"No competencies for {discipline_name}",
                discipline_id=discipline_id,
            ))

        all_rpd: list[str] = []
        comp_results: list[CompetencyCoverage] = []

        for ccode, skills in competencies.items():
            comp_matched = 0
            comp_weighted = 0.0
            comp_gaps = []
            for s in skills:
                all_rpd.append(s)
                match_result = self.matcher.match(s)
                if match_result.is_err():
                    logger.warning("skill_match_failed", skill=s)
                    continue
                m, _, conf = match_result.unwrap()
                if m:
                    comp_matched += 1
                    comp_weighted += conf
                else:
                    comp_gaps.append(s)
            n = len(skills)
            comp_results.append(CompetencyCoverage(
                code=ccode,
                total_skills=n,
                matched_skills=comp_matched,
                coverage=round(comp_matched / n, 4) if n else 0,
                weighted_coverage=round(comp_weighted / n, 4) if n else 0,
                gap_skills=comp_gaps[:10],
            ))

        # Aggregate parent competency coverage from children
        # Parent: "ОПК-3" (no dot), Child: "ОПК-3.1" (has dot)
        child_by_parent: dict[str, list[CompetencyCoverage]] = {}
        for cc in comp_results:
            if "." in cc.code:
                parent_code = cc.code.rsplit(".", 1)[0]
                child_by_parent.setdefault(parent_code, []).append(cc)
        if child_by_parent:
            for idx, cc in enumerate(comp_results):
                children = child_by_parent.get(cc.code)
                if children and cc.total_skills == 0:
                    total = sum(c.total_skills for c in children)
                    matched = sum(c.matched_skills for c in children)
                    weighted = sum(c.matched_skills * c.weighted_coverage for c in children) / max(matched, 1)
                    comp_results[idx] = CompetencyCoverage(
                        code=cc.code,
                        total_skills=total,
                        matched_skills=matched,
                        coverage=round(matched / total, 4) if total else 0,
                        weighted_coverage=round(weighted, 4) if total else 0,
                        gap_skills=cc.gap_skills,
                    )

        matched_list: list[SkillMatch] = []
        gaps_list: list[str] = []
        rpd_norm: set[str] = set()

        for s in all_rpd:
            n = normalize_skill(s)
            if not n or len(n) < 3:
                continue
            rpd_norm.add(n)
            match_result = self.matcher.match(s)
            if match_result.is_err():
                continue
            m, mtype, conf = match_result.unwrap()
            if m:
                matched_list.append(SkillMatch(
                    skill_name=s, market_match=m,
                    frequency=self.matcher.market_skills.get(m, 0),
                    match_type=mtype, confidence=conf,
                ))
            else:
                gaps_list.append(s)

        # Per-discipline emerging (not in THIS discipline)
        emerging_result = self.matcher.get_emerging(rpd_norm, top_n=10)
        emerging_skills: list[SkillMatch] = []
        if emerging_result.is_ok():
            emerging_raw = emerging_result.unwrap()
            emerging_skills = [
                SkillMatch(skill_name=s, frequency=f, match_type="emerging")
                for s, f, _ in emerging_raw
            ]

        # Cross-discipline: truly missing vs covered elsewhere
        truly_missing: list[SkillMatch] = []
        cross_refs: list[CrossReference] = []
        if direction_rpd_norm is not None and discipline_skill_map is not None:
            dir_result = self.matcher.get_emerging(
                rpd_norm, top_n=30,
                also_exclude=direction_rpd_norm,
            )
            if dir_result.is_ok():
                dir_emerging = {s for s, _, _ in dir_result.unwrap()}
                for em in emerging_skills:
                    if em.skill_name not in dir_emerging:
                        found = False
                        if self._discipline_scorer is not None:
                            self._discipline_scorer._ensure_embeddings()
                            for dn in discipline_skill_map:
                                if dn == discipline_name:
                                    continue
                                disc_emb = self._discipline_scorer.get_discipline_embedding(dn)
                                if disc_emb is not None:
                                    from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory
                                    prov = EmbeddingProviderFactory.get()
                                    sk_emb = prov.encode([em.skill_name], show_progress_bar=False)[0]
                                    sk_norm = np.linalg.norm(sk_emb)
                                    if sk_norm > 0:
                                        sk_emb = sk_emb / sk_norm
                                        sim = float(np.dot(sk_emb, disc_emb))
                                        if sim > 0.55:
                                            cross_refs.append(CrossReference(
                                                skill_name=em.skill_name,
                                                frequency=em.frequency,
                                                discipline=dn,
                                            ))
                                            found = True
                                            break
                        if not found:
                            for dn, dskills in discipline_skill_map.items():
                                if dn == discipline_name:
                                    continue
                                if em.skill_name in dskills:
                                    cross_refs.append(CrossReference(
                                        skill_name=em.skill_name,
                                        frequency=em.frequency,
                                        discipline=dn,
                                    ))
                                    found = True
                                    break
                        if not found:
                            for dn, dskills in discipline_skill_map.items():
                                if dn == discipline_name:
                                    continue
                                for rn in dskills:
                                    if (self.matcher._word_match(em.skill_name, rn)
                                        or self.matcher._word_match(rn, em.skill_name)):
                                        cross_refs.append(CrossReference(
                                            skill_name=em.skill_name,
                                            frequency=em.frequency,
                                            discipline=dn,
                                        ))
                                        found = True
                                        break
                                if found:
                                    break
                truly_raw = self.matcher.get_emerging(
                    direction_rpd_norm, top_n=10,
                )
                if truly_raw.is_ok():
                    truly_missing = [
                        SkillMatch(skill_name=s, frequency=f, match_type="emerging")
                        for s, f, _ in truly_raw.unwrap()
                    ]

        total = len(matched_list) + len(gaps_list)
        ratio = round(len(matched_list) / total, 4) if total else 0
        weighted = round(sum(m.confidence for m in matched_list) / total, 4) if total else 0.0

        logger.info("discipline_analyzed",
                     discipline=discipline_name,
                     total=total, matched=len(matched_list),
                     gaps=len(gaps_list), coverage=ratio, weighted=weighted)

        return Ok(DisciplineCoverage(
            discipline_id=discipline_id,
            discipline_name=discipline_name,
            total_skills=total,
            market_matched=len(matched_list),
            gaps=len(gaps_list),
            coverage_ratio=ratio,
            weighted_coverage=weighted,
            coverage_level=coverage_level(ratio),
            top_matched=sorted(matched_list, key=lambda x: -x.frequency)[:10],
            gaps_list=gaps_list[:20],
            emerging=emerging_skills,
            truly_missing=truly_missing,
            cross_references=cross_refs,
            competencies=comp_results,
        ))
