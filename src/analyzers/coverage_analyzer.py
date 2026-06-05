"""Coverage analyzer: discipline vs market coverage."""
from __future__ import annotations

from collections import Counter

import structlog

from src.result import Ok, Err, Result
from src.errors import CoverageError
from src.models.teacher_analysis import CompetencyCoverage, DisciplineCoverage, SkillMatch
from src.analyzers.skill_matcher import SkillMatcher, coverage_level, normalize as normalize_skill

logger = structlog.get_logger(__name__)


class CoverageAnalyzer:
    def __init__(self, matcher: SkillMatcher):
        self.matcher = matcher

    def analyze_discipline(
        self,
        discipline_id: str,
        discipline_name: str,
        competencies: dict[str, list[str]],
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
            comp_gaps = []
            for s in skills:
                all_rpd.append(s)
                match_result = self.matcher.match(s)
                if match_result.is_err():
                    logger.warning("skill_match_failed", skill=s)
                    continue
                m, _ = match_result.unwrap()
                if m:
                    comp_matched += 1
                else:
                    comp_gaps.append(s)
            comp_results.append(CompetencyCoverage(
                code=ccode,
                total_skills=len(skills),
                matched_skills=comp_matched,
                coverage=round(comp_matched / len(skills), 4) if skills else 0,
                gap_skills=comp_gaps[:10],
            ))

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
            m, mtype = match_result.unwrap()
            if m:
                matched_list.append(SkillMatch(
                    skill_name=s, market_match=m,
                    frequency=self.matcher.market_skills.get(m, 0),
                    match_type=mtype,
                ))
            else:
                gaps_list.append(s)

        emerging_result = self.matcher.get_emerging(rpd_norm, top_n=10)
        emerging_skills: list[SkillMatch] = []
        if emerging_result.is_ok():
            emerging_raw = emerging_result.unwrap()
            emerging_skills = [
                SkillMatch(skill_name=s, frequency=f, match_type="emerging")
                for s, f, _ in emerging_raw
            ]

        total = len(matched_list) + len(gaps_list)
        ratio = round(len(matched_list) / total, 4) if total else 0

        logger.info("discipline_analyzed",
                     discipline=discipline_name,
                     total=total, matched=len(matched_list),
                     gaps=len(gaps_list), coverage=ratio)

        return Ok(DisciplineCoverage(
            discipline_id=discipline_id,
            discipline_name=discipline_name,
            total_skills=total,
            market_matched=len(matched_list),
            gaps=len(gaps_list),
            coverage_ratio=ratio,
            coverage_level=coverage_level(ratio),
            top_matched=sorted(matched_list, key=lambda x: -x.frequency)[:10],
            gaps_list=gaps_list[:20],
            emerging=emerging_skills,
            competencies=comp_results,
        ))
