"""Skill matching: normalize, exact, fuzzy, substring matching."""
from __future__ import annotations

import re
from typing import Any

import structlog

from src.result import Ok, Err, Result
from src.errors import MatchingError

logger = structlog.get_logger(__name__)

NORMALIZE_RE = re.compile(r"[^\w\s-]")


def normalize(s: str) -> str:
    return NORMALIZE_RE.sub("", s.lower().strip())


def coverage_level(ratio: float) -> str:
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.2:
        return "medium"
    return "low"


class SkillMatcher:
    def __init__(self, market_skills: dict[str, int] | None = None):
        self.market_skills: dict[str, int] = market_skills or {}

    def set_market(self, market_skills: dict[str, int]) -> Result[None, MatchingError]:
        if not market_skills:
            logger.warning("market_skills_empty")
            return Err(MatchingError(skill_name="", message="Empty market skills map"))
        self.market_skills = market_skills
        logger.info("market_skills_set", count=len(market_skills))
        return Ok(None)

    def match(self, skill_name: str) -> Result[tuple[str | None, str], MatchingError]:
        n = normalize(skill_name)
        if not n or len(n) < 3:
            logger.debug("skill_too_short", skill=skill_name)
            return Ok((None, "no_match"))

        if n in self.market_skills:
            logger.debug("skill_exact_match", skill=n)
            return Ok((n, "exact"))

        for mn in self.market_skills:
            if n in mn or mn in n:
                logger.debug("skill_fuzzy_match", rpd_skill=n, market_skill=mn)
                return Ok((mn, "fuzzy"))

        return Ok((None, "no_match"))

    def get_emerging(
        self, rpd_normalized: set[str], top_n: int = 10
    ) -> Result[list[tuple[str, int, str]], MatchingError]:
        if not self.market_skills:
            logger.warning("no_market_skills_for_emerging")
            return Err(MatchingError(skill_name="", message="No market skills loaded"))
        result = []
        for mn, mf in sorted(self.market_skills.items(), key=lambda x: -x[1]):
            if mn in rpd_normalized:
                continue
            skip = False
            for rn in rpd_normalized:
                if mn in rn or rn in mn:
                    skip = True
                    break
            if not skip:
                result.append((mn, mf, "emerging"))
                if len(result) >= top_n:
                    break
        logger.info("emerging_skills_found", count=len(result))
        return Ok(result)
