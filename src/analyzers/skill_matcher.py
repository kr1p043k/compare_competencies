"""Skill matching: normalize, exact, fuzzy, semantic (embedding)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog

from src.result import Ok, Err, Result
from src.errors import MatchingError

logger = structlog.get_logger(__name__)

NORMALIZE_RE = re.compile(r"[^\w\s-]")
SEMANTIC_THRESHOLD = 0.78


def normalize(s: str) -> str:
    return NORMALIZE_RE.sub("", s.lower().strip())


def coverage_level(ratio: float) -> str:
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.2:
        return "medium"
    return "low"


class SkillMatcher:
    def __init__(self, market_skills: dict[str, int] | None = None,
                 embedding_provider: Any | None = None):
        self.market_skills: dict[str, int] = market_skills or {}
        self._embedding_provider = embedding_provider
        self._market_embeddings: np.ndarray | None = None
        self._market_names: list[str] = []
        self._semantic_cache: dict[str, str] = {}

    def set_market(self, market_skills: dict[str, int]) -> Result[None, MatchingError]:
        if not market_skills:
            logger.warning("market_skills_empty")
            return Err(MatchingError(skill_name="", message="Empty market skills map"))
        self.market_skills = market_skills
        self._semantic_cache.clear()
        if self._embedding_provider and market_skills:
            names = list(market_skills.keys())
            embs = self._embedding_provider.encode(names, show_progress_bar=False)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._market_embeddings = embs / norms
            self._market_names = names
        logger.info("market_skills_set", count=len(market_skills))
        return Ok(None)

    @staticmethod
    def _word_match(a: str, b: str) -> bool:
        """True if 'a' appears as a whole word in 'b' (word-boundary aware)."""
        return bool(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b))

    def _semantic_match(self, n: str) -> tuple[str | None, str, float]:
        if n in self._semantic_cache:
            return (self._semantic_cache[n], "semantic", 1.0)
        if self._market_embeddings is None or self._embedding_provider is None:
            return (None, "no_match", 0.0)
        qemb = self._embedding_provider.encode([n])
        qnorm = np.linalg.norm(qemb)
        if qnorm > 0:
            qemb /= qnorm
        sims = self._market_embeddings @ qemb.T
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= SEMANTIC_THRESHOLD:
            self._semantic_cache[n] = self._market_names[best]
            return (self._market_names[best], "semantic", score)
        return (None, "no_match", 0.0)

    def match(self, skill_name: str) -> Result[tuple[str | None, str, float], MatchingError]:
        n = normalize(skill_name)
        if not n or len(n) < 3:
            logger.debug("skill_too_short", skill=skill_name)
            return Ok((None, "no_match", 0.0))

        if n in self.market_skills:
            logger.debug("skill_exact_match", skill=n)
            return Ok((n, "exact", 1.0))

        for mn in self.market_skills:
            if self._word_match(n, mn) or self._word_match(mn, n):
                logger.debug("skill_fuzzy_match", rpd_skill=n, market_skill=mn)
                return Ok((mn, "fuzzy", 0.5))

        mn, mt, score = self._semantic_match(n)
        if mn:
            logger.debug("skill_semantic_match", rpd_skill=n, market_skill=mn)
        return Ok((mn, mt, score))

    def get_emerging(
        self, rpd_normalized: set[str], top_n: int = 10,
        also_exclude: set[str] | None = None,
    ) -> Result[list[tuple[str, int, str]], MatchingError]:
        if not self.market_skills:
            logger.warning("no_market_skills_for_emerging")
            return Err(MatchingError(skill_name="", message="No market skills loaded"))
        result = []
        for mn, mf in sorted(self.market_skills.items(), key=lambda x: -x[1]):
            if mn in rpd_normalized:
                continue
            if also_exclude and mn in also_exclude:
                continue
            skip = False
            for rn in rpd_normalized:
                if self._word_match(mn, rn) or self._word_match(rn, mn):
                    skip = True
                    break
            if also_exclude and not skip:
                for rn in also_exclude:
                    if self._word_match(mn, rn) or self._word_match(rn, mn):
                        skip = True
                        break
            if not skip:
                result.append((mn, mf, "emerging"))
                if len(result) >= top_n:
                    break
        logger.info("emerging_skills_found", count=len(result))
        return Ok(result)
