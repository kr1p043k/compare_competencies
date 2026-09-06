"""Гибридный матчер навыков для статьи: RapidFuzz И границы слов + адаптивная семантика.

Дизайн (выводы Exp1-Exp6):
- Только word-containment: recall опечаток 0.07, FP ловушек 0.67.
- Только RapidFuzz@70: recall опечаток 1.00, FP ловушек 1.00.
- Гибридный fuzzy = RapidFuzz И сторож границ: устойчив к опечаткам И точен на ловушках.
- Порог семантики адаптивный (не фикс. 0.78): база 0.70,
  -0.05 кросс-язык, -0.05 длинные запросы, +0.10 короткие, clamp [0.55, 0.85].
- Уверенность: exact 1.0; hybrid_fuzzy 0.55..1.0 из RF-скора;
  semantic = сырой косинус.

API-совместим со SkillMatcher: match(), set_market(), get_emerging().
Тип hybrid_fuzzy отличает от legacy fuzzy в бенчмарках.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog
from rapidfuzz import fuzz

from src.errors import MatchingError
from src.result import Err, Ok, Result

logger = structlog.get_logger(__name__)

NORMALIZE_RE = re.compile(r"[^\w\s\-/]")
RF_THRESHOLD = 70
RF_HIGH = 88
SEM_BASE = 0.70
SEM_MIN = 0.55
SEM_MAX = 0.85


def normalize(s: str) -> str:
    """Нижний регистр + чистка не-словных символов (как SkillMatcher.normalize)."""
    return NORMALIZE_RE.sub("", s.lower().strip())


def _has_cyrillic(s: str) -> bool:
    """True, если в строке есть кириллица."""
    return bool(re.search(r"[а-яёА-ЯЁ]", s))


def _has_latin_word(s: str) -> bool:
    """True, если есть латинское слово из 3+ букв."""
    return bool(re.search(r"[a-zA-Z]{3,}", s))


def _is_strict_prefix_trap(query: str, candidate: str) -> bool:
    """True, если запрос — собственный префикс кандидата БЕЗ границы слова.

    e.g. "java" vs "javascript": prefix + next char 's' is alnum -> trap.
    e.g. "react" vs "react native": next char is space -> NOT a trap.
    """
    q, c = query.lower(), candidate.lower()
    if q == c:
        return False
    if c.startswith(q) and len(c) > len(q):
        nxt = c[len(q)]
        if nxt.isalnum():
            return True
    if q.startswith(c) and len(q) > len(c):
        nxt = q[len(c)]
        if nxt.isalnum():
            return True
    return False


def _word_contained(a: str, b: str) -> bool:
    """Вхождение целым словом в любую сторону (семантика legacy fuzzy)."""
    try:
        return bool(
            re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b)
            or re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", a)
        )
    except Exception:
        return False


def adaptive_threshold(query: str, candidate: str) -> float:
    """Адаптивный порог косинусной близости для семантики."""
    t = SEM_BASE
    # Cross-script (RU query <-> EN skill): embeddings score lower, be lenient
    if (_has_cyrillic(query) and _has_latin_word(candidate)) or (
        _has_cyrillic(candidate) and _has_latin_word(query)
    ):
        t -= 0.05
    # Long queries dilute similarity: be lenient
    if len(query) > 40:
        t -= 0.05
    # Very short queries are noisy: be strict
    if len(normalize(query)) < 5:
        t += 0.10
    return max(SEM_MIN, min(SEM_MAX, t))


def fuzzy_confidence(rf_score: float) -> float:
    """Калибровка RapidFuzz 70..100 -> 0.55..1.0."""
    return round(0.55 + 0.45 * (rf_score - RF_THRESHOLD) / (100 - RF_THRESHOLD), 4)


class HybridMatcher:
    """Замена SkillMatcher: гибридный fuzzy + адаптивная семантика."""

    def __init__(
        self,
        market_skills: dict[str, int] | None = None,
        embedding_provider: Any | None = None,
        rf_threshold: float = RF_THRESHOLD,
        sem_base: float = SEM_BASE,
    ):
        """Создать матчер. Контракт как у SkillMatcher (drop-in замена)."""
        self.market_skills: dict[str, int] = market_skills or {}
        self._embedding_provider = embedding_provider
        self._rf_threshold = rf_threshold
        self._sem_base = sem_base
        self._market_embeddings: np.ndarray | None = None
        self._market_names: list[str] = []
        self._match_cache: dict[str, tuple[str | None, str, float]] = {}

    # -- market management (same contract as SkillMatcher) --
    def set_market(self, market_skills: dict[str, int]) -> Result[None, MatchingError]:
        """Загрузить навыки рынка + предвычислить эмбеддинги. Чистит кэши."""
        if not market_skills:
            logger.warning("market_skills_empty")
            return Err(MatchingError(skill_name="", message="Empty market skills map"))
        self.market_skills = market_skills
        self._match_cache.clear()
        if self._embedding_provider and market_skills:
            names = list(market_skills.keys())
            embs = self._embedding_provider.encode(names, show_progress_bar=False)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._market_embeddings = embs / norms
            self._market_names = names
        logger.info("hybrid_market_set", count=len(market_skills))
        return Ok(None)

    # -- core matching --
    def match(self, skill_name: str) -> Result[tuple[str | None, str, float], MatchingError]:
        """Матчинг: exact -> hybrid_fuzzy (RF И границы) -> адаптивная семантика."""
        n = normalize(skill_name)
        if not n or len(n) < 3:
            return Ok((None, "no_match", 0.0))
        cached = self._match_cache.get(n)
        if cached is not None:
            return Ok(cached)
        result = self._match_uncached(skill_name, n)
        if result.is_ok():
            self._match_cache[n] = result.unwrap()
        return result

    def _match_uncached(
        self, original: str, n: str
    ) -> Result[tuple[str | None, str, float], MatchingError]:
        """Пайплайн без кэша. `n` — нормализованный запрос."""
        # Stage 1: exact
        if n in self.market_skills:
            return Ok((n, "exact", 1.0))
        for key in self.market_skills:
            if normalize(key) == n:
                return Ok((key, "exact", 1.0))

        # Stage 2: hybrid fuzzy = RapidFuzz AND boundary guard
        best_hf: tuple[str | None, float] = (None, 0.0)
        for mn in self.market_skills:
            if len(mn) < 3:
                continue
            score = fuzz.WRatio(n, normalize(mn))
            if score < self._rf_threshold:
                continue
            # Boundary guard: reject strict prefix traps
            if _is_strict_prefix_trap(n, normalize(mn)):
                continue
            if score > best_hf[1]:
                best_hf = (mn, float(score))
        if best_hf[0] is not None:
            conf = fuzzy_confidence(best_hf[1])
            logger.debug("hybrid_fuzzy_match", query=n, match=best_hf[0], score=best_hf[1])
            return Ok((best_hf[0], "hybrid_fuzzy", conf))

        # Stage 3: adaptive semantic
        if self._market_embeddings is not None and self._embedding_provider is not None:
            try:
                qemb = self._embedding_provider.encode([original])
                qnorm = np.linalg.norm(qemb)
                if qnorm > 0:
                    qemb = qemb / qnorm
                sims = self._market_embeddings @ qemb.T
                best = int(np.argmax(sims))
                score = float(sims[best])
                cand = self._market_names[best]
                thresh = adaptive_threshold(original, cand)
                if score >= thresh:
                    return Ok((cand, "semantic", round(score, 4)))
            except Exception as exc:
                logger.debug("hybrid_semantic_failed", error=str(exc))
        return Ok((None, "no_match", 0.0))

    def get_emerging(
        self,
        rpd_normalized: set[str],
        top_n: int = 10,
        also_exclude: set[str] | None = None,
    ) -> Result[list[tuple[str, int, str]], MatchingError]:
        """Делегирование word-containment логике (как SkillMatcher, для сравнимости)."""
        if not self.market_skills:
            return Err(MatchingError(skill_name="", message="No market skills loaded"))
        result = []
        for mn, mf in sorted(self.market_skills.items(), key=lambda x: -x[1]):
            if mn in rpd_normalized:
                continue
            if also_exclude and mn in also_exclude:
                continue
            skip = False
            for rn in rpd_normalized:
                if _word_contained(mn, rn):
                    skip = True
                    break
            if also_exclude and not skip:
                for rn in also_exclude:
                    if _word_contained(mn, rn):
                        skip = True
                        break
            if not skip:
                result.append((mn, mf, "emerging"))
                if len(result) >= top_n:
                    break
        return Ok(result)
