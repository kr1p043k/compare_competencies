"""Skill matching: normalize, exact, fuzzy, semantic (embedding)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog

from src.result import Ok, Err, Result
from src.errors import MatchingError

logger = structlog.get_logger(__name__)

NORMALIZE_RE = re.compile(r"[^\w\s\-/]")
SEMANTIC_THRESHOLD = 0.78
MARKET_EMB_CACHE_NAME = "market_embeddings_middle.joblib"
MARKET_CACHE_MIN_SKILLS = 300


def normalize(s: str) -> str:
    return NORMALIZE_RE.sub("", s.lower().strip())


def coverage_level(ratio: float) -> str:
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.2:
        return "medium"
    return "low"


def _word_pattern(word: str) -> re.Pattern[str]:
    """Precompiled regex matching `word` as a whole word (word-boundary aware)."""
    return re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)")


def adaptive_semantic_threshold(query: str, candidate: str, base: float = 0.70) -> float:
    """Адаптивный порог (вывод Exp1: фикс. 0.78 режет recall вдвое).

    Base 0.70; -0.05 cross-script; -0.05 long query; +0.10 very short query.
    Clamped to [0.55, 0.85].
    """
    t = base
    has_cyr_q = bool(re.search(r"[а-яёА-ЯЁ]", query))
    has_lat_c = bool(re.search(r"[a-zA-Z]{3,}", candidate))
    has_cyr_c = bool(re.search(r"[а-яёА-ЯЁ]", candidate))
    has_lat_q = bool(re.search(r"[a-zA-Z]{3,}", query))
    if (has_cyr_q and has_lat_c) or (has_cyr_c and has_lat_q):
        t -= 0.05
    if len(query) > 40:
        t -= 0.05
    if len(query.strip()) < 5:
        t += 0.10
    return max(0.55, min(0.85, t))


class SkillMatcher:
    def __init__(self, market_skills: dict[str, int] | None = None,
                 embedding_provider: Any | None = None):
        self.market_skills: dict[str, int] = market_skills or {}
        self._embedding_provider = embedding_provider
        self._market_embeddings: np.ndarray | None = None
        self._market_names: list[str] = []
        # Precompiled word-boundary patterns per market name, in dict order,
        # so fuzzy matching stays 100% equivalent to the old per-name loop
        # (which compiled a regex on every call inside the hot path).
        self._market_word_pats: list[tuple[str, re.Pattern[str]]] = []
        self._semantic_cache: dict[str, str] = {}
        self._match_cache: dict[str, tuple[str | None, str, float]] = {}
        self._rebuild_fuzzy_patterns()

    def set_market(self, market_skills: dict[str, int]) -> Result[None, MatchingError]:
        """Загрузить навыки рынка, предвычислить эмбеддинги, сбросить кэши."""
        if not market_skills:
            logger.warning("market_skills_empty")
            return Err(MatchingError(skill_name="", message="Empty market skills map"))
        self.market_skills = dict(sorted(market_skills.items(), key=lambda kv: (-kv[1], kv[0])))
        self._semantic_cache.clear()
        self._match_cache.clear()
        self._rebuild_fuzzy_patterns()
        if self._embedding_provider and market_skills:
            names = list(market_skills.keys())
            embs = None
            if len(names) >= MARKET_CACHE_MIN_SKILLS:
                embs = self._try_load_market_embs(names)
            if embs is None:
                embs = self._embedding_provider.encode(names, show_progress_bar=False)
                if len(names) >= MARKET_CACHE_MIN_SKILLS:
                    self._save_market_embs(names, embs)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._market_embeddings = embs / norms
            self._market_names = names
        logger.info("market_skills_set", count=len(market_skills))
        return Ok(None)

    def _market_cache_path(self):
        from src import config

        return config.EMBEDDINGS_CACHE_DIR / MARKET_EMB_CACHE_NAME

    def _try_load_market_embs(self, names: list[str]) -> np.ndarray | None:
        """Загружает кэш эмбеддингов рынка (формат совместим с EmbeddingComparator)."""
        try:
            import joblib

            cache_path = self._market_cache_path()
            if not cache_path.exists():
                return None
            manifest_ok = False
            try:
                from src.artifacts import ArtifactManifest

                match ArtifactManifest.load(cache_path):
                    case Ok(manifest):
                        match manifest.is_compatible():
                            case Ok(True):
                                manifest_ok = True
                            case _:
                                logger.info("market_emb_cache_invalidated_by_model")
                    case Err(err):
                        logger.warning("market_emb_cache_manifest_load_failed", error=str(err))
            except ImportError:
                manifest_ok = True
            if not manifest_ok:
                return None
            loaded = joblib.load(cache_path)
            skills = loaded["skills"] if isinstance(loaded, dict) else loaded[1]
            embs = loaded["embeddings"] if isinstance(loaded, dict) else loaded[0]
            if len(skills) != len(embs) or set(skills) != set(names):
                return None
            row_by_skill = {s: i for i, s in enumerate(skills)}
            idx = np.fromiter((row_by_skill[n] for n in names), dtype=np.int64, count=len(names))
            logger.info("market_embeddings_cache_reused", count=len(names))
            return np.asarray(embs)[idx]
        except Exception as exc:
            logger.debug("market_emb_cache_unavailable", error=str(exc))
            return None

    def _save_market_embs(self, names: list[str], embs: np.ndarray) -> None:
        """Атомарно сохраняет сырые эмбеддинги + манифест (как EmbeddingComparator)."""
        tmp_path = None
        try:
            import os
            import tempfile

            import joblib

            cache_path = self._market_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".joblib.tmp")
            os.close(fd)
            joblib.dump({"embeddings": embs, "skills": names}, tmp_path)
            os.replace(tmp_path, cache_path)
            tmp_path = None
            try:
                from src.artifacts import ArtifactManifest

                ArtifactManifest(
                    artifact_path=cache_path,
                    metrics={"num_skills": len(names)},
                ).save()
            except Exception as exc:
                logger.warning("market_emb_cache_manifest_save_failed", error=str(exc))
            logger.info("market_embeddings_cache_saved", count=len(names), path=str(cache_path))
        except Exception as exc:
            logger.warning("market_emb_cache_save_failed", error=str(exc))
            if tmp_path:
                import contextlib

                with contextlib.suppress(Exception):
                    os.unlink(tmp_path)

    def _rebuild_fuzzy_patterns(self) -> None:
        """Precompile whole-word patterns for every market name.

        `_fuzzy_hit_name` then scans the precompiled list instead of calling
        `re.search` with a freshly built pattern per iteration.
        """
        self._market_word_pats = [
            (name, _word_pattern(name)) for name in self.market_skills
        ]

    def _fuzzy_hit_name(self, n: str) -> str | None:
        """First market name having a whole-word overlap with `n`.

        Identical semantics and order to the original loop:
        for mn in market_skills:
            if _word_match(n, mn) or _word_match(mn, n): return mn
        """
        if not self._market_word_pats:
            return None
        qpat = _word_pattern(n)
        for mn, mpat in self._market_word_pats:
            if qpat.search(mn) or mpat.search(n):
                return mn
        return None

    @staticmethod
    def _word_match(a: str, b: str) -> bool:
        """True if 'a' appears as a whole word in 'b' (word-boundary aware)."""
        return bool(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b))

    def _semantic_match(self, n: str, original: str | None = None) -> tuple[str | None, str, float]:
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
        _thresh = adaptive_semantic_threshold(original or n, self._market_names[best])
        if score >= _thresh:
            self._semantic_cache[n] = self._market_names[best]
            return (self._market_names[best], "semantic", score)
        return (None, "no_match", 0.0)

    def match(self, skill_name: str) -> Result[tuple[str | None, str, float], MatchingError]:
        """Сопоставить навык: exact -> fuzzy -> typo? -> semantic. Возвращает (совпадение, тип, уверенность)."""
        n = normalize(skill_name)
        if not n or len(n) < 3:
            logger.debug("skill_too_short", skill=skill_name)
            return Ok((None, "no_match", 0.0))

        cached = self._match_cache.get(n)
        if cached is not None:
            return Ok(cached)

        result = self._match_uncached(n, skill_name)
        if result.is_ok():
            self._match_cache[n] = result.unwrap()
        return result

    def _match_uncached(self, n: str, skill_name: str) -> Result[tuple[str | None, str, float], MatchingError]:
        if n in self.market_skills:
            logger.debug("skill_exact_match", skill=n)
            return Ok((n, "exact", 1.0))

        mn = self._fuzzy_hit_name(n)
        if mn:
            logger.debug("skill_fuzzy_match", rpd_skill=n, market_skill=mn)
            return Ok((mn, "fuzzy", 0.5))

        mn, mt, score = self._semantic_match(n, skill_name)
        if mn:
            logger.debug("skill_semantic_match", rpd_skill=n, market_skill=mn)
        return Ok((mn, mt, score))

    def get_emerging(
        self, rpd_normalized: set[str], top_n: int = 10,
        also_exclude: set[str] | None = None,
    ) -> Result[list[tuple[str, int, str]], MatchingError]:
        """Топ-N навыков рынка, отсутствующих в переданном наборе."""
        if not self.market_skills:
            logger.warning("no_market_skills_for_emerging")
            return Err(MatchingError(skill_name="", message="No market skills loaded"))

        rpd_pats = [(_word_pattern(r)) for r in rpd_normalized] if rpd_normalized else []
        excl_pats = [(_word_pattern(r)) for r in also_exclude] if also_exclude else []

        result = []
        for mn, mf in sorted(self.market_skills.items(), key=lambda x: (-x[1], x[0])):
            if not (mn or "").strip():
                continue
            if mn in rpd_normalized:
                continue
            if also_exclude and mn in also_exclude:
                continue
            # Equivalent to the original nested loop:
            #   for rn in rpd_normalized:
            #       if _word_match(mn, rn) or _word_match(rn, mn): skip
            # where _word_match(mn, rn) == mn word inside rn
            # and   _word_match(rn, mn) == rn word inside mn
            qpat = _word_pattern(mn)
            skip = False
            if rpd_pats:
                if any(p.search(mn) for p in rpd_pats):
                    skip = True
                elif any(qpat.search(rn) for rn in rpd_normalized):
                    skip = True
            if also_exclude and not skip and excl_pats:
                if any(p.search(mn) for p in excl_pats):
                    skip = True
                elif any(qpat.search(rn) for rn in also_exclude):
                    skip = True
            if not skip:
                result.append((mn, mf, "emerging"))
                if len(result) >= top_n:
                    break
        logger.info("emerging_skills_found", count=len(result))
        return Ok(result)
