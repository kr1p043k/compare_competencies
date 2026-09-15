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
# Single-char market tokens are почти всегда мусор парсинга (напр. 'я').
# Allowlist: языки с однобуквенным именем. Остальное режется везде (v28).
_MARKET_SINGLE_ALLOW = frozenset({"r", "c"})
SEMANTIC_THRESHOLD = 0.78
# A market token with fewer vacancies is fringe: it must neither count as
# coverage (see CoverageAnalyzer) nor shadow stronger stages (v36: e.g. the
# fringe token 'документация'/1 hijacked 'отчетная документация' via fuzzy
# and blocked the mapped hit 'техническая документация'/63).
MARKET_MIN_FREQ = 5
# Leading competency verbs stripped before matching (v44, flag-gated).
# RPD speaks in actions ('владеть X'), the market in nouns ('X').
_LEAD_VERB_LEMMAS = frozenset({
    "владеть", "знать", "уметь", "применять", "понимать", "обладать",
})
# Fallback wordforms when pymorphy is unavailable.
_LEAD_VERB_FORMS = frozenset({
    "владеть", "владеет", "владеют", "знать", "знает", "знают",
    "уметь", "умеет", "умеют", "применять", "применяет", "применяют",
    "понимать", "понимает", "понимают", "обладать", "обладает",
})


def strip_lead_verbs(text: str) -> str:
    """Drop leading competency verbs ("знать python" -> "python"). Deterministic."""
    words = (text or "").split()
    try:
        from src.text.ru_morph import lemma as _lemma, available as _avail
        use_morph = _avail()
    except Exception:
        use_morph = False
    i = 0
    while i < len(words):
        w = words[i].lower()
        is_verb = (_lemma(w) in _LEAD_VERB_LEMMAS) if use_morph else (w in _LEAD_VERB_FORMS)
        if not is_verb:
            break
        i += 1
    stripped = " ".join(words[i:])
    return stripped or (text or "")
# Baseline giants (measured 13.09.2026: 10 skills with freq >= 1227 appear in
# 42-49 of 49 emerging lists) carry no per-discipline signal. Emerging keeps
# only freq < cap (kubernetes/1199 survives as genuine infra signal).
EMERGING_MAX_FREQ = 1200
# Version-split market aliases folded into canonical keys at market build
# (v32, flag-gated via FF_MARKET_SYNONYMS). Minimal grounded set: each alias
# verified to denote the same tool (DB check 12.09.2026: python3 freq 1,
# 'python 3' freq 8 - both the Python language; java 17/11/21 deliberately
# NOT folded - versions can matter for curriculum).
MARKET_SYNONYMS: dict[str, str] = {"python3": "python", "python 3": "python"}


def fold_market_synonyms(market: dict[str, int]) -> dict[str, int]:
    """Return a copy with aliases merged into canonical keys (freq summed). Deterministic."""
    out = dict(market)
    for alias in sorted(MARKET_SYNONYMS):
        if alias in out:
            canon = MARKET_SYNONYMS[alias]
            out[canon] = out.get(canon, 0) + out.pop(alias)
    return out
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
        self.market_skills: dict[str, int] = self._clean_market(market_skills or {})
        self._embedding_provider = embedding_provider
        self._market_embeddings: np.ndarray | None = None
        self._market_names: list[str] = []
        # Precompiled word-boundary patterns per market name, in dict order,
        # so fuzzy matching stays 100% equivalent to the old per-name loop
        # (which compiled a regex on every call inside the hot path).
        self._market_word_pats: list[tuple[str, re.Pattern[str]]] = []
        self._market_lemma_pats: list[tuple[str, re.Pattern[str]]] = []
        self._semantic_cache: dict[str, str] = {}
        self._match_cache: dict[str, tuple[str | None, str, float]] = {}
        self._rebuild_fuzzy_patterns()

    def set_market(self, market_skills: dict[str, int]) -> Result[None, MatchingError]:
        """Загрузить навыки рынка, предвычислить эмбеддинги, сбросить кэши."""
        if not market_skills:
            logger.warning("market_skills_empty")
            return Err(MatchingError(skill_name="", message="Empty market skills map"))
        self.market_skills = dict(sorted(self._clean_market(market_skills).items(), key=lambda kv: (-kv[1], kv[0])))
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

    @staticmethod
    def _clean_market(market_skills: dict[str, int]) -> dict[str, int]:
        """Drop falsy and single-char junk tokens (v28: market contained 'я')."""
        return {k: v for k, v in market_skills.items()
                if (k or "").strip() and (len(k.strip()) > 1 or k.strip().lower() in _MARKET_SINGLE_ALLOW)}

    def _rebuild_fuzzy_patterns(self) -> None:
        """Precompile whole-word patterns for every market name.

        `_fuzzy_hit_name` then scans the precompiled list instead of calling
        `re.search` with a freshly built pattern per iteration.
        """
        self._market_word_pats = [
            (name, _word_pattern(name)) for name in self.market_skills
        ]
        # Lemma-space patterns for inflection-insensitive fuzzy (v44).
        self._market_lemma_pats = []
        try:
            from src.text.ru_morph import available as _avail, lemma_key as _lkey
            if _avail():
                self._market_lemma_pats = [
                    (name, _word_pattern(lk))
                    for name in self.market_skills
                    for lk in [_lkey(name)]
                    # Degenerate keys ('с#' -> 'с') would match prepositions (v44 ablation).
                    if len(lk) >= 2
                ]
        except Exception:
            self._market_lemma_pats = []

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

    def _mapped_match(self, n: str) -> tuple[str | None, str, float]:
        """ACTION->TOOL resolution (v31): RPD verbs resolve to market tools."""
        try:
            from src.analyzers.action_map import phrase_lemmas, resolve_action_tools
            lemmas = phrase_lemmas(n)
            if not lemmas:
                return (None, "no_match", 0.0)
            hit = resolve_action_tools(lemmas, self.market_skills)
            if hit:
                return (hit[0], "mapped", 0.85)
        except Exception as exc:
            logger.debug("mapped_match_failed", error=str(exc))
        return (None, "no_match", 0.0)

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
        from src.feature_flags import lemma_fuzzy_enabled as _lf_on
        from src.feature_flags import market_verbs_enabled as _verbs_on
        if _verbs_on():
            stripped = strip_lead_verbs(n)
            if stripped and stripped != n:
                logger.debug("skill_verbs_stripped", original=n, stripped=stripped)
                n = stripped

        cached = self._match_cache.get(n)
        if cached is not None:
            return Ok(cached)

        result = self._match_uncached(n, skill_name, lemma_fuzzy=_lf_on())
        if result.is_ok():
            self._match_cache[n] = result.unwrap()
        return result

    def _lemma_fuzzy_hit_name(self, n: str) -> str | None:
        """Whole-word overlap in lemma space (declension-insensitive, v44)."""
        if not self._market_lemma_pats:
            return None
        try:
            from src.text.ru_morph import lemma_key as _lkey
        except Exception:
            return None
        ln = _lkey(n)
        if not ln:
            return None
        qpat = _word_pattern(ln)
        for mn, mpat in self._market_lemma_pats:
            lm = _lkey(mn)
            if qpat.search(lm) or mpat.search(ln):
                return mn
        return None

    def _match_uncached(self, n: str, skill_name: str, lemma_fuzzy: bool = False) -> Result[tuple[str | None, str, float], MatchingError]:
        if n in self.market_skills:
            logger.debug("skill_exact_match", skill=n)
            return Ok((n, "exact", 1.0))

        mn = self._fuzzy_hit_name(n)
        if mn:
            if self.market_skills.get(mn, 0) < MARKET_MIN_FREQ:
                logger.debug("skill_fuzzy_fringe_skipped", rpd_skill=n, market_skill=mn)
            else:
                logger.debug("skill_fuzzy_match", rpd_skill=n, market_skill=mn)
                return Ok((mn, "fuzzy", 0.5))


        mapped, _, _ = self._mapped_match(n)
        if mapped:
            logger.debug("skill_mapped_match", rpd_skill=n, market_skill=mapped)
            return Ok((mapped, "mapped", 0.85))

        if lemma_fuzzy:
            lm = self._lemma_fuzzy_hit_name(n)
            if lm:
                if self.market_skills.get(lm, 0) < MARKET_MIN_FREQ:
                    logger.debug("skill_lemma_fringe_skipped", rpd_skill=n, market_skill=lm)
                else:
                    logger.debug("skill_lemma_match", rpd_skill=n, market_skill=lm)
                    return Ok((lm, "lemma", 0.5))

        mn, mt, score = self._semantic_match(n, skill_name)
        if mn:
            logger.debug("skill_semantic_match", rpd_skill=n, market_skill=mn)
        return Ok((mn, mt, score))

    def get_emerging(
        self, rpd_normalized: set[str], top_n: int = 10,
        also_exclude: set[str] | None = None,
        max_freq: int | None = None,
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
            if max_freq is not None and mf >= max_freq:
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
