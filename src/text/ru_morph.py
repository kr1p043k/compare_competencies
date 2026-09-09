"""Russian morphology helpers (v10): pymorphy3 lemmatization, lazy, cached."""
from __future__ import annotations

import re

_WORD = re.compile(r"\w+", flags=re.UNICODE)

_analyzer = None
_analyzer_failed = False
_token_cache: dict[str, str] = {}


def get_analyzer():
    """Shared MorphAnalyzer or None (lazy, never raises)."""
    global _analyzer, _analyzer_failed
    if _analyzer is not None:
        return _analyzer
    if _analyzer_failed:
        return None
    try:
        import pymorphy3

        _analyzer = pymorphy3.MorphAnalyzer()
        return _analyzer
    except Exception:
        _analyzer_failed = True
        return None


def lemma(token: str) -> str:
    """Normal form of a single token (lowercased)."""
    t = (token or "").lower()
    if not t:
        return ""
    hit = _token_cache.get(t)
    if hit is not None:
        return hit
    out = t
    analyzer = get_analyzer()
    if analyzer is not None:
        try:
            out = analyzer.parse(t)[0].normal_form
        except Exception:
            out = t
    _token_cache[t] = out
    return out


def lemmas(text: str) -> list[str]:
    """Lemma sequence for free text (dedup/fold keys)."""
    return [lemma(w) for w in _WORD.findall((text or "").lower())]


def lemma_key(text: str) -> str:
    """Canonical key: space-joined lemmas (declension-insensitive)."""
    return " ".join(lemmas(text))


def available() -> bool:
    return get_analyzer() is not None
