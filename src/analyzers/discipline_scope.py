"""Scope of market skills a discipline is responsible for.

The demand-coverage metric needs a denominator. Scoring the discipline *name*
with an embedding is too weak: for 11 of 36 disciplines the profile yields
fewer than five relevant market skills, so the metric degenerates.

This module builds the scope from what the discipline actually teaches:

  1. explicit mapping  - data/reference/discipline_market_scope.json, edited by
                         a teacher ("уточнение инструментов"); wins when present
  2. derived scope     - semantic neighbours of the discipline's own skills
  3. relevance         - fallback to the discipline-embedding filter

The derived scope is not circular: a discipline teaching 20 skills gets a
scope of a few hundred related market skills, so the ratio still measures how
much of its own domain's demand it covers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

from src import config

logger = structlog.get_logger(__name__)

OVERRIDES_PATH = config.REFERENCE_DIR / "discipline_market_scope.json"

# Neighbour expansion for the derived scope.
NEIGHBOURS_PER_ANCHOR = 12
MIN_NEIGHBOUR_SIM = 0.55
# Ниже этого размера scope считается тонким и берётся другой источник.
MIN_RELIABLE_SCOPE = 15


def load_overrides(path: Path | None = None) -> dict[str, dict]:
    """Read the manual mapping: {discipline: {"skills": [...], "keywords": [...]}}."""
    p = path or OVERRIDES_PATH
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Файл необязателен: без него действуют только правила методики.
        return {}
    except json.JSONDecodeError as exc:
        # Битый файл - это не «настроек нет»: все ручные исключения молча
        # пропадут и в анализ попадут дисциплины, которые преподаватель убрал.
        logger.warning("scope_overrides_corrupt", path=str(p), error=str(exc))
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def save_overrides(overrides: dict[str, dict], path: Path | None = None) -> None:
    p = Path(path or OVERRIDES_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def neighbours(
    matcher,
    skill: str,
    limit: int = NEIGHBOURS_PER_ANCHOR,
    min_sim: float = MIN_NEIGHBOUR_SIM,
) -> list[str]:
    """Nearest market skills by cosine, using the matcher's prebuilt index."""
    embeddings = getattr(matcher, "_market_embeddings", None)
    names = getattr(matcher, "_market_names", None)
    provider = getattr(matcher, "_embedding_provider", None)
    if embeddings is None or not names or provider is None:
        return []
    try:
        vector = provider.encode([skill], show_progress_bar=False)[0]
    except Exception:
        return []
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return []
    sims = embeddings @ (vector / norm)
    order = np.argsort(-sims)[: max(limit * 4, limit)]
    out: list[str] = []
    for idx in order:
        score = float(sims[idx])
        if score < min_sim:
            break
        name = names[int(idx)]
        if name != skill:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def resolve_scope(
    matcher,
    discipline_name: str,
    anchors: list[str],
    overrides: dict[str, dict] | None = None,
    rpd_text: str | None = None,
) -> tuple[set[str], str] | None:
    """Return (market skills this discipline is responsible for, source)."""
    overrides = overrides or {}
    entry = overrides.get(discipline_name) or {}

    explicit = {str(s).strip().lower() for s in entry.get("skills") or [] if str(s).strip()}
    keywords = [str(k).strip() for k in entry.get("keywords") or [] if str(k).strip()]

    if explicit or keywords:
        scope = set(explicit)
        for keyword in keywords:
            scope.update(neighbours(matcher, keyword))
            scope.add(keyword.lower())
        if scope:
            return scope, "explicit"

    derived: set[str] = set()
    for anchor in anchors:
        derived.update(neighbours(matcher, anchor))
    if len(derived) >= MIN_RELIABLE_SCOPE:
        return derived, "derived"

    # Few skills on the discipline: lean on the RPD text, which describes it
    # far better than its name.
    if rpd_text:
        from_text = scope_from_text(matcher, rpd_text)
        if len(from_text) >= MIN_RELIABLE_SCOPE:
            return from_text, "rpd_text"

    return (derived, "derived") if derived else None


def describe(matcher, discipline_name: str, anchors: list[str],
             overrides: dict[str, dict] | None = None) -> dict:
    """Diagnostics for the API: what scope a discipline resolves to."""
    resolved = resolve_scope(matcher, discipline_name, anchors, overrides)
    entry = (overrides or {}).get(discipline_name) or {}
    return {
        "discipline": discipline_name,
        "source": resolved[1] if resolved else "relevance",
        "size": len(resolved[0]) if resolved else 0,
        "configured": bool(entry),
        "skills": sorted(entry.get("skills") or []),
        "keywords": list(entry.get("keywords") or []),
    }

# --- RPD full text as a discipline profile -------------------------------
# data/reference/sfu_annotations_all.json holds the parsed RPD files (source:
# the university disk). The full text describes the discipline far better than
# its name, so it is used as an anchor when the skill-derived scope is thin.
ANNOTATIONS_PATH = config.REFERENCE_DIR / "sfu_annotations_all.json"
TEXT_SCOPE_LIMIT = 60
TEXT_SCOPE_MIN_SIM = 0.45


def _norm_name(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def load_rpd_texts(path: Path | None = None) -> dict[str, str]:
    """Map normalized discipline name -> RPD text, across every direction."""
    p = Path(path or ANNOTATIONS_PATH)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        # Без корпуса РПД профиль дисциплины остаётся пустым, и scope
        # считается только по названию - молча менее точно.
        logger.warning("rpd_texts_corrupt", path=str(p), error=str(exc))
        return {}
    out: dict[str, str] = {}
    if not isinstance(data, dict):
        return out
    for entries in data.values():
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            name, text = item.get("name"), item.get("text")
            if not name or not text:
                continue
            key = _norm_name(name)
            # Keep the longest text per name: some RPDs repeat with truncation.
            if len(text) > len(out.get(key, "")):
                out[key] = text
    return out


def find_rpd_text(name: str, texts: dict[str, str]) -> str | None:
    """Exact normalized match first, then a close-enough name match."""
    key = _norm_name(name)
    if not key:
        return None
    if key in texts:
        return texts[key]
    import difflib

    close = difflib.get_close_matches(key, list(texts), n=1, cutoff=0.86)
    return texts[close[0]] if close else None


def scope_from_text(matcher, text: str,
                    limit: int = TEXT_SCOPE_LIMIT,
                    min_sim: float = TEXT_SCOPE_MIN_SIM) -> set[str]:
    """Nearest market skills to the embedding of a discipline's RPD text."""
    embeddings = getattr(matcher, "_market_embeddings", None)
    names = getattr(matcher, "_market_names", None)
    provider = getattr(matcher, "_embedding_provider", None)
    if embeddings is None or not names or provider is None or not text:
        return set()
    try:
        vector = provider.encode([text[:4000]], show_progress_bar=False)[0]
    except Exception:
        return set()
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return set()
    sims = embeddings @ (vector / norm)
    order = np.argsort(-sims)[: limit * 3]
    out: set[str] = set()
    for idx in order:
        if float(sims[int(idx)]) < min_sim:
            break
        out.add(names[int(idx)])
        if len(out) >= limit:
            break
    return out
