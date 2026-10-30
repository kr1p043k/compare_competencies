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

from src import config

OVERRIDES_PATH = config.REFERENCE_DIR / "discipline_market_scope.json"

# Neighbour expansion for the derived scope.
NEIGHBOURS_PER_ANCHOR = 12
MIN_NEIGHBOUR_SIM = 0.55


def load_overrides(path: Path | None = None) -> dict[str, dict]:
    """Read the manual mapping: {discipline: {"skills": [...], "keywords": [...]}}."""
    p = path or OVERRIDES_PATH
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
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
    if derived:
        return derived, "derived"
    return None


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