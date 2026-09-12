"""Feature flags: env-driven rollout switches (minimal slice).

Defaults are ON (proven safe in the v32 measurement run 12.09.2026:
    60/60 disciplines, coverage unchanged, +96 targeted recs, 18 unit tests).
Opt out per-run, e.g.: FF_MARKET_SYNONYMS=0 FF_WEAK_COMP_RECS=0 ...
Active flags are recorded into _report_meta.json by the teacher runner (lineage),
so any report can be traced to the exact flags it was built with.
"""
from __future__ import annotations

import os

FF_MARKET_SYNONYMS = "FF_MARKET_SYNONYMS"
FF_WEAK_COMP_RECS = "FF_WEAK_COMP_RECS"


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def market_synonyms_enabled() -> bool:
    """Fold version-split market aliases (python3 -> python) at market build."""
    return _env_bool(FF_MARKET_SYNONYMS, True)


def weak_comp_recs_enabled() -> bool:
    """Emit targeted add_new recs for weak (0 < coverage < 0.5) competencies."""
    return _env_bool(FF_WEAK_COMP_RECS, True)


def active_flags() -> dict[str, bool]:
    return {
        FF_MARKET_SYNONYMS: market_synonyms_enabled(),
        FF_WEAK_COMP_RECS: weak_comp_recs_enabled(),
    }
