"""Feature flags: env-driven rollout switches (minimal slice).

Defaults are ON (proven safe in the v32 measurement run 12.09.2026:
    60/60 disciplines, coverage unchanged, +96 targeted recs, 18 unit tests).
Opt out per-run, e.g.: FF_MARKET_SYNONYMS=0 FF_WEAK_COMP_RECS=0 ...
Active flags are recorded into _report_meta.json by the teacher runner (lineage),
so any report can be traced to the exact flags it was built with.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

FF_MARKET_SYNONYMS = "FF_MARKET_SYNONYMS"
FF_WEAK_COMP_RECS = "FF_WEAK_COMP_RECS"
FF_MATCH_VERBS = "FF_MATCH_VERBS"
FF_LEMMA_FUZZY = "FF_LEMMA_FUZZY"


@dataclass(frozen=True)
class Flag:
    """Флаг: имя env-переменной, дефолт и назначение (единый источник правды)."""

    name: str
    default: bool
    description: str


#: Реестр всех флагов. Добавление фичи = одна строка здесь; active_flags() и
#: *_enabled() выводятся из реестра, поэтому расхождения невозможны.
FLAGS: tuple[Flag, ...] = (
    Flag(FF_MARKET_SYNONYMS, True, "Fold version-split market aliases (python3 -> python) at market build"),
    Flag(FF_WEAK_COMP_RECS, True, "Emit targeted add_new recs for weak (0 < coverage < 0.5) competencies"),
    Flag(FF_MATCH_VERBS, True, "Strip leading competency verbs before matching"),
    Flag(FF_LEMMA_FUZZY, True, "Lemma-space fuzzy stage after mapped (v44: mapped keeps precedence)"),
)

_BY_NAME: dict[str, Flag] = {f.name: f for f in FLAGS}

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    return default


def is_enabled(flag: Flag | str) -> bool:
    """Включён ли флаг (по имени из реестра). Неизвестный флаг -> KeyError."""
    name = flag.name if isinstance(flag, Flag) else flag
    spec = _BY_NAME.get(name)
    if spec is None:
        raise KeyError(f"Unknown feature flag: {name!r}")
    return _env_bool(spec.name, spec.default)


def market_synonyms_enabled() -> bool:
    return is_enabled(FF_MARKET_SYNONYMS)


def weak_comp_recs_enabled() -> bool:
    return is_enabled(FF_WEAK_COMP_RECS)


def market_verbs_enabled() -> bool:
    return is_enabled(FF_MATCH_VERBS)


def lemma_fuzzy_enabled() -> bool:
    return is_enabled(FF_LEMMA_FUZZY)


def active_flags() -> dict[str, bool]:
    """Снимок всех флагов (для lineage в отчётах)."""
    return {f.name: is_enabled(f) for f in FLAGS}
