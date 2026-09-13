"""Direction analysis scope: methodology exclusions + UI overrides (v41).

Methodology floor (SCOPE_EXCLUDED + english rule) lives in the teacher runner.
UI rows in discipline_scope override it in both directions (full control).
"""
from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def _methodology():
    from src.pipeline.teacher_analysis_runner import SCOPE_EXCLUDED, discipline_in_scope
    return SCOPE_EXCLUDED, discipline_in_scope


async def load_scope_overrides(pool, dir_code: str) -> dict[str, bool]:
    """{discipline_name: included}. Missing table/DB error -> {} (never breaks reads)."""
    try:
        rows = await pool.fetch(
            "SELECT discipline_name, included FROM discipline_scope WHERE direction_code = $1",
            dir_code,
        )
        return {r["discipline_name"]: bool(r["included"]) for r in rows}
    except Exception as exc:
        logger.warning("scope_overrides_unavailable", error=str(exc))
        return {}


def effective_in_scope(name: str, overrides: dict[str, bool] | None = None) -> bool:
    """Explicit UI row wins; otherwise the methodology predicate applies."""
    if overrides and name in overrides:
        return overrides[name]
    _, predicate = _methodology()
    return predicate(name)


def scope_source(name: str, overrides: dict[str, bool] | None = None) -> str:
    """"custom" | "methodology" | "default" — for UI badges."""
    if overrides and name in overrides:
        return "custom"
    excluded, predicate = _methodology()
    if not predicate(name):
        return "methodology"
    return "default"


def effective_excluded(names, overrides: dict[str, bool] | None = None) -> list[str]:
    return sorted(n for n in names if not effective_in_scope(n, overrides))
