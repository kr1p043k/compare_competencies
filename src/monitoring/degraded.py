"""Общий флаг деградации сервисов (BACKLOG 2.1, P0).

`channel_diag` уже даёт API-safe статус каналов пайплайна, но общего признака
«сервис деградирован» (нет пула БД, не загружен evaluator или движок
рекомендаций) в ответах не было. Функция возвращает компактный dict для
заголовков/эндпоинтов.

Проверки лёгкие и не бьют в БД на каждый запрос: смотрим наличие пула БД и
загруженных движков. Реальная проверка живости БД/провайдеров остаётся за
`/health` и `/ready` и здесь намеренно не дублируется.
"""

from __future__ import annotations

from typing import Any

import src.db as _db


def degraded_reasons() -> list[str]:
    """Причины деградации (пусто, если сервис в норме)."""
    reasons: list[str] = []
    if _db.pool is None:
        reasons.append("db_pool_uninitialized")
    try:
        from src.api_pkg import deps

        if getattr(deps, "evaluator", None) is None:
            reasons.append("evaluator_not_loaded")
        if getattr(deps, "recommendation_engine", None) is None:
            reasons.append("recommendation_engine_not_loaded")
    except Exception:
        reasons.append("engine_state_unknown")
    return reasons


def degraded_state() -> dict[str, Any]:
    """API-safe состояние деградации: {degraded, reasons, count}."""
    reasons = degraded_reasons()
    return {"degraded": bool(reasons), "reasons": reasons, "count": len(reasons)}
