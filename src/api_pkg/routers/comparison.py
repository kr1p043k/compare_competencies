"""Сравнение покрытия дисциплин между двумя годами набора направления.

Отдаёт тот же отчёт, что и `scripts/compare_intakes.py` (логика не дублируется -
скрипт грузится через importlib, вызывается `build_intake_report` напрямую).
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api_pkg.routers.auth import require_any_role

router = APIRouter(tags=["teacher"])

_ROLES = ["admin", "teacher", "rop"]

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "compare_intakes.py"

_compare_module: Any = None


def _load_compare() -> Any:
    """Ленивая загрузка скрипта сравнения (как `test_krm_merge_scripts._load`)."""
    global _compare_module
    if _compare_module is None:
        spec = importlib.util.spec_from_file_location("compare_intakes", _SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _compare_module = module
    return _compare_module


@router.get("/teacher/intake-comparison", dependencies=[Depends(require_any_role(*_ROLES))])
async def intake_comparison(
    dir_code: str = "09.03.02",
    old: int = 2024,
    new: int = 2025,
) -> Any:
    """Что изменилось в покрытии дисциплин между `old` и `new` набором."""
    compare = _load_compare()
    try:
        report = await asyncio.to_thread(compare.build_intake_report, dir_code, old, new)
    except compare.IntakeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    return report
