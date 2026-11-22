"""Сравнение покрытия дисциплин между двумя годами набора направления.

Отдаёт тот же отчёт, что и `scripts/compare_intakes.py` (логика не дублируется -
скрипт грузится через importlib, вызывается `build_intake_report` напрямую).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api_pkg.routers.auth import require_any_role

router = APIRouter(tags=["teacher"])


# ---------- схемы ответа (6.2.1) ----------
# Контракт задаётся `scripts/compare_intakes.py::build_intake_report`; тест
# tests/api/test_comparison_coverage.py сверяет модели с реальным выводом скрипта.


class IntakeDisciplineCard(BaseModel):
    """Дисциплина в списках added/removed."""

    model_config = ConfigDict(extra="allow")

    name: str
    coverage_ratio: float | None = None
    coverage_level: str | None = None
    gaps: int | None = None


class IntakeKeptDiscipline(BaseModel):
    """Дисциплина, присутствующая в обоих наборах."""

    model_config = ConfigDict(extra="allow")

    name: str
    old_name: str | None = None
    coverage_ratio_old: float
    coverage_ratio_new: float
    coverage_delta: float
    gaps_old: int
    gaps_new: int
    gaps_delta: int
    significant_change: bool


class IntakeSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    added: int
    removed: int
    kept: int
    avg_coverage_old: float
    avg_coverage_new: float
    avg_coverage_delta: float
    improved: int
    worsened: int
    unchanged: int


class IntakeComparisonReport(BaseModel):
    """Ответ GET /teacher/intake-comparison."""

    model_config = ConfigDict(extra="allow")

    direction: str
    old_year: int
    new_year: int
    direction_name: str | None = None
    added: list[IntakeDisciplineCard] = Field(default_factory=list)
    removed: list[IntakeDisciplineCard] = Field(default_factory=list)
    kept: list[IntakeKeptDiscipline] = Field(default_factory=list)
    summary: IntakeSummary

_ROLES = ["admin", "teacher", "rop"]

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "compare_intakes.py"

_compare_module: Any = None


def _load_compare() -> Any:
    global _compare_module
    if _compare_module is not None:
        return _compare_module

    scripts_dir = str(_SCRIPT.parent)
    with contextlib.suppress(ValueError):
        sys.path.remove(scripts_dir)
    sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location("compare_intakes", _SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts_dir)
    _compare_module = module
    return _compare_module


@router.get(
    "/teacher/intake-comparison",
    response_model=IntakeComparisonReport,
    dependencies=[Depends(require_any_role(*_ROLES))],
)
async def intake_comparison(
    dir_code: str = "09.03.02",
    old: int = 2024,
    new: int = 2025,
) -> Any:
    """Что изменилось в покрытии дисциплин между `old` и `new` набором."""
    compare = _load_compare()
    try:
        report = await asyncio.to_thread(compare.build_intake_report, dir_code, old, new)
    except compare.IntakeNotFoundError:
        # Текст исключения не отдаём наружу: он собирается из путей и может
        # содержать внутренние детали. Пользователю достаточно, какого набора
        # не хватает.
        raise HTTPException(
            status_code=404,
            detail=f"Нет данных анализа для набора {old} или {new} по направлению {dir_code}",
        ) from None
    return report
