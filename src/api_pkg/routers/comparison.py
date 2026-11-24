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
from pydantic import BaseModel, ConfigDict

from src.api_pkg.routers.auth import require_any_role

router = APIRouter(tags=["teacher"])


class IntakeDisciplineCard(BaseModel):
    """Дисциплина в отчёте сравнения приёмов.

    Отчёт собирает scripts/compare_intakes.py; поля необязательные, чтобы
    частичные отчёты не падали на валидации.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    coverage_ratio: float | None = None
    coverage_level: str | None = None
    gaps: Any = None


class IntakeKeptDiscipline(BaseModel):
    """Дисциплина, сохранившаяся в обоих приёмах."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    old_name: str | None = None
    coverage_ratio_old: float | None = None
    coverage_ratio_new: float | None = None
    coverage_delta: float | None = None
    gaps_old: Any = None
    gaps_new: Any = None
    gaps_delta: Any = None
    significant_change: bool | None = None


class IntakeComparisonSummary(BaseModel):
    """Агрегаты сравнения приёмов."""

    model_config = ConfigDict(extra="allow")

    added: int | None = None
    removed: int | None = None
    kept: int | None = None
    avg_coverage_old: float | None = None
    avg_coverage_new: float | None = None
    avg_coverage_delta: float | None = None
    improved: int | None = None
    worsened: int | None = None
    unchanged: int | None = None


class IntakeComparisonResponse(BaseModel):
    """Сводка сравнения двух приёмов по направлению."""

    model_config = ConfigDict(extra="allow")

    direction: str | None = None
    old_year: int | None = None
    new_year: int | None = None
    direction_name: str | None = None
    added: list[IntakeDisciplineCard] | None = None
    removed: list[IntakeDisciplineCard] | None = None
    kept: list[IntakeKeptDiscipline] | None = None
    summary: IntakeComparisonSummary | None = None

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
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts_dir)
    _compare_module = module
    return _compare_module


@router.get(
    "/teacher/intake-comparison",
    response_model=IntakeComparisonResponse,
    response_model_exclude_unset=True,
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
