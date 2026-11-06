"""История снимков сводок анализа (time series покрытия).

Каждая сводка `_summary.json` сейчас перезаписывается, поэтому прошлые цифры
исчезают. `scripts/snapshot_summary.py` кладёт компактные записи в
`data/result/history/<код>_<год>.jsonl`; этот роутер отдаёт ряд точек (от старых
к новым) и дельту между первой и последней точкой («покрытие выросло с X до Y»).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from src.api_pkg.routers.auth import require_any_role

router = APIRouter(tags=["teacher"])

_ROLES = ["admin", "teacher", "rop"]

_ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = _ROOT / "data" / "result" / "history"

_DEFAULT_LIMIT = 100

_snapshot_module: Any = None


def _load_snapshot() -> Any:
    """Лениво грузит `scripts/snapshot_summary.py` (как `test_krm_merge_scripts._load`)."""
    global _snapshot_module
    if _snapshot_module is None:
        script = _ROOT / "scripts" / "snapshot_summary.py"
        with contextlib.suppress(ValueError):
            sys.path.remove(str(script.parent))
        sys.path.insert(0, str(script.parent))
        try:
            spec = importlib.util.spec_from_file_location("snapshot_summary", script)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(script.parent))
        _snapshot_module = module
    return _snapshot_module


@router.get("/teacher/history", dependencies=[Depends(require_any_role(*_ROLES))])
async def teacher_history(
    dir_code: str = "09.03.02",
    year: int = 2025,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Публика истории своего направления (oldest first) и дельта спроса/предложения."""
    # dir_code уходит в имя файла: без проверки формата возможен выход из каталога.
    from src.api_pkg.routers.zun import _validate_dir_code

    _validate_dir_code(dir_code)
    snapshot = _load_snapshot()
    path = HISTORY_DIR / f"{dir_code}_{year}.jsonl"

    points, skipped = await asyncio.to_thread(snapshot.read_history, path)

    first = points[0] if points else None
    last = points[-1] if points else None
    delta = snapshot.compute_delta(first, last)

    limited = points[:limit] if limit and limit > 0 else points

    return {
        "points": limited,
        "first": first,
        "last": last,
        "delta": delta,
        "skipped_lines": skipped,
    }
