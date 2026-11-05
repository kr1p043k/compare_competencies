"""Снимок сводки анализа направления в историю (time series).

Каждый запуск добавляет компактную запись в `data/result/history/<код>_<год>.jsonl`
(append-only): раньше `_summary.json` просто перезаписывался, и прошлые цифры
терялись. Теперь платформа может показать, как покрытие росло от запуска к запуску.

    python scripts/snapshot_summary.py --direction 09.03.02 --year 2025
    python scripts/snapshot_summary.py --direction 09.03.02 --year 2025 --empty-ok
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "data" / "result" / "teacher"
DEFAULT_STORE = ROOT / "data" / "result" / "history"

EXIT_MISSING = 2

STATUS_APPENDED = "appended"
STATUS_DUPLICATE = "skipped-duplicate"
STATUS_MISSING = "missing"


def summary_path(direction: str, year: int, base: Path = RESULT_DIR) -> Path | None:
    """Путь к сводке года (как в `compare_intakes.summary_path`).

    Текущий год лежит в корне (`<код>/_summary.json`), прежние - в подкаталоге
    `<код>/<год>/_summary.json`. Год сверяем внутри файла, а не только по имени
    каталога, чтобы не спутать наборы.
    """
    subdir = base / direction / str(year) / "_summary.json"
    if subdir.is_file():
        return subdir
    root = base / direction / "_summary.json"
    if root.is_file():
        try:
            doc = json.loads(root.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if doc.get("year") == year:
            return root
    return None


def load_summary(direction: str, year: int, base: Path = RESULT_DIR) -> dict[str, Any] | None:
    """Читает сводку, либо None, если её нет."""
    path = summary_path(direction, year, base)
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _count_no_data(disciplines: Any) -> int:
    """Число дисциплин с `coverage_level == "no_data"`."""
    if not isinstance(disciplines, list):
        return 0
    return sum(
        1
        for d in disciplines
        if isinstance(d, dict) and d.get("coverage_level") == "no_data"
    )


def build_record(doc: dict[str, Any], now: str | None = None) -> dict[str, Any]:
    """Компактная запись снимка (схема совпадает с читателем в роутере)."""
    captured_at = now if now is not None else datetime.now(UTC).isoformat()
    return {
        "captured_at": captured_at,
        "generated_at": doc.get("generated_at"),
        "year": doc.get("year"),
        "code_version": doc.get("code_version"),
        "total_disciplines": doc.get("total_disciplines"),
        "average_coverage": doc.get("average_coverage"),
        "average_quality_coverage": doc.get("average_quality_coverage"),
        "average_market_top_coverage": doc.get("average_market_top_coverage"),
        "average_market_weight_coverage": doc.get("average_market_weight_coverage"),
        "total_gaps": doc.get("total_gaps_across_all"),
        "no_data_count": _count_no_data(doc.get("disciplines")),
    }


def last_record(path: Path) -> dict[str, Any] | None:
    """Последняя валидная запись JSONL (для дедупликации), либо None."""
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def snapshot(
    direction: str,
    year: int,
    result_dir: Path = RESULT_DIR,
    store_dir: Path = DEFAULT_STORE,
    now: str | None = None,
) -> str:
    """Снимает сводку в JSONL. Возвращает appended | skipped-duplicate | missing."""
    doc = load_summary(direction, year, result_dir)
    if doc is None:
        return STATUS_MISSING

    record = build_record(doc, now=now)

    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / f"{direction}_{year}.jsonl"

    last = last_record(path)
    if last is not None:
        generated = record.get("generated_at")
        prev_generated = last.get("generated_at")
        same_generated = generated is not None and prev_generated is not None and generated == prev_generated
        if same_generated and record.get("code_version") == last.get("code_version"):
            return STATUS_DUPLICATE

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return STATUS_APPENDED


def read_history(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Читает JSONL в список точек (от старых к новым) и число пропущенных строк.

    Битые строки пропускаются и считаются, краша не бывает - отсутствие файла
    даёт пустой список (отсутствие истории это не ошибка).
    """
    points: list[dict[str, Any]] = []
    skipped = 0
    if not path.is_file():
        return points, skipped
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return points, skipped
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(obj, dict):
            skipped += 1
            continue
        points.append(obj)
    points.sort(key=lambda p: str(p.get("captured_at") or ""))
    return points, skipped


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def compute_delta(first: dict[str, Any] | None, last: dict[str, Any] | None) -> dict[str, Any] | None:
    """Дельта последней точки относительно первой (для «роста с X до Y»)."""
    if first is None or last is None:
        return None
    return {
        "average_coverage": round(
            _as_float(last.get("average_coverage")) - _as_float(first.get("average_coverage")),
            6,
        ),
        "total_gaps": _as_int(last.get("total_gaps")) - _as_int(first.get("total_gaps")),
        "total_disciplines": _as_int(last.get("total_disciplines"))
        - _as_int(first.get("total_disciplines")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Снимок сводки анализа в историю")
    parser.add_argument("--direction", required=True, help="код направления, напр. 09.03.02")
    parser.add_argument("--year", required=True, type=int, help="год набора, напр. 2025")
    parser.add_argument(
        "--store",
        default=str(DEFAULT_STORE),
        help="каталог истории (по умолчанию data/result/history)",
    )
    parser.add_argument("--empty-ok", action="store_true", help="не считать отсутствие сводки ошибкой")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="реконструкция истории из старых прогонов (не требуется)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.backfill:
        print("backfill: not required, skipped")

    status = snapshot(args.direction, args.year, RESULT_DIR, Path(args.store))
    store_path = Path(args.store) / f"{args.direction}_{args.year}.jsonl"

    if status == STATUS_MISSING:
        print(f"missing: нет сводки для {args.direction} / {args.year}")
        return 0 if args.empty_ok else EXIT_MISSING

    print(f"{status}: {store_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
