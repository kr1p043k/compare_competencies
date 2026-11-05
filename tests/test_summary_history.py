"""Тесты снимков истории сводок анализа (`scripts/snapshot_summary.py`).

Не зависят от реальных данных проекта: сводки строятся во временном каталоге,
поведение проверяется через чистые функции модуля (snapshot / read_history /
compute_delta / main).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "snapshot_summary", ROOT / "scripts" / "snapshot_summary.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _make_summary(
    generated_at: str = "2026-11-04T15:00:00",
    avg: float = 0.6853,
    code_version: int = 49,
    year: int = 2025,
) -> dict:
    return {
        "direction": "09.03.02",
        "year": year,
        "code_version": code_version,
        "total_disciplines": 36,
        "average_coverage": avg,
        "average_quality_coverage": 0.4669,
        "average_market_top_coverage": 0.2265,
        "average_market_weight_coverage": 0.2926,
        "total_gaps_across_all": 498,
        "generated_at": generated_at,
        "disciplines": [
            {"name": "A", "coverage_level": "high"},
            {"name": "B", "coverage_level": "no_data"},
            {"name": "C", "coverage_level": "no_data"},
        ],
    }


def _write_summary(base: pathlib.Path, direction: str, year: int, doc: dict) -> pathlib.Path:
    d = base / direction
    d.mkdir(parents=True, exist_ok=True)
    # Год кладём в корень только если он совпадает с `year` внутри файла,
    # иначе snapshot для другого года спишет его в отдельную точку.
    sub = d / str(year) / "_summary.json"
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text(json.dumps(doc), encoding="utf-8")
    return sub


def test_snapshot_idempotent(mod, tmp_path):
    result = tmp_path / "result"
    store = tmp_path / "history"
    _write_summary(result, "09.03.02", 2025, _make_summary())

    assert mod.snapshot("09.03.02", 2025, result_dir=result, store_dir=store) == "appended"
    assert mod.snapshot("09.03.02", 2025, result_dir=result, store_dir=store) == "skipped-duplicate"

    lines = (store / "09.03.02_2025.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["no_data_count"] == 2
    assert rec["total_gaps"] == 498
    assert rec["code_version"] == 49


def test_two_points_oldest_first_in_api_helper(mod, tmp_path):
    result = tmp_path / "result"
    store = tmp_path / "history"

    _write_summary(result, "09.03.02", 2025, _make_summary(generated_at="2026-11-01T00:00:00", avg=0.1))
    assert (
        mod.snapshot(
            "09.03.02", 2025, result_dir=result, store_dir=store, now="2026-11-01T10:00:00+00:00"
        )
        == "appended"
    )

    _write_summary(result, "09.03.02", 2025, _make_summary(generated_at="2026-11-02T00:00:00", avg=0.5))
    assert (
        mod.snapshot(
            "09.03.02", 2025, result_dir=result, store_dir=store, now="2026-11-02T10:00:00+00:00"
        )
        == "appended"
    )

    points, skipped = mod.read_history(store / "09.03.02_2025.jsonl")
    assert skipped == 0
    assert len(points) == 2

    # oldest first
    assert points[0]["generated_at"] == "2026-11-01T00:00:00"
    assert points[0]["average_coverage"] == pytest.approx(0.1)
    assert points[1]["generated_at"] == "2026-11-02T00:00:00"

    delta = mod.compute_delta(points[0], points[-1])
    assert delta is not None
    assert delta["average_coverage"] == pytest.approx(0.4)


def test_missing_summary_exit_code_and_no_file(mod, tmp_path):
    store = tmp_path / "history"

    # Без --empty-ok отсутствие сводки это ошибка (код 2), файл не создаётся.
    rc = mod.main(["--direction", "99.99.99", "--year", "2099", "--store", str(store)])
    assert rc == mod.EXIT_MISSING == 2
    assert not (store / "99.99.99_2099.jsonl").exists()

    # --empty-ok превращает отсутствие в успех (0), по-прежнему ничего не пишем.
    rc_ok = mod.main(
        ["--direction", "99.99.99", "--year", "2099", "--store", str(store), "--empty-ok"]
    )
    assert rc_ok == 0
    assert not (store / "99.99.99_2099.jsonl").exists()


def test_malformed_line_skipped_and_counted(mod, tmp_path):
    path = tmp_path / "09.03.02_2025.jsonl"
    good = {
        "captured_at": "2026-11-01T10:00:00+00:00",
        "generated_at": "g",
        "year": 2025,
        "code_version": 49,
        "total_disciplines": 36,
        "average_coverage": 0.1,
        "average_quality_coverage": 0.1,
        "average_market_top_coverage": 0.1,
        "average_market_weight_coverage": 0.1,
        "total_gaps": 10,
        "no_data_count": 0,
    }
    path.write_text("{not json}\n" + json.dumps(good) + "\n", encoding="utf-8")

    points, skipped = mod.read_history(path)
    assert skipped == 1
    assert len(points) == 1
    assert points[0]["average_coverage"] == pytest.approx(0.1)


def test_read_history_missing_file_is_empty(mod, tmp_path):
    points, skipped = mod.read_history(tmp_path / "absent.jsonl")
    assert points == []
    assert skipped == 0
    assert mod.compute_delta(None, None) is None
