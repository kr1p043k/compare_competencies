"""Аудит качества прогнозов Prophet: hold-out MAPE по снимкам рынка.

Показывает, насколько точны прогнозы Prophet/trend для навыков с достаточной
историей, и средний горизонт прогноза.

Usage:
    python -m src.cli forecast-audit --top 20
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import structlog

from src import config

logger = structlog.get_logger(__name__)

HISTORY_DIR = config.HISTORY_DIR


def _load_snapshots() -> list[tuple[date, dict[str, float]]]:
    snaps = []
    for f in sorted(HISTORY_DIR.glob("freq_market_*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            meta = raw.pop("_meta", {}) if isinstance(raw, dict) else {}
            data = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
            sd = meta.get("snapshot_date", "")
            try:
                dt = datetime.strptime(sd, "%Y-%m-%d").date()
            except ValueError:
                try:
                    dt = datetime.strptime(sd, "%Y-%m").date()
                except ValueError:
                    continue
            snaps.append((dt, data))
        except Exception:
            continue
    snaps.sort(key=lambda x: x[0])
    return snaps


def _interp(snaps: list[tuple[date, dict[str, float]]]) -> list[tuple[date, dict[str, float]]]:
    """Та же интерполяция, что в prophet_forecast.load_time_series."""
    if len(snaps) < 2:
        return snaps
    result: list[tuple[date, dict[str, float]]] = []
    for i in range(len(snaps)):
        cur_date, cur_data = snaps[i]
        if i == 0:
            result.append((cur_date, cur_data))
            continue
        prev_date, prev_data = snaps[i - 1]
        gap = (cur_date.year - prev_date.year) * 12 + (cur_date.month - prev_date.month)
        if gap <= 1:
            result.append((cur_date, cur_data))
            continue
        all_skills = set(prev_data) | set(cur_data)
        for step in range(1, gap):
            t = step / gap
            mid_idx = prev_date.month + step - 1
            mid = date(prev_date.year + mid_idx // 12, mid_idx % 12 + 1, 1)
            interp = {}
            for skill in all_skills:
                val = prev_data.get(skill, 0.0) + (cur_data.get(skill, 0.0) - prev_data.get(skill, 0.0)) * t
                if val >= 1.0:
                    interp[skill] = round(val, 1)
            result.append((mid, interp))
        result.append((cur_date, cur_data))
    return result


def _holdout_mape(points: list[tuple[date, float]]) -> float:
    if len(points) < 3:
        return 0.0
    train = points[:-1]
    x = np.array([(d - train[0][0]).days for d, _ in train], dtype=float)
    y = np.array([f for _, f in train], dtype=float)
    if len(x) < 2 or np.all(x == x[0]):
        return 0.0
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return 0.0
    actual = points[-1][1]
    days_ahead = (points[-1][0] - train[0][0]).days
    pred = intercept + slope * days_ahead
    if abs(actual) < 1.0:
        return 0.0
    return float(abs(pred - actual) / abs(actual))


def main(args: argparse.Namespace) -> None:
    snaps = _load_snapshots()
    if not snaps:
        print("Снимки freq_market_*.json не найдены в", HISTORY_DIR)
        return
    print(f"Снимков: {len(snaps)} | период: {snaps[0][0]} .. {snaps[-1][0]}")
    snaps = _interp(snaps)
    print(f"После интерполяции пропусков: {len(snaps)} точек")

    # Пропущенные месяцы
    gaps = []
    for i in range(1, len(snaps)):
        g = (snaps[i][0].year - snaps[i - 1][0].year) * 12 + (snaps[i][0].month - snaps[i - 1][0].month)
        if g > 1:
            gaps.append(f"{snaps[i-1][0]} -> {snaps[i][0]} (+{g-1})")
    if gaps:
        print("Пропуски месяцев:", "; ".join(gaps))
    else:
        print("Пропусков месяцев нет.")

    # Собираем историю навыков
    history: dict[str, list[tuple[date, float]]] = {}
    for dt, data in snaps:
        for skill, freq in data.items():
            history.setdefault(skill, []).append((dt, freq))
    for skill in history:
        history[skill].sort(key=lambda x: x[0])

    rows = []
    for skill, points in history.items():
        if len(points) < 4:
            continue
        mape = _holdout_mape(points)
        rows.append((skill, len(points), mape, points[-1][1]))

    rows.sort(key=lambda x: x[2], reverse=True)
    n = getattr(args, "top", 20)
    print(f"\nНавыков с историей >=4 точек: {len(rows)}")
    print(f"\nТоп-{n} по hold-out MAPE (худшие прогнозы):")
    print(f"  {'навык':35s} {'точек':>5} {'MAPE':>7} {'текущая':>8}")
    for skill, pts, mape, freq in rows[:n]:
        print(f"  {skill:35s} {pts:5d} {mape:7.2f} {freq:8.1f}")

    good = [r for r in rows if r[2] <= 0.3]
    med = [r for r in rows if 0.3 < r[2] <= 0.6]
    bad = [r for r in rows if r[2] > 0.6]
    print(f"\nКачество (hold-out MAPE): хорошие <=0.3: {len(good)} | средние: {len(med)} | плохие >0.6: {len(bad)}")

    # Средний горизонт
    n_pts_avg = sum(r[1] for r in rows) / max(len(rows), 1)
    print(f"Среднее число точек истории: {n_pts_avg:.1f} (макс. горизонт ~{int(min(n_pts_avg, 12))} мес)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Аудит качества прогнозов Prophet")
    parser.add_argument("--top", type=int, default=20)
    main(parser.parse_args())
