"""Trend-based fallback forecast engine (replaces broken GA).

Used when Prophet is unavailable or fails. Simple linear regression
on historical skill frequency data from trend_snapshots.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import structlog

from src import Err, Ok, Result, config
from src.errors import DomainError
from src.predictors.base import BasePredictor

logger = structlog.get_logger(__name__)


def drop_broken_snapshots(
    snapshots: list[tuple[date, dict[str, float]]],
) -> list[tuple[date, dict[str, float]]]:
    """Карантин битых коллекций: снимок, в котором навыков меньше половины
    медианы, — это оборванный сбор, а не обвал спроса. Такие точки рисуют
    ложную V (крах + восстановление) и отравляют все тренды.

    Последний снимок удерживаем всегда: он задаёт current_frequency.
    """
    if len(snapshots) < 3:
        return snapshots
    counts = sorted(len(d) for _, d in snapshots)
    floor = max(1, counts[len(counts) // 2] // 2)
    kept = [(dt, d) for dt, d in snapshots if len(d) >= floor]
    if not kept:
        logger.warning("snapshot_quarantine_all_broken")
        return snapshots
    dropped = len(snapshots) - len(kept)
    if dropped:
        logger.warning(
            "snapshot_quarantine_dropped",
            dropped=dropped,
            floor=floor,
            kept=[str(dt) for dt, _ in kept],
        )
    if kept[-1][0] != snapshots[-1][0]:
        logger.error("snapshot_quarantine_latest_dropped_kept_anyway")
        kept.append(snapshots[-1])
        kept.sort(key=lambda x: x[0])
    return kept


# Наблюдаемое падение: сколько точек истории нужно на окно (месяцев).
# Точек меньше — честное "нет данных", а не экстраполяция.
OBSERVED_REQUIRED = {1: 2, 3: 3, 6: 6, 12: 12}
OBSERVED_MONTHS = (1, 3, 6, 12)


def compute_observed_drops(
    observed: dict[str, list[tuple[date, float]]],
    months: int,
    min_freq: float = 0,
) -> list[dict]:
    """Измеренное падение спроса за окно — факт по снимкам, не прогноз.

    Берутся последние REQUIRED[months] точек; изменение =
    (last - first) / first. Точек меньше — навык пропускается
    (UI пишет "нет данных", см. max_points в мете).
    """
    need = OBSERVED_REQUIRED.get(months)
    if need is None:
        raise ValueError(f"months must be one of {OBSERVED_MONTHS}, got {months}")
    out = []
    for skill, pts in (observed or {}).items():
        pts = sorted(pts, key=lambda p: p[0])
        if len(pts) < need:
            continue
        window = pts[-need:]
        first = float(window[0][1])
        last = float(window[-1][1])
        if first <= 0:
            continue
        if min_freq and last < min_freq:
            continue
        out.append({
            "skill": skill,
            "observed_change_pct": round((last - first) / first * 100, 2),
            "first_frequency": round(first, 1),
            "last_frequency": round(last, 1),
            "points": len(pts),
        })
    out.sort(key=lambda r: r["observed_change_pct"])
    return out


@dataclass
class ForecastResult:
    skill: str
    current_frequency: float
    predicted_growth: float
    confidence: float
    next_year_frequency: float
    engine_used: str = "trend"
    data_points: int = 0
    mape: float | None = None
    forecast_months: int = 0


class SkillForecastEngine(BasePredictor):
    """Lightweight trend-based forecast using historical snapshots.

    Fits a simple linear regression per skill: freq ~ time.
    Predicts future frequency based on trend direction + strength.
    Falls back to flat (no growth) when < 2 data points.
    """

    MIN_FREQ = 10
    MAX_GROWTH_CAP = 1.5

    def __init__(self):
        self._models: dict[str, dict] = {}
        self._is_fitted = False
        self._observed: dict[str, list[tuple[date, float]]] = {}

    @property
    def name(self) -> str:
        return "TrendForecast"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        skill_frequencies: dict[str, float] | None = None,
        **kwargs,
    ) -> Result[SkillForecastEngine, Exception]:
        """Fit linear trends from historical snapshot data.

        Reads freq_market_*.json files from data/history/ to build
        time series per skill. If unavailable, falls back to flat forecast.
        """
        history_dir = config.HISTORY_DIR
        snapshots: list[tuple[date, dict]] = []

        for f in sorted(history_dir.glob("freq_market_*.json")):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                meta = raw.pop("_meta", {})
                data = raw
                sd = meta.get("snapshot_date", "")
                try:
                    dt = datetime.strptime(sd, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        dt = datetime.strptime(sd, "%Y-%m").date()
                    except ValueError:
                        continue
                snapshots.append((dt, data))
            except Exception:
                continue

        if len(snapshots) < 2:
            logger.warning("trend_fallback_insufficient_data", snapshots=len(snapshots))
            if skill_frequencies:
                for skill, freq in skill_frequencies.items():
                    self._models[skill] = {"slope": 0.0, "intercept": freq, "n": 1, "rmse": None, "mape": None, "last_freq": freq}
            self._is_fitted = True
            return Ok(self)

        snapshots = drop_broken_snapshots(snapshots)

        # Build per-skill time series
        skill_dates: dict[str, list[date]] = {}
        skill_freqs: dict[str, list[float]] = {}
        for dt, data in snapshots:
            for skill, freq in data.items():
                skill_dates.setdefault(skill, []).append(dt)
                skill_freqs.setdefault(skill, []).append(freq)
        # Полные наблюдаемые истории для /forecast/observed.
        self._observed = {
            skill: sorted(zip(skill_dates[skill], skill_freqs[skill]), key=lambda p: p[0])
            for skill in skill_dates
        }

        # Fit linear trend per skill
        for skill in skill_dates:
            x = np.array([(d - snapshots[0][0]).days for d in skill_dates[skill]], dtype=float)
            y = np.array(skill_freqs[skill], dtype=float)
            if len(x) < 2:
                self._models[skill] = {"slope": 0.0, "intercept": y[0] if len(y) > 0 else 0.0, "n": len(x), "last_freq": float(y[-1]) if len(y) > 0 else 0.0}
                continue
            slope, intercept = np.polyfit(x, y, 1)
            residuals = y - (slope * x + intercept)
            rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) > 1 else 0.0
            mean_y = float(np.mean(y))
            mape = rmse / max(mean_y, 1.0)
            self._models[skill] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "n": len(x),
                "rmse": rmse,
                "mape": mape,
                "last_freq": float(y[-1]),
            }

        # Merge with skill_frequencies (current snapshot)
        if skill_frequencies:
            for skill, freq in skill_frequencies.items():
                if skill not in self._models:
                    self._models[skill] = {"slope": 0.0, "intercept": freq, "n": 1, "rmse": None, "mape": None, "last_freq": freq}

        self._is_fitted = True
        logger.info("trend_forecast_fitted", models=len(self._models), snapshots=len(snapshots))
        return Ok(self)

    def predict(self, skill: str, months: int = 12) -> Result[ForecastResult, DomainError]:
        if months < 1 or months > 60:
            return Err(DomainError(f"months must be 1-60, got {months}"))

        model = self._models.get(skill)
        if model is None:
            return Err(DomainError(f"Skill '{skill}' not found"))

        n = model["n"]
        last_freq = model["last_freq"]

        if n < 2 or model["slope"] == 0.0:
            return Ok(ForecastResult(
                skill=skill,
                current_frequency=round(last_freq, 4),
                predicted_growth=0.0,
                confidence=0.2,
                next_year_frequency=round(last_freq, 4),
                engine_used="trend_flat",
                data_points=n,
                mape=None,
                forecast_months=0,
            ))

        # Limit forecast horizon: don't extrapolate beyond half the observed
        # history (mirrors ProphetForecastEngine) and anchor on the last point.
        days_ahead = min(months, max(1, n // 2)) * 30
        predicted = last_freq + model["slope"] * days_ahead
        predicted = max(predicted, 0.0)

        growth = (predicted - last_freq) / max(last_freq, 1.0)
        growth = max(min(growth, self.MAX_GROWTH_CAP), -self.MAX_GROWTH_CAP)

        _mape = model.get("mape")
        confidence = max(0.0, min(0.9, 1.0 - min((_mape if _mape is not None else 1.0) * 2.0, 0.8)))
        confidence *= min(1.0, n / 4.0)

        return Ok(ForecastResult(
            skill=skill,
            current_frequency=round(last_freq, 4),
            predicted_growth=round(growth, 4),
            confidence=round(confidence, 4),
            next_year_frequency=round(max(predicted, 0.0), 4),
            engine_used="trend",
            data_points=n,
            mape=round(_mape, 4) if _mape is not None else None,
            forecast_months=min(months, max(1, n // 2)),
        ))

    def forecast(self, skill: str, months: int = 12) -> Result[ForecastResult, DomainError]:
        """Alias for predict() — used by ProphetForecastEngine."""
        return self.predict(skill, months)

    def forecast_all(self, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        results = []
        for skill in self._models:
            match self.predict(skill, months):
                case Ok(r):
                    results.append(r)
                case _:
                    pass
        return Ok(results)

    def top_growing(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                # Flats остаются в выдаче (контракт тестов), но маркированы:
                # mape=None + forecast_months=0 + data_points<2. Сортировка —
                # сначала валидные тренды, затем flats.
                results = [r for r in results if r.current_frequency >= self.MIN_FREQ]
                results.sort(
                    key=lambda x: (
                        x.engine_used.startswith("trend_flat"),
                        -x.predicted_growth,
                    )
                )
                return Ok(results[:n])
            case Err(e):
                return Err(e)

    def top_declining(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [r for r in results if r.current_frequency >= self.MIN_FREQ]
                results.sort(key=lambda x: x.predicted_growth)
                return Ok([r for r in results if r.predicted_growth < 0][:n])
            case Err(e):
                return Err(e)
