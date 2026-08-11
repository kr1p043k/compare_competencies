"""Trend-based fallback forecast engine (replaces broken GA).

Used when Prophet is unavailable or fails. Simple linear regression
on historical skill frequency data from trend_snapshots.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from src import config, Ok, Err, Result
from src.errors import DomainError
from src.predictors.base import BasePredictor

logger = structlog.get_logger(__name__)


@dataclass
class ForecastResult:
    skill: str
    current_frequency: float
    predicted_growth: float
    confidence: float
    next_year_frequency: float
    engine_used: str = "trend"


class SkillForecastEngine(BasePredictor):
    """Lightweight trend-based forecast using historical snapshots.

    Fits a simple linear regression per skill: freq ~ time.
    Predicts future frequency based on trend direction + strength.
    Falls back to flat (no growth) when < 2 data points.
    """

    MIN_FREQ = 50
    MAX_GROWTH_CAP = 10.0
    MIN_POINTS_FOR_TOP = 3

    def __init__(self):
        self._models: dict[str, dict] = {}
        self._is_fitted = False

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
    ) -> Result["SkillForecastEngine", Exception]:
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
                    self._models[skill] = {"slope": 0.0, "intercept": 0.0, "n": 1, "rmse": 0.0, "mape": 0.0, "last_freq": freq, "last_share": 0.0}
            self._is_fitted = True
            return Ok(self)

        # Build per-skill time series, normalized by per-snapshot total volume
        # (so expanding collection coverage does not fake growth).
        snapshot_totals: dict[date, float] = {}
        for dt, data in snapshots:
            snapshot_totals[dt] = float(sum(v for v in data.values() if isinstance(v, (int, float))))

        skill_dates: dict[str, list[date]] = {}
        skill_shares: dict[str, list[float]] = {}
        skill_freqs: dict[str, list[float]] = {}
        for dt, data in snapshots:
            total = snapshot_totals.get(dt, 0.0)
            for skill, freq in data.items():
                skill_dates.setdefault(skill, []).append(dt)
                skill_freqs.setdefault(skill, []).append(float(freq))
                skill_shares.setdefault(skill, []).append(float(freq) / total if total > 0 else 0.0)

        # Fit linear trend per skill on relative share
        for skill in skill_dates:
            x = np.array([(d - snapshots[0][0]).days for d in skill_dates[skill]], dtype=float)
            y = np.array(skill_shares[skill], dtype=float)
            y_abs = np.array(skill_freqs[skill], dtype=float)
            if len(x) < 2:
                self._models[skill] = {
                    "slope": 0.0,
                    "intercept": y[0] if len(y) > 0 else 0.0,
                    "n": len(x),
                    "rmse": 0.0,
                    "mape": 0.0,
                    "last_freq": float(y_abs[-1]) if len(y_abs) > 0 else 0.0,
                    "last_share": float(y[-1]) if len(y) > 0 else 0.0,
                }
                continue
            slope, intercept = np.polyfit(x, y, 1)
            residuals = y - (slope * x + intercept)
            rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) > 1 else 0.0
            mean_y = float(np.mean(y))
            mape = rmse / max(mean_y, 1e-6)
            self._models[skill] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "n": len(x),
                "rmse": rmse,
                "mape": mape,
                "last_freq": float(y_abs[-1]),
                "last_share": float(y[-1]),
            }

        # Merge with skill_frequencies (current snapshot)
        if skill_frequencies:
            for skill, freq in skill_frequencies.items():
                if skill not in self._models:
                    self._models[skill] = {"slope": 0.0, "intercept": 0.0, "n": 1, "rmse": 0.0, "mape": 0.0, "last_freq": freq, "last_share": 0.0}

        self._is_fitted = True
        self._models["__n_snapshots__"] = {"slope": 0.0, "intercept": 0.0, "n": len(snapshots), "rmse": 0.0, "mape": 0.0, "last_freq": 0.0, "last_share": 0.0}
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
        last_share = model.get("last_share", 0.0)

        if n < 2 or model["slope"] == 0.0:
            return Ok(ForecastResult(
                skill=skill,
                current_frequency=round(last_freq, 4),
                predicted_growth=0.0,
                confidence=0.2,
                next_year_frequency=round(last_freq, 4),
                engine_used="trend_flat",
            ))

        # Limit forecast horizon by data depth: 3 pts -> ~1 month, 6 pts -> 3 months
        max_horizon = max(1, n // 2)
        eff_months = min(months, max_horizon)
        days_ahead = eff_months * 30
        # Extrapolate from the LAST observed share (not intercept at x=0),
        # otherwise a declining skill with negative slope is wrongly predicted as growing.
        predicted_share = last_share + model["slope"] * days_ahead
        predicted_share = max(predicted_share, 0.0)

        if last_share > 0:
            growth = (predicted_share - last_share) / last_share
        else:
            growth = 0.0
        growth = max(min(growth, self.MAX_GROWTH_CAP), -self.MAX_GROWTH_CAP)

        confidence = max(0.0, min(0.9, 1.0 - min(model["mape"] * 2.0, 0.8)))
        confidence *= min(1.0, n / 4.0)

        # Derive absolute next frequency from share growth (baseline = last actual)
        next_year_frequency = max(last_freq * (1.0 + growth), 0.0)

        return Ok(ForecastResult(
            skill=skill,
            current_frequency=round(last_freq, 4),
            predicted_growth=round(growth, 4),
            confidence=round(confidence, 4),
            next_year_frequency=round(next_year_frequency, 4),
            engine_used="trend",
        ))

    def forecast(self, skill: str, months: int = 12) -> Result[ForecastResult, DomainError]:
        """Alias for predict() — used by ProphetForecastEngine."""
        return self.predict(skill, months)

    def forecast_all(self, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        results = []
        for skill in self._models:
            if skill.startswith("__"):
                continue
            match self.predict(skill, months):
                case Ok(r):
                    results.append(r)
                case _:
                    pass
        return Ok(results)

    def top_growing(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [
                    r for r in results
                    if r.current_frequency >= self.MIN_FREQ
                    and self._models.get(r.skill, {}).get("n", 0) >= self.MIN_POINTS_FOR_TOP
                    and r.predicted_growth > 0
                ]
                # Exclude unreliable predictions: growth > 200% with low confidence
                results = [r for r in results if not (r.predicted_growth > 2.0 and r.confidence < 0.3)]
                if not results:
                    return Ok([])
                max_freq = max(r.current_frequency for r in results) or 1
                max_growth = max(r.predicted_growth for r in results) or 1
                results.sort(
                    key=lambda x: 0.3 * (x.predicted_growth / max_growth) + 0.7 * (x.current_frequency / max_freq),
                    reverse=True,
                )
                return Ok(results[:n])
            case Err(e):
                return Err(e)

    def top_declining(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [
                    r for r in results
                    if r.current_frequency >= self.MIN_FREQ
                    and self._models.get(r.skill, {}).get("n", 0) >= self.MIN_POINTS_FOR_TOP
                ]
                results.sort(key=lambda x: x.predicted_growth)
                return Ok([r for r in results if r.predicted_growth < 0][:n])
            case Err(e):
                return Err(e)

    def top_popular(self, n: int = 25, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [r for r in results if r.current_frequency >= self.MIN_FREQ]
                results.sort(key=lambda x: x.current_frequency, reverse=True)
                return Ok(results[:n])
            case Err(e):
                return Err(e)
