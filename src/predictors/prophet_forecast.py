"""Prophet-based forecast engine with DB-sourced time series."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src import Err, Ok, Result
from src.errors import DomainError
from src.predictors.base import BasePredictor
from src.predictors.skill_forecast import ForecastResult, SkillForecastEngine

try:
    from prophet import Prophet
    from cmdstanpy.utils.logging import disable_logging as _disable_cmdstan
    _disable_cmdstan().__enter__()
except ImportError:
    Prophet = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


@dataclass
class Snapshot:
    date: date
    frequencies: dict[str, float]


async def load_time_series(session: AsyncSession) -> Result[list[Snapshot], DomainError]:
    """Build monthly skill-frequency snapshots from freq_market_*.json files,
    supplemented by parsed_skills from DB for skills not in those files.

    Each snapshot = per-month frequency (absolute count, not running total).
    """
    import json
    from pathlib import Path
    from src import config

    # 1. Load freq_market_*.json files as primary source
    history_dir: Path = config.HISTORY_DIR
    file_snapshots: list[tuple[date, dict[str, float]]] = []
    all_skills_in_files: set[str] = set()

    for f in sorted(history_dir.glob("freq_market_*.json")):
        try:
            raw = json.loads(f.read_bytes())
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
            file_snapshots.append((dt, data))
            all_skills_in_files.update(data.keys())
        except Exception:
            continue

    if not file_snapshots:
        logger.warning("no_freq_market_files_found_falling_back_to_db")

    # 2. Supplement with parsed_skills from DB for NEW skills not in freq_market
    try:
        rows = await session.execute(text("""
            SELECT
                date_trunc('month', v.published_at::timestamp)::date AS month,
                ps::text AS skill,
                COUNT(DISTINCT v.id) AS freq
            FROM vacancies v
            CROSS JOIN LATERAL jsonb_array_elements_text(v.parsed_skills::jsonb) AS ps
            WHERE v.parsed_skills IS NOT NULL
              AND v.parsed_skills::text != '[]'
              AND v.published_at IS NOT NULL
            GROUP BY month, ps::text
            ORDER BY month
        """))
        db_monthly: dict[date, Counter] = {}
        for row in rows:
            m = row.month if isinstance(row.month, date) else row.month.date()
            if row.skill not in all_skills_in_files:
                db_monthly.setdefault(m, Counter())[row.skill] += row.freq

        # Convert DB data into snapshot format
        for m in sorted(db_monthly):
            file_snapshots.append((m, dict(db_monthly[m])))
    except Exception:
        logger.warning("db_supplement_failed")

    if not file_snapshots:
        return Err(DomainError("No snapshot data available"))

    # 3. Sort by date and return
    file_snapshots.sort(key=lambda x: x[0])
    return Ok([Snapshot(m, data) for m, data in file_snapshots])


class ProphetForecastEngine(BasePredictor):
    """Forecast engine using Prophet for skills with >= 3 history points and
    actual frequency >= MIN_FREQ, falling back to SkillForecastEngine."""

    MIN_FREQ = 10
    MAX_GROWTH_CAP = 20.0
    # Top-prediction display: only show skills with meaningful frequency
    TOP_DISPLAY_MIN_FREQ = 50

    def __init__(self):
        self._models: dict[str, Prophet] = {}
        self._fallback_engine: SkillForecastEngine | None = None
        self._skill_history: dict[str, list[tuple[date, float]]] = {}
        self._last_actual_freq: dict[str, float] = {}
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "ProphetForecast"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _gather_history(self, snapshots: list[Snapshot]):
        history: dict[str, list[tuple[date, float]]] = {}
        for snap in snapshots:
            for skill, freq in snap.frequencies.items():
                history.setdefault(skill, []).append((snap.date, freq))
        for skill in history:
            history[skill].sort(key=lambda x: x[0])
        # Exclude skills not present in the latest snapshot (removed during cleanup)
        if snapshots:
            latest_skills = set(snapshots[-1].frequencies.keys())
            for skill in list(history.keys()):
                if skill not in latest_skills:
                    del history[skill]
        return history

    def _fit_prophet_for_skill(self, skill: str, points: list[tuple[date, float]]):
        from cmdstanpy.utils.logging import disable_logging
        import numpy as np
        df = pd.DataFrame({"ds": [p[0] for p in points], "y": [p[1] for p in points]})
        n_points = len(points)
        # Sanity check: detect extreme variance that causes "inf in matrix" errors
        y = df["y"].values
        if np.any(~np.isfinite(y)) or (y.max() - y.min()) > 1e6:
            raise ValueError(f"Unstable data for Prophet: min={y.min()}, max={y.max()}, n={n_points}")
        model = Prophet(
            yearly_seasonality=n_points >= 24,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.80,
            changepoint_prior_scale=0.05 if n_points < 6 else (0.5 if n_points < 12 else 0.05),
        )
        with disable_logging():
            model.fit(df, iter=1000)
        return model

    def fit(
        self,
        snapshots: list[Snapshot],
        fallback_freqs: dict[str, float] | None = None,
    ) -> Result[ProphetForecastEngine, DomainError]:
        import logging
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)

        if not snapshots:
            return Err(DomainError("No snapshots provided to Prophet engine"))

        history = self._gather_history(snapshots)

        # Separate skills by data depth: Prophet (≥3 pts) vs trend (fallback)
        prophet_candidates: list[tuple[str, list[tuple[date, float]]]] = []
        for skill, points in history.items():
            last_actual = points[-1][1]
            self._last_actual_freq[skill] = last_actual
            if len(points) >= 3 and last_actual >= self.MIN_FREQ:
                prophet_candidates.append((skill, points))
            else:
                self._skill_history[skill] = points

        # Parallel Prophet fitting
        if prophet_candidates:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(self._fit_prophet_for_skill, s, p): s for s, p in prophet_candidates}
                for future in as_completed(futures):
                    skill = futures[future]
                    try:
                        self._models[skill] = future.result()
                    except Exception as e:
                        logger.warning("prophet_skill_fit_failed", skill=skill, error=str(e))

        prophet_skills = len(self._models)
        fallback_skills = len(self._skill_history)
        logger.info(
            "prophet_fitted",
            prophet_skills=prophet_skills,
            fallback_skills=fallback_skills,
            snapshots=len(snapshots),
        )

        if fallback_freqs:
            gen = SkillForecastEngine()
            match gen.fit(fallback_freqs):
                case Ok(_):
                    self._fallback_engine = gen
                case Err(e):
                    logger.warning("prophet_fallback_engine_fit_failed", error=str(e))

        if not self._models and not self._fallback_engine:
            return Err(DomainError("No skills could be fitted by Prophet or fallback"))

        self._is_fitted = True
        return Ok(self)

    def predict(self, skill: str, months: int = 12) -> Result[ForecastResult, DomainError]:
        if months < 1 or months > 60:
            return Err(DomainError(f"months must be 1-60, got {months}"))

        if skill in self._models:
            model = self._models[skill]
            n_pts = len(model.history) if hasattr(model, "history") and model.history is not None else 3
            # Limit forecast horizon based on data points: 3 pts → 1m, 6 pts → 3m, 12+ pts → 12m
            max_months = max(1, n_pts // 2)
            if months > max_months:
                months = max_months
            future = model.make_future_dataframe(periods=months, freq="ME")
            forecast = model.predict(future)
            last_row = forecast.iloc[-1]
            next_freq = max(float(last_row["yhat"]), 0.0)

            last_actual = self._last_actual_freq.get(skill, 0.0)
            baseline = max(last_actual, self.MIN_FREQ)
            growth = (next_freq - baseline) / baseline
            growth = max(min(growth, self.MAX_GROWTH_CAP), -self.MAX_GROWTH_CAP)

            uncertainty = float(last_row["yhat_upper"] - last_row["yhat_lower"])
            conf = max(0.0, 1.0 - min(uncertainty / max(next_freq, 1.0), 0.85))
            # Penalize confidence and cap growth when few data points
            n_pts = len(model.history) if hasattr(model, "history") and model.history is not None else 3
            if n_pts < 6:
                conf *= n_pts / 6.0
                # Tighten growth cap for low-data skills (prevents absurd 600% spikes)
                tight_cap = 2.0 if n_pts < 4 else 10.0
                growth = max(min(growth, tight_cap), -tight_cap)
            return Ok(ForecastResult(
                skill=skill,
                current_frequency=round(last_actual, 4),
                predicted_growth=round(growth, 4),
                confidence=round(max(conf, 0.0), 4),
                next_year_frequency=round(next_freq, 4),
                engine_used="prophet",
            ))
        if self._fallback_engine:
            result = self._fallback_engine.forecast(skill, months)
            if result.is_ok():
                fr = result.unwrap()
                return Ok(ForecastResult(
                    skill=fr.skill,
                    current_frequency=fr.current_frequency,
                    predicted_growth=fr.predicted_growth,
                    confidence=fr.confidence,
                    next_year_frequency=fr.next_year_frequency,
                    engine_used="fallback",
                ))
            return result
        return Err(DomainError(f"Skill '{skill}' not found"))

    def max_forecast_months(self) -> int:
        """Return max safe forecast horizon based on data points across all models."""
        max_n = 0
        for skill, model in self._models.items():
            n = len(model.history) if hasattr(model, "history") and model.history is not None else 3
            max_n = max(max_n, n)
        return max(1, max_n // 2) if max_n > 0 else 1

    def forecast_all(self, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        results = []
        for skill in self._models:
            match self.predict(skill, months):
                case Ok(r):
                    results.append(r)
                case _:
                    pass
        if self._fallback_engine:
            match self._fallback_engine.forecast_all(months):
                case Ok(fb):
                    for r in fb:
                        if not any(ex.skill == r.skill for ex in results):
                            results.append(r)
                case _:
                    pass
        return Ok(results)

    def top_growing(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [r for r in results if r.current_frequency >= self.TOP_DISPLAY_MIN_FREQ and r.next_year_frequency > 0]
                # Hybrid ranking: blend growth rate and popularity
                max_freq = max(r.current_frequency for r in results) or 1
                max_growth = max(abs(r.predicted_growth) for r in results) or 1
                results.sort(
                    key=lambda x: 0.4 * (x.predicted_growth / max_growth) + 0.6 * (x.current_frequency / max_freq),
                    reverse=True,
                )
                return Ok(results[:n])
            case Err(e):
                return Err(e)

    def top_declining(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [r for r in results if r.current_frequency >= self.TOP_DISPLAY_MIN_FREQ and r.next_year_frequency > 0]
                max_freq = max(r.current_frequency for r in results) or 1
                max_growth = max(abs(r.predicted_growth) for r in results) or 1
                results.sort(
                    key=lambda x: 0.4 * (x.predicted_growth / max_growth) + 0.6 * (x.current_frequency / max_freq),
                )
                return Ok([r for r in results if r.predicted_growth < 0][:n])
            case Err(e):
                return Err(e)

    def top_popular(self, n: int = 25, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results.sort(key=lambda x: x.current_frequency, reverse=True)
                return Ok(results[:n])
            case Err(e):
                return Err(e)
