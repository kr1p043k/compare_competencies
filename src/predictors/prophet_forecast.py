"""Prophet-based forecast engine with DB-sourced time series."""

from __future__ import annotations

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
    import logging
    # Suppress cmdstanpy: remove its handler and set level to WARNING
    _cmdstan_logger = logging.getLogger("cmdstanpy")
    _cmdstan_logger.setLevel(logging.WARNING)
    for _h in _cmdstan_logger.handlers[:]:
        _cmdstan_logger.removeHandler(_h)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy.cmdstan").setLevel(logging.WARNING)
except ImportError:
    Prophet = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


@dataclass
class Snapshot:
    date: date
    frequencies: dict[str, float]


async def load_time_series(session: AsyncSession) -> Result[list[Snapshot], DomainError]:
    """Build monthly skill-frequency snapshots using running total (rolling window).

    Each month's value = total unique vacancies with this skill from start up to
    the end of that month. This prevents incomplete-month dips from skewing trends.
    """
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
    monthly: dict[date, Counter] = {}
    for row in rows:
        m = row.month if isinstance(row.month, date) else row.month.date()
        monthly.setdefault(m, Counter())[row.skill] += row.freq
    if not monthly:
        return Err(DomainError("No vacancies with parsed_skills in database"))

    # Convert to running total: each snapshot includes all prior months
    running: dict[str, float] = {}
    snapshots: list[Snapshot] = []
    for m in sorted(monthly):
        for skill, freq in monthly[m].items():
            running[skill] = running.get(skill, 0.0) + freq
        snapshots.append(Snapshot(m, dict(running)))
    return Ok(snapshots)


class ProphetForecastEngine(BasePredictor):
    """Forecast engine using Prophet for skills with >= 3 history points and
    actual frequency >= MIN_FREQ, falling back to SkillForecastEngine."""

    MIN_FREQ = 10
    MAX_GROWTH_CAP = 20.0

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
        return history

    def _fit_prophet_for_skill(self, skill: str, points: list[tuple[date, float]]):
        df = pd.DataFrame({"ds": [p[0] for p in points], "y": [p[1] for p in points]})
        n_points = len(points)
        model = Prophet(
            yearly_seasonality=n_points >= 24,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.80,
            changepoint_prior_scale=0.5 if n_points < 12 else 0.05,
        )
        model.fit(df)
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
        for skill, points in history.items():
            last_actual = points[-1][1]
            self._last_actual_freq[skill] = last_actual
            if len(points) >= 3 and last_actual >= self.MIN_FREQ:
                try:
                    self._models[skill] = self._fit_prophet_for_skill(skill, points)
                except Exception as e:
                    logger.warning("prophet_skill_fit_failed", skill=skill, error=str(e))
            else:
                self._skill_history[skill] = points

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
            # Penalize confidence when few data points
            n_pts = len(model.history) if hasattr(model, "history") and model.history is not None else 3
            if n_pts < 3:
                conf *= n_pts / 3.0
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
                results = [r for r in results if r.current_frequency >= self.MIN_FREQ]
                results.sort(key=lambda x: x.predicted_growth, reverse=True)
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
