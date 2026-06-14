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
except ImportError:
    Prophet = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


@dataclass
class Snapshot:
    date: date
    frequencies: dict[str, float]


async def load_time_series(session: AsyncSession) -> Result[list[Snapshot], DomainError]:
    """Build monthly skill-frequency snapshots from vacancies.parsed_skills."""
    rows = await session.execute(text("""
        SELECT
            date_trunc('month', v.published_at::timestamp)::date AS month,
            ps::text AS skill,
            COUNT(DISTINCT v.id) AS freq
        FROM vacancies v
        CROSS JOIN LATERAL jsonb_array_elements_text(v.parsed_skills) AS ps
        WHERE v.parsed_skills IS NOT NULL
          AND v.parsed_skills != '[]'::jsonb
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
    return Ok([Snapshot(m, dict(f)) for m, f in sorted(monthly.items())])


class ProphetForecastEngine(BasePredictor):
    """Forecast engine using Prophet for skills with >= 2 history points,
    falling back to GeneticForecastSkill for the rest."""

    def __init__(self):
        self._models: dict[str, Prophet] = {}
        self._fallback_engine: SkillForecastEngine | None = None
        self._skill_history: dict[str, list[tuple[date, float]]] = {}
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
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
        )
        model.fit(df)
        return model

    def fit(
        self,
        snapshots: list[Snapshot],
        fallback_freqs: dict[str, float] | None = None,
    ) -> Result[ProphetForecastEngine, DomainError]:
        if not snapshots:
            return Err(DomainError("No snapshots provided to Prophet engine"))

        history = self._gather_history(snapshots)
        for skill, points in history.items():
            if len(points) >= 2:
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
        if skill in self._models:
            model = self._models[skill]
            future = model.make_future_dataframe(periods=months, freq="ME")
            forecast = model.predict(future)
            current_idx = max(0, len(forecast) - months - 1)
            current_row = forecast.iloc[current_idx]
            last_row = forecast.iloc[-1]
            current_freq = max(float(current_row["yhat"]), 0.0)
            next_freq = max(float(last_row["yhat"]), 0.0)
            growth = (next_freq - current_freq) / max(current_freq, 0.001)
            conf = min(1.0, 1.0 - float(last_row["yhat_upper"] - last_row["yhat_lower"]) / max(next_freq, 0.001))
            return Ok(ForecastResult(
                skill=skill,
                current_frequency=round(current_freq, 4),
                predicted_growth=round(growth, 4),
                confidence=round(max(conf, 0.0), 4),
                next_year_frequency=round(next_freq, 4),
            ))
        if self._fallback_engine:
            return self._fallback_engine.forecast(skill, months)
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
                results.sort(key=lambda x: x.predicted_growth, reverse=True)
                return Ok(results[:n])
            case Err(e):
                return Err(e)
