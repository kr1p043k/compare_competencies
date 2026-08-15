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


def _holdout_mape(points: list[tuple[date, float]]) -> float:
    """Hold-out MAPE: линейный тренд на всех точках кроме последней, ошибка на последней.

    Возвращает MAPE (0 = идеально, 1 = 100% ошибка). При стабильных данных близко к 0.
    """
    import numpy as np
    if len(points) < 2:
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
    last_date, actual = points[-1]
    days_ahead = (last_date - train[0][0]).days
    pred = intercept + slope * days_ahead
    if abs(actual) < 1.0:
        return 0.0
    return float(abs(pred - actual) / abs(actual))


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

    # 3. Sort by date
    file_snapshots.sort(key=lambda x: x[0])
    file_snapshots = _interpolate_missing_months(file_snapshots)
    return Ok([Snapshot(m, data) for m, data in file_snapshots])


def _interpolate_missing_months(snapshots: list[tuple[date, dict[str, float]]]) -> list[tuple[date, dict[str, float]]]:
    """Заполняет пропущенные календарные месяцы линейной интерполяцией.

    Нерегулярные снимки (напр. апр, май, июн, авг) дают Prophet'у разрозненные
    точки без июля — интерполяция восстанавливает ежемесячный ряд.
    """
    if len(snapshots) < 2:
        return snapshots
    result: list[tuple[date, dict[str, float]]] = []
    for i in range(len(snapshots)):
        cur_date, cur_data = snapshots[i]
        if i == 0:
            result.append((cur_date, cur_data))
            continue
        prev_date, prev_data = snapshots[i - 1]
        gap_months = (cur_date.year - prev_date.year) * 12 + (cur_date.month - prev_date.month)
        if gap_months <= 1:
            result.append((cur_date, cur_data))
            continue
        all_skills = set(prev_data) | set(cur_data)
        for step in range(1, gap_months):
            t = step / gap_months
            month_idx = prev_date.month + step - 1
            mid_year = prev_date.year + month_idx // 12
            mid_month = month_idx % 12 + 1
            mid = date(mid_year, mid_month, 1)
            interp: dict[str, float] = {}
            for skill in all_skills:
                a = prev_data.get(skill, 0.0)
                b = cur_data.get(skill, 0.0)
                val = a + (b - a) * t
                if val >= 1.0:
                    interp[skill] = round(val, 1)
            result.append((mid, interp))
        result.append((cur_date, cur_data))
    return result


class ProphetForecastEngine(BasePredictor):
    """Forecast engine using Prophet for skills with >= 3 history points and
    actual frequency >= MIN_FREQ, falling back to SkillForecastEngine."""

    MIN_FREQ = 10
    MAX_GROWTH_CAP = 2.0
    # Top-prediction display: only show skills with meaningful frequency
    TOP_DISPLAY_MIN_FREQ = 50

    def __init__(self):
        self._models: dict[str, Prophet] = {}
        self._fallback_engine: SkillForecastEngine | None = None
        self._skill_history: dict[str, list[tuple[date, float]]] = {}
        self._last_actual_freq: dict[str, float] = {}
        self._skill_mape: dict[str, float] = {}
        self._skill_npoints: dict[str, int] = {}
        self._is_fitted = False
        self._n_snapshots = 0

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

        self._n_snapshots = len(snapshots)
        history = self._gather_history(snapshots)

        # Separate skills by data depth: Prophet (≥3 pts) vs trend (fallback)
        prophet_candidates: list[tuple[str, list[tuple[date, float]]]] = []
        for skill, points in history.items():
            last_actual = points[-1][1]
            self._last_actual_freq[skill] = last_actual
            self._skill_npoints[skill] = len(points)
            if len(points) >= 5:
                self._skill_mape[skill] = _holdout_mape(points)
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
            # Limit forecast horizon based on data points, but be more generous:
            # 3 pts -> 3m, 6 pts -> 6m, 12+ pts -> 12m (was n_pts//2 — too conservative).
            max_months = max(1, min(n_pts, 12))
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
                # Tighten growth cap for low-data skills (prevents absurd spikes)
                tight_cap = 1.5 if n_pts < 4 else 2.0
                growth = max(min(growth, tight_cap), -tight_cap)
            return Ok(ForecastResult(
                skill=skill,
                current_frequency=round(last_actual, 4),
                predicted_growth=round(growth, 4),
                confidence=round(max(conf, 0.0), 4),
                next_year_frequency=round(next_freq, 4),
                engine_used="prophet",
                data_points=self._skill_npoints.get(skill, n_pts),
                mape=round(self._skill_mape.get(skill, 0.0), 4),
                forecast_months=months,
            ))
        if self._fallback_engine:
            result = self._fallback_engine.forecast(skill, min(months, self.max_forecast_months()))
            if result.is_ok():
                fr = result.unwrap()
                n_pts = self._skill_npoints.get(skill, 0)
                # Явный статус: недостаточно данных для прогноза
                if n_pts < 3:
                    return Ok(ForecastResult(
                        skill=fr.skill,
                        current_frequency=fr.current_frequency,
                        predicted_growth=0.0,
                        confidence=0.0,
                        next_year_frequency=fr.current_frequency,
                        engine_used="insufficient_data",
                        data_points=n_pts,
                        mape=0.0,
                        forecast_months=0,
                    ))
                return Ok(ForecastResult(
                    skill=fr.skill,
                    current_frequency=fr.current_frequency,
                    predicted_growth=fr.predicted_growth,
                    confidence=fr.confidence,
                    next_year_frequency=fr.next_year_frequency,
                    engine_used="trend",
                    data_points=n_pts,
                    mape=round(self._skill_mape.get(skill, 0.0), 4),
                    forecast_months=min(months, self.max_forecast_months()),
                ))
            return result
        return Err(DomainError(f"Skill '{skill}' not found"))

    def max_forecast_months(self) -> int:
        """Return max safe forecast horizon based on snapshot count."""
        return max(1, min(self._n_snapshots, 12))

    def forecast_all(self, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        results = []
        for skill in self._models:
            match self.predict(skill, months):
                case Ok(r):
                    results.append(r)
                case _:
                    pass
        if self._fallback_engine:
            match self._fallback_engine.forecast_all(min(months, self.max_forecast_months())):
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
                results = [r for r in results if r.current_frequency >= self.TOP_DISPLAY_MIN_FREQ and r.next_year_frequency > 0 and r.predicted_growth > 0]
                # Exclude unreliable predictions: growth > 200% with confidence < 30%
                results = [r for r in results if not (r.predicted_growth > 2.0 and r.confidence < 0.3)]
                if not results:
                    return Ok([])
                max_freq = max(r.current_frequency for r in results) or 1
                max_growth = max(r.predicted_growth for r in results) or 1
                results.sort(key=lambda x: 0.3 * (x.predicted_growth / max_growth) + 0.7 * (x.current_frequency / max_freq), reverse=True)
                return Ok(results[:n])
            case Err(e):
                return Err(e)

    def top_declining(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [r for r in results if r.current_frequency >= self.TOP_DISPLAY_MIN_FREQ and r.next_year_frequency > 0 and r.predicted_growth < 0]
                results.sort(key=lambda x: x.predicted_growth)
                return Ok(results[:n])
            case Err(e):
                return Err(e)

    def top_popular(self, n: int = 25, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results.sort(key=lambda x: x.current_frequency, reverse=True)
                return Ok(results[:n])
            case Err(e):
                return Err(e)
