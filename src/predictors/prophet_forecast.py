"""Prophet-based forecast engine with DB-sourced time series."""

from __future__ import annotations

import asyncio
import contextlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src import Err, Ok, Result
from src.errors import DomainError
from src.predictors.base import BasePredictor
from src.predictors.skill_forecast import (
    ForecastResult,
    SkillForecastEngine,
    drop_broken_snapshots,
)

# Статус фонового фита Prophet (для countdown в UI): обновляется по ходу
# ThreadPool-фита; читается endpoint'ом /forecast/engine без блокировок.
FIT_STATUS: dict = {"state": "idle", "done": 0, "total": 0, "started_at": None}

try:
    from cmdstanpy.utils.logging import disable_logging as _disable_cmdstan
    from prophet import Prophet
    _disable_cmdstan().__enter__()
except ImportError:
    Prophet = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

# Prophet needs one full annual cycle before yearly seasonality can be
# estimated; 12 monthly points is that minimum. The previous cutoff of 24
# switched it off for the current 23-snapshot history.
YEARLY_SEASONALITY_MIN_POINTS = 12


def _holdout_mape(points: list[tuple[date, float]]) -> float | None:
    """Hold-out MAPE: линейный тренд на всех точках кроме последней, ошибка на последней.

    Возвращает MAPE (0 = идеально, 1 = 100% ошибка). При стабильных данных близко к 0.
    None = посчитать нельзя (<2 точек, вырожденный ряд, actual < 1).
    Раньше возвращал 0.0 («идеально») при отсутствии данных - враньё:
    UI прячет бейдж только при mape<=0, и 0.0 выглядел как отличный прогноз.
    """
    import numpy as np
    if len(points) < 2:
        return None
    train = points[:-1]
    x = np.array([(d - train[0][0]).days for d, _ in train], dtype=float)
    y = np.array([f for _, f in train], dtype=float)
    if len(x) < 2 or np.all(x == x[0]):
        return None
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return None
    last_date, actual = points[-1]
    days_ahead = (last_date - train[0][0]).days
    pred = intercept + slope * days_ahead
    if abs(actual) < 1.0:
        return None
    return float(abs(pred - actual) / abs(actual))


def _rounded_mape(mapping: dict[str, float | None], skill: str) -> float | None:
    v = mapping.get(skill)
    return round(v, 4) if v is not None else None


@dataclass
class Snapshot:
    date: date
    frequencies: dict[str, float]


def anchor_monthly_counts(
    monthly: dict[date, Counter],
    totals: dict[date, int],
) -> dict[date, dict[str, float]]:
    """Привести помесячные counts к единицам последнего месяца.

    Объём коллекции растёт (3 957 → 15 046 разобранных вакансий за 5 мес.),
    и абсолютные counts путают рост коллекции с ростом спроса. Делим каждый
    месяц на его объём и умножаем на объём последнего месяца: уровни остаются
    абсолютными (пороги MIN_FREQ работают), а форма ряда = динамика долей.
    Последний месяц не меняется (k=1), так что current_frequency честен.
    """
    months = sorted(m for m in monthly if (totals.get(m) or 0) > 0)
    if not months:
        return {}
    v_last = totals[months[-1]]
    out: dict[date, dict[str, float]] = {}
    for m in months:
        k = v_last / totals[m]
        out[m] = {s: round(c * k, 1) for s, c in monthly[m].items()}
    return out


def anchor_snapshot_shares(
    snapshots: list[tuple[date, dict[str, float]]],
) -> dict[date, dict[str, float]]:
    """Свернуть снимки к помесячным долям и вернуть в единицах последнего снимка.

    Объём сбора между снимками гуляет в разы (6 221 и 65 615 навыко-счётчиков на
    соседних датах), поэтому сырые counts рисуют ложные скачки. Делим каждый
    снимок на его сумму и умножаем на сумму последнего: уровень последнего месяца
    честен (k=1), а форма ряда - динамика долей. Несколько снимков одного месяца
    усредняются по долям.
    """
    ordered = sorted(snapshots, key=lambda x: x[0])
    if not ordered:
        return {}
    ref = sum(ordered[-1][1].values())
    if ref <= 0:
        return {}
    by_month: dict[date, list[dict[str, float]]] = {}
    for d, data in ordered:
        total = sum(data.values())
        if total <= 0:
            continue
        month = date(d.year, d.month, 1)
        by_month.setdefault(month, []).append({k: v / total for k, v in data.items()})
    out: dict[date, dict[str, float]] = {}
    for month, shares in by_month.items():
        acc: dict[str, float] = {}
        for share in shares:
            for k, v in share.items():
                acc[k] = acc.get(k, 0.0) + v
        n = len(shares)
        out[month] = {k: round(v / n * ref, 1) for k, v in acc.items()}
    return out


def _has_mixed_token(skill: str) -> bool:
    """Внутритокеновый микс скриптов ('cиcтeмнoe') - артефакт раскладки.

    Легитимный микс ('a/b тестирование', 'c++', '1c предприятие') идёт по
    разным токенам и не задевается. Однобуквенные токены игнорим.
    """
    import re
    import unicodedata

    for tok in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", skill or ""):
        if len(tok) <= 1:
            continue
        scripts = set()
        for ch in tok:
            if ch.isalpha():
                with contextlib.suppress(Exception):
                    scripts.add(unicodedata.name(ch).split()[0])
        if len(scripts) > 1:
            return True
    return False


def _is_blocked(name: str, blocked) -> bool:
    """Blacklist-семантика как в валидаторе: =xxx - точное, иначе подстрока."""
    low = (name or "").lower()
    for bad in blocked or ():
        b = str(bad).lower()
        if b.startswith("="):
            if low == b[1:]:
                return True
        elif b and b in low:
            return True
    return False


def merge_supplement_rows(
    rows: list[tuple[date, str, int]],
    file_skills: set[str],
    blocked=None,
) -> dict[date, Counter]:
    """Схлопнуть сырые DB-навыки для supplement: чистка + fold-merge.

    - пустоты отбрасываются;
    - уже покрытое файлами (по fold-ключу) пропускается - без двойного счёта;
    - гомоглиф-дубли ('aнaлиз дaнныx' + 'анализ данных') сливаются, дисплей -
      самое частое ЧИСТОЕ написание;
    - fold-группы без чистого написания (остаточный мусор) дропаются целиком;
    - blocked (blacklist справочника): вендоры и мусор, вычищенные из файлов,
      не втаскиваются обратно через DB.
    Возвращает {month: Counter[display_name]}.
    """
    from src.analyzers.skill_matcher import fold_script

    file_folds = {fold_script(k.lower()) for k in file_skills}
    per_month: dict[date, Counter] = {}
    votes: dict[str, Counter] = {}
    for m, raw, freq in rows:
        s = (raw or "").strip()
        if not s:
            continue
        fk = fold_script(s.lower())
        if not fk.strip() or fk in file_folds:
            continue
        if _is_blocked(s, blocked):
            continue
        per_month.setdefault(m, Counter())[fk] += int(freq)
        votes.setdefault(fk, Counter())[s] += int(freq)
    out: dict[date, Counter] = {}
    dropped_groups = 0
    dropped_freq = 0
    for m, counter in per_month.items():
        for fk, c in counter.items():
            clean = [sp for sp in votes[fk] if not _has_mixed_token(sp)]
            if not clean:
                dropped_groups += 1
                dropped_freq += c
                logger.debug("supplement_junk_dropped", fold_key=fk, freq=c)
                continue
            display = max(clean, key=lambda sp: (votes[fk][sp], -len(sp), sp))
            out.setdefault(m, Counter())[display] += c
    if dropped_groups:
        logger.info("supplement_junk_summary", groups=dropped_groups, freq=dropped_freq)
    return out


async def _load_trend_snapshots(session: AsyncSession) -> list[tuple[date, dict[str, float]]]:
    """Снимки рынка из trend_snapshots (source='hh_vacancies').

    Таблица - канонический стор снимков: её же читает вкладка «Тренды», и в ней
    вдесятеро больше дат, чем в data/history/freq_market_*.json. Файловые снимки
    там тоже лежат (импортируются из freq_*.json), поэтому источник избыточен,
    а не конфликтен - он лишь продлевает ряд.
    """
    rows = await session.execute(text("""
        SELECT snapshot_date, skill_freq
        FROM trend_snapshots
        WHERE source = 'hh_vacancies' AND skill_freq IS NOT NULL
        ORDER BY snapshot_date
    """))
    out: list[tuple[date, dict[str, float]]] = []
    for row in rows:
        raw = row.skill_freq
        if not isinstance(raw, dict):
            continue
        d = row.snapshot_date
        d = d if isinstance(d, date) else d.date()
        clean: dict[str, float] = {}
        for key, value in raw.items():
            try:
                freq = float(value)
            except (TypeError, ValueError):
                continue
            name = (key or "").strip()
            if name and freq > 0:
                clean[name] = freq
        if clean:
            out.append((d, clean))
    return out


async def load_time_series(session: AsyncSession) -> Result[list[Snapshot], DomainError]:
    """Build monthly skill-frequency snapshots from freq_market_*.json files,
    supplemented by parsed_skills from DB for skills not in those files.

    Each snapshot = per-month frequency (absolute count, not running total).
    DB-добавка нормализована к объёму последнего месяца (см. anchor_monthly_counts):
    иначе рост коллекции выдаётся за рост спроса. У файловых снимков объёмы
    неизвестны (vacancy_count=None), их правим только при наличии метаданных.
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
    else:
        file_snapshots = drop_broken_snapshots(file_snapshots)
        # Союз пересчитываем по уцелевшим: иначе навыки из битых снимков
        # блокируют DB-добавку (считаются "покрытыми файлами").
        all_skills_in_files = set()
        for _, data in file_snapshots:
            all_skills_in_files.update(data.keys())

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
              AND NULLIF(TRIM(BOTH FROM ps::text), '') IS NOT NULL
            GROUP BY month, ps::text
            ORDER BY month
        """))
        db_rows: list[tuple[date, str, int]] = []
        for row in rows:
            m = row.month if isinstance(row.month, date) else row.month.date()
            db_rows.append((m, row.skill, int(row.freq)))
        blocked: list = []
        try:
            from src import config as _cfg

            bl_path = _cfg.SKILL_BLACKLIST_PATH
            if bl_path.exists():
                import json as _json

                loaded = _json.loads(bl_path.read_text(encoding="utf-8"))
                blocked = loaded if isinstance(loaded, list) else list(loaded.keys())
        except Exception:
            logger.debug("supplement_blacklist_unavailable")
        db_monthly = merge_supplement_rows(db_rows, all_skills_in_files, blocked=blocked)

        totals_rows = await session.execute(text("""
            SELECT
                date_trunc('month', published_at::timestamp)::date AS month,
                COUNT(*) AS total
            FROM vacancies
            WHERE parsed_skills IS NOT NULL
              AND parsed_skills::text != '[]'
              AND published_at IS NOT NULL
            GROUP BY month
            ORDER BY month
        """))
        totals: dict[date, int] = {}
        for row in totals_rows:
            m = row.month if isinstance(row.month, date) else row.month.date()
            totals[m] = int(row.total or 0)

        # Convert DB data into snapshot format (объём-нормализованные,
        # см. anchor_monthly_counts - иначе рост коллекции = "рост спроса")
        for m, data in anchor_monthly_counts(db_monthly, totals).items():
            file_snapshots.append((m, data))
    except Exception:
        logger.warning("db_supplement_failed")

    if not file_snapshots:
        return Err(DomainError("No snapshot data available"))

    # 4. Канонический стор снимков (trend_snapshots) как основа ряда. В
    # data/history всего два файла freq_market, поэтому их навыки (sql, python,
    # linux...) имели две точки и выпадали из прогноза как insufficient_data.
    # Снимки дают им полную месячную историю; значения файлов/DB на своих
    # месяцах остаются приоритетными (снимки их и содержат).
    ts_snapshots = await _load_trend_snapshots(session)
    if ts_snapshots:
        merged: dict[date, dict[str, float]] = {
            m: dict(v) for m, v in anchor_snapshot_shares(ts_snapshots).items()
        }
        for d, data in file_snapshots:
            merged.setdefault(date(d.year, d.month, 1), {}).update(data)
        file_snapshots = sorted(merged.items())
        logger.info(
            "forecast_history_merged",
            months=len(file_snapshots),
            snapshot_days=len(ts_snapshots),
        )

    # 5. Sort by date
    file_snapshots.sort(key=lambda x: x[0])
    file_snapshots = _interpolate_missing_months(file_snapshots)
    return Ok([Snapshot(m, data) for m, data in file_snapshots])


def _interpolate_missing_months(snapshots: list[tuple[date, dict[str, float]]]) -> list[tuple[date, dict[str, float]]]:
    """Заполняет пропущенные календарные месяцы линейной интерполяцией.

    Нерегулярные снимки (напр. апр, май, июн, авг) дают Prophet'у разрозненные
    точки без июля - интерполяция восстанавливает ежемесячный ряд.
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


MIN_HISTORY_POINTS = 6
def has_enough_history(n_points: int, minimum: int = MIN_HISTORY_POINTS) -> bool:
    """Достаточно ли согласованных точек для прогноза (< 6 -> insufficient_data)."""
    return n_points >= minimum


class ProphetForecastEngine(BasePredictor):
    """Forecast engine using Prophet for skills with >= MIN_HISTORY_FOR_PROPHET
    history points and actual frequency >= MIN_FREQ, falling back to
    SkillForecastEngine (trend/insufficient) otherwise."""

    MIN_FREQ = 10
    # Fewer than 6 monthly points makes Prophet unstable (wild extrapolation:
    # see the "unreliable +200%" reports). Such skills are served by the trend
    # fallback and surfaced as insufficient, not by a fitted Prophet model.
    MIN_HISTORY_FOR_PROPHET = 6
    MAX_GROWTH_CAP = 2.0
    # Top-prediction display: only show skills with meaningful frequency
    TOP_DISPLAY_MIN_FREQ = 50
    # Потолок числа Prophet-моделей: фит тяжёлый (~секунды на модель), а топа на
    # показ нужно десятки. Остальные навыки обслуживает trend-fallback.
    MAX_PROPHET_MODELS = 60
    # Воркеры синхронного fit (тесты/CLI). В API используется fit_async: он
    # фитит по одной модели, чтобы не держать event loop.
    FIT_WORKERS = 4

    def __init__(self):
        self._models: dict[str, Prophet] = {}
        self._fallback_engine: SkillForecastEngine | None = None
        self._skill_history: dict[str, list[tuple[date, float]]] = {}
        self._last_actual_freq: dict[str, float] = {}
        self._skill_mape: dict[str, float | None] = {}
        self._skill_npoints: dict[str, int] = {}
        self._observed: dict[str, list[tuple[date, float]]] = {}
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
        canon: dict[str, str] = {}  # lower -> kept original spelling
        for snap in snapshots:
            for skill, freq in snap.frequencies.items():
                name = (skill or "").strip()
                if not name:
                    continue  # мусор парсинга (пустые parsed_skills из БД)
                key = canon.setdefault(name.lower(), name)
                history.setdefault(key, []).append((snap.date, freq))
        for skill, pts in history.items():
            by_date: dict[date, float] = {}
            for d, f in pts:
                by_date[d] = by_date.get(d, 0.0) + float(f)
            history[skill] = sorted(by_date.items())
        return history

    def _fit_prophet_for_skill(self, skill: str, points: list[tuple[date, float]]):
        import numpy as np
        from cmdstanpy.utils.logging import disable_logging
        df = pd.DataFrame({"ds": [p[0] for p in points], "y": [p[1] for p in points]})
        n_points = len(points)
        # Sanity check: detect extreme variance that causes "inf in matrix" errors
        y = df["y"].values
        if np.any(~np.isfinite(y)) or (y.max() - y.min()) > 1e6:
            raise ValueError(f"Unstable data for Prophet: min={y.min()}, max={y.max()}, n={n_points}")
        model = Prophet(
            yearly_seasonality=n_points >= YEARLY_SEASONALITY_MIN_POINTS,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.80,
            changepoint_prior_scale=0.05 if n_points < 6 else (0.5 if n_points < 12 else 0.05),
        )
        with disable_logging():
            # 300 итераций хватает для коротких месячных рядов; 1000 растягивал
            # каждый фит и на это время отбирал GIL у event loop сервера.
            model.fit(df, iter=300)
        return model

    def _prepare_fit(
        self,
        snapshots: list[Snapshot],
        fallback_freqs: dict[str, float] | None,
    ) -> list[tuple[str, list[tuple[date, float]]]]:
        """Общая часть fit/fit_async: истории, раскладка кандидатов, fallback."""
        import logging
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)

        self._n_snapshots = len(snapshots)
        history = self._gather_history(snapshots)
        # Полные наблюдаемые истории - для вкладки измеренных падений
        # (/forecast/observed): факты по снимкам, без экстраполяции.
        self._observed = {s: list(pts) for s, pts in history.items()}

        # Separate skills by data depth: Prophet (≥3 pts) vs trend (fallback)
        prophet_candidates: list[tuple[str, list[tuple[date, float]]]] = []
        for skill, points in history.items():
            last_actual = points[-1][1]
            self._last_actual_freq[skill] = last_actual
            self._skill_npoints[skill] = len(points)
            if len(points) >= 5:
                self._skill_mape[skill] = _holdout_mape(points)
            if len(points) >= self.MIN_HISTORY_FOR_PROPHET and last_actual >= self.MIN_FREQ:
                prophet_candidates.append((skill, points))
            else:
                self._skill_history[skill] = points

        # Дорогой фит ограничиваем топом по последней частоте: с полной историей
        # снимков кандидатов сотни, и без потолка старт висел бы часами. Остальные
        # получают дешёвый прогноз через trend-fallback-движок.
        if len(prophet_candidates) > self.MAX_PROPHET_MODELS:
            prophet_candidates.sort(
                key=lambda c: self._last_actual_freq.get(c[0], 0.0), reverse=True
            )
            for skill, points in prophet_candidates[self.MAX_PROPHET_MODELS:]:
                self._skill_history[skill] = points
            prophet_candidates = prophet_candidates[: self.MAX_PROPHET_MODELS]
            logger.info(
                "prophet_candidates_capped",
                cap=self.MAX_PROPHET_MODELS,
                fallback_skills=len(self._skill_history),
            )

        if fallback_freqs:
            gen = SkillForecastEngine()
            # Тот же ряд, что у Prophet (файлы + БД + trend_snapshots): иначе
            # трендовый fallback строился по двум файлам и выдавал экстраполяции
            # в потолок +150%.
            match gen.fit(fallback_freqs, snapshots=[(s.date, s.frequencies) for s in snapshots]):
                case Ok(_):
                    self._fallback_engine = gen
                case Err(e):
                    logger.warning("prophet_fallback_engine_fit_failed", error=str(e))

        return prophet_candidates

    def _finalize_fit(self) -> Result[ProphetForecastEngine, DomainError]:
        logger.info(
            "prophet_fitted",
            prophet_skills=len(self._models),
            fallback_skills=len(self._skill_history),
            snapshots=self._n_snapshots,
        )
        if not self._models and not self._fallback_engine:
            FIT_STATUS.update(state="failed")
            return Err(DomainError("No skills could be fitted by Prophet or fallback"))
        self._is_fitted = True
        FIT_STATUS.update(state="ready")
        return Ok(self)

    def fit(
        self,
        snapshots: list[Snapshot],
        fallback_freqs: dict[str, float] | None = None,
    ) -> Result[ProphetForecastEngine, DomainError]:
        if not snapshots:
            FIT_STATUS.update(state="failed")
            return Err(DomainError("No snapshots provided to Prophet engine"))

        prophet_candidates = self._prepare_fit(snapshots, fallback_freqs)

        if prophet_candidates:
            import time as _time

            FIT_STATUS.update(state="fitting", done=0, total=len(prophet_candidates),
                              started_at=_time.monotonic())
            with ThreadPoolExecutor(max_workers=self.FIT_WORKERS) as pool:
                futures = {pool.submit(self._fit_prophet_for_skill, s, p): s for s, p in prophet_candidates}
                for future in as_completed(futures):
                    skill = futures[future]
                    try:
                        self._models[skill] = future.result()
                    except Exception as e:
                        logger.warning("prophet_skill_fit_failed", skill=skill, error=str(e))
                    FIT_STATUS["done"] = FIT_STATUS.get("done", 0) + 1

        return self._finalize_fit()

    async def fit_async(
        self,
        snapshots: list[Snapshot],
        fallback_freqs: dict[str, float] | None = None,
    ) -> Result[ProphetForecastEngine, DomainError]:
        """Как fit(), но модели фитятся по одной с возвратом управления loop.

        Параллельный фит держит GIL и роняет отзывчивость API на всё время
        работы (в замерах /health таймаутил ~13 с). Здесь между моделями
        отдаём управление - health отвечает, фит идёт фоном.
        """
        if not snapshots:
            FIT_STATUS.update(state="failed")
            return Err(DomainError("No snapshots provided to Prophet engine"))

        prophet_candidates = self._prepare_fit(snapshots, fallback_freqs)

        if prophet_candidates:
            import time as _time

            FIT_STATUS.update(state="fitting", done=0, total=len(prophet_candidates),
                              started_at=_time.monotonic())
            for skill, points in prophet_candidates:
                try:
                    self._models[skill] = await asyncio.to_thread(
                        self._fit_prophet_for_skill, skill, points
                    )
                except Exception as e:
                    logger.warning("prophet_skill_fit_failed", skill=skill, error=str(e))
                FIT_STATUS["done"] = FIT_STATUS.get("done", 0) + 1
                await asyncio.sleep(0.01)

        return self._finalize_fit()

    def predict(self, skill: str, months: int = 12) -> Result[ForecastResult, DomainError]:
        if months < 1 or months > 60:
            return Err(DomainError(f"months must be 1-60, got {months}"))

        if skill in self._models:
            n_pts = self._skill_npoints.get(skill, 0)
            if not has_enough_history(n_pts):
                # Слишком короткая история -> явный статус, а не дикие проценты.
                last_actual = self._last_actual_freq.get(skill, 0.0)
                return Ok(ForecastResult(
                    skill=skill,
                    current_frequency=round(last_actual, 4),
                    predicted_growth=0.0,
                    confidence=0.0,
                    next_year_frequency=round(last_actual, 4),
                    engine_used="insufficient_data",
                    data_points=n_pts,
                    mape=None,
                    forecast_months=0,
                ))
            model = self._models[skill]
            # Limit forecast horizon based on data points, but be more generous:
            # 3 pts -> 3m, 6 pts -> 6m, 12+ pts -> 12m (was n_pts//2 - too conservative).
            max_months = max(1, min(n_pts, 12))
            if months > max_months:
                logger.warning(
                    "forecast_horizon_truncated",
                    skill=skill, requested=months, effective=max_months,
                )
                months = max_months
            future = model.make_future_dataframe(periods=months, freq="ME")
            forecast = model.predict(future)
            last_row = forecast.iloc[-1]
            next_freq = max(float(last_row["yhat"]), 0.0)

            last_actual = self._last_actual_freq.get(skill, 0.0)
            baseline = max(last_actual, self.MIN_FREQ)
            growth = (next_freq - baseline) / baseline
            # Soft, data-driven regularization instead of a fixed +/-200% clip:
            # fewer months of history -> pull the forecast toward flat, and the
            # safety cap grows with the number of observed points.
            growth *= min(1.0, n_pts / 12.0)
            _soft_cap = 1.0 + 0.25 * n_pts
            growth = max(min(growth, _soft_cap), -_soft_cap)

            uncertainty = float(last_row["yhat_upper"] - last_row["yhat_lower"])
            conf = max(0.0, 1.0 - min(uncertainty / max(next_freq, 1.0), 0.85))
            # Backtest (hold-out MAPE): worse historical error -> lower trust.
            _mape = self._skill_mape.get(skill)
            if _mape is not None:
                conf *= max(0.0, 1.0 - min(_mape, 1.0))
            # F6 (2026-10-14): несём реальные границы интервала наружу
            # (раньше использовались только для conf и выбрасывались).
            unc_lower = max(float(last_row["yhat_lower"]), 0.0)
            unc_upper = float(last_row["yhat_upper"])
            return Ok(ForecastResult(
                skill=skill,
                current_frequency=round(last_actual, 4),
                predicted_growth=round(growth, 4),
                confidence=round(max(conf, 0.0), 4),
                next_year_frequency=round(next_freq, 4),
                engine_used="prophet",
                data_points=self._skill_npoints.get(skill, n_pts),
                mape=_rounded_mape(self._skill_mape, skill),
                forecast_months=months,
                uncertainty_lower=round(unc_lower, 4),
                uncertainty_upper=round(unc_upper, 4),
            ))
        if self._fallback_engine:
            result = self._fallback_engine.forecast(skill, min(months, self.max_forecast_months()))
            if result.is_ok():
                fr = result.unwrap()
                n_pts = self._skill_npoints.get(skill, 0)
                # Явный статус: недостаточно данных для прогноза
                if not has_enough_history(n_pts):
                    return Ok(ForecastResult(
                        skill=fr.skill,
                        current_frequency=fr.current_frequency,
                        predicted_growth=0.0,
                        confidence=0.0,
                        next_year_frequency=fr.current_frequency,
                        engine_used="insufficient_data",
                        data_points=n_pts,
                        mape=None,
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
                    mape=_rounded_mape(self._skill_mape, skill),
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
                        if any(ex.skill == r.skill for ex in results):
                            continue
                        # Единый контракт: fallback-результат оборачиваем как в predict(),
                        # а не отдаем сырым (иначе insufficient_data выглядит валидным трендом).
                        n_pts = self._skill_npoints.get(r.skill, r.data_points)
                        # Тот же порог, что в predict(): иначе forecast_all отдаёт
                        # «тренд» там, где одиночный predict уже insufficient_data.
                        if not has_enough_history(n_pts):
                            results.append(ForecastResult(
                                skill=r.skill,
                                current_frequency=r.current_frequency,
                                predicted_growth=0.0,
                                confidence=0.0,
                                next_year_frequency=r.current_frequency,
                                engine_used="insufficient_data",
                                data_points=n_pts,
                                mape=None,
                                forecast_months=0,
                            ))
                        else:
                            results.append(ForecastResult(
                                skill=r.skill,
                                current_frequency=r.current_frequency,
                                predicted_growth=r.predicted_growth,
                                confidence=r.confidence,
                                next_year_frequency=r.next_year_frequency,
                            engine_used="trend",
                            data_points=n_pts,
                            mape=_rounded_mape(self._skill_mape, r.skill),
                            forecast_months=min(months, self.max_forecast_months()),
                        ))
                case _:
                    pass
        return Ok(results)

    def top_growing(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [
                    r for r in results if r.current_frequency >= self.TOP_DISPLAY_MIN_FREQ
                    and r.next_year_frequency > 0 and r.predicted_growth > 0
                ]
                # Exclude unreliable predictions: growth > 200% with confidence < 30%
                results = [r for r in results if not (r.predicted_growth > 2.0 and r.confidence < 0.3)]
                if not results:
                    return Ok([])
                # Строго по росту: вкладка называется "Растущие", частота уже
                # отгейчена порогами (MIN_FREQ / TOP_DISPLAY_MIN_FREQ / min_freq).
                # Композит 0.3*рост + 0.7*частота ставил jira +0.6% выше c++ +38%
                # и врал про смысл вкладки (как и trend-движок: только рост).
                results.sort(key=lambda x: x.predicted_growth, reverse=True)
                return Ok(results[:n])
            case Err(e):
                return Err(e)

    def top_declining(self, n: int = 10, months: int = 12) -> Result[list[ForecastResult], DomainError]:
        match self.forecast_all(months):
            case Ok(results):
                results = [
                    r for r in results if r.current_frequency >= self.TOP_DISPLAY_MIN_FREQ
                    and r.next_year_frequency > 0 and r.predicted_growth < 0
                ]
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
