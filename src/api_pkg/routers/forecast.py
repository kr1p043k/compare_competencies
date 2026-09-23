import asyncio
import re
from datetime import date
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text

from src import Ok, Err, Result, config, DomainError
from src.predictors.prophet_forecast import ProphetForecastEngine
from src.predictors.skill_forecast import SkillForecastEngine, ForecastResult
from src.api_pkg.routers.auth import user_error_detail
from src.utils import safe_read_json
import src.api_pkg.deps as deps

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["forecast"])
limiter = Limiter(key_func=get_remote_address)


def _engine_info(engine) -> dict:
    """Чем посчитано: имя движка + состояние Prophet (диагностика 'почему genetic').

    genetic на все запросы = Prophet не дофитился на старте (долгий fit сотен
    моделей) или упал — это видно здесь, а не гадается по бейджам.
    """
    prophet_fitted = bool(
        getattr(deps, "prophet_engine", None) is not None
        and getattr(deps.prophet_engine, "is_fitted", False)
    )
    if isinstance(engine, ProphetForecastEngine):
        return {
            "engine": "prophet",
            "prophet_fitted": prophet_fitted,
            "prophet_models": len(getattr(engine, "_models", {}) or {}),
        }
    return {"engine": "genetic", "prophet_fitted": prophet_fitted, "prophet_models": 0}

# Кэш топов: fit происходит на старте/пайплайне, а compute — сотни Prophet-
# predict'ов (десятки секунд GIL). Без кэша каждое переключение горизонта
# в UI вешает event loop и роняет health-checks.
_TOP_CACHE: dict[tuple, tuple[float, tuple]] = {}
_TOP_CACHE_TTL = 300.0
_TOP_CACHE_MAX = 64


def _top_cache_get(key: tuple):
    import time as _time

    entry = _TOP_CACHE.get(key)
    if entry and _time.monotonic() - entry[0] < _TOP_CACHE_TTL:
        return entry[1]
    return None


def _top_cache_set(key: tuple, value: tuple) -> None:
    import time as _time

    if len(_TOP_CACHE) >= _TOP_CACHE_MAX:
        oldest = min(_TOP_CACHE, key=lambda k: _TOP_CACHE[k][0])
        _TOP_CACHE.pop(oldest, None)
    _TOP_CACHE[key] = (_time.monotonic(), value)


def _compute_top(engine, direction: str, months: int, want: int, sort: str):
    """Синхронный подсчёт топа (стартует в to_thread — НЕ в event loop).

    Возвращает (Result[list[ForecastResult]], method).
    """
    if isinstance(engine, ProphetForecastEngine):
        if direction == "declining":
            return engine.top_declining(n=want, months=months), "prophet"
        return engine.top_growing(n=want, months=months), "prophet"
    res = engine.top_growing(n=want, months=months)
    if res.is_ok() and direction == "declining" and sort == "growth":
        all_results = sorted(res.unwrap(), key=lambda x: x.predicted_growth)
        return Ok(all_results), "genetic"
    return res, "genetic"


def _get_forecast_data() -> dict[str, float]:
    data: dict[str, float] = {}
    freq_path = config.COMPETENCY_FREQ_PATH
    if freq_path.exists():
        raw = safe_read_json(freq_path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    data[k] = float(v)
    weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
    if weights_path.exists():
        raw = safe_read_json(weights_path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k not in data and isinstance(v, (int, float)):
                    data[k] = float(v)
    return data


async def _get_forecast_engine() -> Result[ProphetForecastEngine | SkillForecastEngine, DomainError]:
    if deps.prophet_engine is not None and deps.prophet_engine.is_fitted:
        return Ok(deps.prophet_engine)
    if deps.skill_engine is not None:
        return Ok(deps.skill_engine)
    freqs = _get_forecast_data()
    if not freqs:
        return Err(DomainError("No frequency data available for forecast"))
    engine = SkillForecastEngine()
    result = await asyncio.to_thread(engine.fit, freqs)
    if isinstance(result, Err):
        return result
    deps.skill_engine = engine
    return Ok(engine)


def _get_forecast_data() -> dict[str, float]:
    data: dict[str, float] = {}
    freq_path = config.COMPETENCY_FREQ_PATH
    if freq_path.exists():
        raw = safe_read_json(freq_path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    data[k] = float(v)
    weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
    if weights_path.exists():
        raw = safe_read_json(weights_path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k not in data and isinstance(v, (int, float)):
                    data[k] = float(v)
    return data


async def _get_vacancy_meta() -> dict:
    meta = {"vacancies_count": 0, "data_from": None, "data_to": date.today().isoformat(),
            "snapshots_count": 0}
    try:
        from src.database import async_session_factory

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    text("""
                        SELECT
                            COUNT(*) AS cnt,
                            MIN(published_at::date) AS min_date
                        FROM vacancies
                        WHERE parsed_skills IS NOT NULL
                          AND parsed_skills != '[]'::jsonb
                    """)
                )
            ).one()
            if row.cnt:
                meta["vacancies_count"] = row.cnt
                meta["data_from"] = row.min_date.isoformat() if row.min_date else None
            try:
                snap_cnt = await session.execute(text("SELECT COUNT(*) FROM trend_snapshots"))
                meta["snapshots_count"] = snap_cnt.scalar() or 0
            except Exception:
                pass
            if row.cnt:
                return meta
    except Exception:
        logger.warning("vacancy_meta_db_failed_falling_back_to_files")
    freq_path = config.COMPETENCY_FREQ_PATH
    if freq_path.exists():
        raw = safe_read_json(freq_path)
        if isinstance(raw, dict):
            meta["vacancies_count"] = len(raw)
    history_dir: Path = config.DATA_DIR / "history"
    if history_dir.is_dir():
        snaps = sorted(history_dir.glob("freq_*.json"))
        if snaps:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", snaps[-1].stem)
            if m:
                meta["data_from"] = m.group(1)
    return meta


def _serialize(r: ForecastResult, direction: str | None = None, method: str = "genetic") -> dict:
    change_pct = round(r.predicted_growth * 100, 2)
    return {
        "skill": r.skill,
        "current_frequency": round(r.current_frequency, 4),
        "predicted_growth": round(r.predicted_growth, 4),
        "predicted_change_pct": change_pct,
        "confidence": round(r.confidence, 4),
        "next_year_frequency": round(r.next_year_frequency, 4),
        "method": method,
        "engine_used": getattr(r, "engine_used", method),
        "data_points": getattr(r, "data_points", 0),
        "mape": (
            round(r.mape, 4) if getattr(r, "mape", None) is not None else None
        ),
        "forecast_months": getattr(r, "forecast_months", 0),
        "trend_direction": direction or ("growing" if r.predicted_growth > 0 else "declining"),
    }


def _detect_method(engine: ProphetForecastEngine | SkillForecastEngine, skill: str | None = None) -> str:
    if isinstance(engine, ProphetForecastEngine):
        if skill is not None:
            if skill in engine._models:
                return "prophet"
            # Fallback-движок внутри Prophet: определяем по результату
            return "trend"
        return "prophet" if engine._models else "trend"
    return "trend"


def _record_forecast_accuracy(engine, forecasts) -> None:
    try:
        from src.monitoring.metrics import forecast_accuracy

        def _mape_for(skill: str) -> float | None:
            model = getattr(engine, "_models", {}).get(skill)
            if isinstance(model, dict) and model.get("mape") is not None:
                return float(model["mape"])
            fallback = getattr(engine, "_fallback_engine", None)
            if fallback is not None:
                model = getattr(fallback, "_models", {}).get(skill)
                if isinstance(model, dict) and model.get("mape") is not None:
                    return float(model["mape"])
            return None

        for r in forecasts:
            mape = _mape_for(r.skill)
            if mape is not None:
                forecast_accuracy.labels(skill=r.skill).set(mape)
    except Exception:
        logger.debug("forecast_accuracy_record_failed")


def _is_usable_forecast(r, min_freq: float = 0) -> bool:
    """Годен ли прогноз для топа: есть горизонт (не flat/insufficient) и частота."""
    if getattr(r, "forecast_months", 0) == 0:
        return False
    if min_freq and r.current_frequency < min_freq:
        return False
    return True


def _select_observed_direction(items: list[dict], direction: str) -> list[dict]:
    """Знак измеренного изменения: declining — только падения, growing — только
    рост, all — всё. Ноль — ни то ни другое (плоский шум в топ не идёт)."""
    if direction == "declining":
        return [r for r in items if r["observed_change_pct"] < 0]
    if direction == "growing":
        return [r for r in items if r["observed_change_pct"] > 0]
    return list(items)


@router.get("/forecast/all")
@limiter.limit("30/minute")
async def get_all_forecasts(request: Request, months: int = Query(12, ge=1, le=24)):
    """Прогнозы по всем навыкам."""
    match await _get_forecast_engine():
        case Ok(engine):
            outcome = await asyncio.to_thread(engine.forecast_all, months)
            match outcome:
                case Ok(forecasts):
                    _record_forecast_accuracy(engine, forecasts)
                    method = _detect_method(engine)
                    items = [_serialize(r, method=method) for r in forecasts]
                    return {"total": len(items), "months": months, "forecasts": items, **_engine_info(engine)}
                case Err(e):
                    logger.warning("forecast_compute_failed", error=str(e))
                    raise HTTPException(
                        status_code=500,
                        detail=await user_error_detail(request, str(e), "Не удалось построить прогноз. Попробуйте позже."),
                    )
        case Err(e):
            logger.warning("forecast_engine_unavailable", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=await user_error_detail(request, str(e), "Сервис прогнозов временно недоступен. Попробуйте позже."),
            )


@router.get("/forecast/top")
@limiter.limit("30/minute")
async def get_top_forecasts(
    request: Request,
    n: int = Query(25, ge=1, le=50),
    months: int = Query(12, ge=1, le=24),
    direction: str = Query("growing", regex="^(growing|declining)$"),
    sort: str = Query("growth", regex="^(growth|popular)$"),
    min_freq: float = Query(50, ge=0),
):
    """Топ навыков: sort=growth (по росту, умолчанию) или popular (по текущей частоте).

    Строки без горизонта прогноза (forecast_months == 0 — flat/insufficient)
    из топа исключаются: это не прогнозы. min_freq=50 по умолчанию режет
    мелочь с накрученными процентами (порог осмысленной частоты).
    """
    match await _get_forecast_engine():
        case Ok(engine):
            meta = await _get_vacancy_meta()
            want = min(n * 2, 50)

            def _finalize(results, method: str):
                results = [r for r in results if _is_usable_forecast(r, min_freq)]
                if sort == "popular":
                    results = sorted(results, key=lambda x: x.current_frequency, reverse=True)
                results = results[:n]
                _record_forecast_accuracy(engine, results)
                return [_serialize(r, direction, method) for r in results]

            cache_key = ("top", direction, months, sort, min_freq, n, id(engine))
            cached = _top_cache_get(cache_key)
            if cached is not None:
                items = cached
            else:
                outcome, method = await asyncio.to_thread(
                    _compute_top, engine, direction, months, want, sort
                )
                match outcome:
                    case Ok(results):
                        items = _finalize(results, method)
                    case Err(e):
                        logger.warning("forecast_compute_failed", error=str(e))
                        raise HTTPException(
                            status_code=500,
                            detail=await user_error_detail(request, str(e), "Не удалось построить прогноз. Попробуйте позже."),
                        )
                _top_cache_set(cache_key, items)
            # Determine actual forecast horizon (Prophet caps it internally)
            actual_months = months
            if isinstance(engine, ProphetForecastEngine) and engine.is_fitted:
                actual_months = min(months, engine.max_forecast_months())
            return {"direction": direction, "n": n, "months": actual_months, "requested_months": months, "forecasts": items, **_engine_info(engine), **meta}
        case Err(e):
            logger.warning("forecast_engine_unavailable", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=await user_error_detail(request, str(e), "Сервис прогнозов временно недоступен. Попробуйте позже."),
            )


@router.get("/forecast/engine")
@limiter.limit("60/minute")
async def get_forecast_engine_status(request: Request):
    """Лёгкий статус движка + прогресс фонового фита Prophet (для countdown)."""
    import time as _time

    from src.predictors.prophet_forecast import FIT_STATUS

    status = dict(FIT_STATUS)
    elapsed = None
    eta_seconds = None
    if status.get("started_at") is not None:
        elapsed = _time.monotonic() - status["started_at"]
        done = status.get("done", 0)
        total = status.get("total", 0)
        if status.get("state") == "fitting" and done > 0 and total > done:
            eta_seconds = round(elapsed / done * (total - done))
    return {**_engine_info_for_status(), **status,
            "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
            "eta_seconds": eta_seconds}


def _engine_info_for_status() -> dict:
    engine = getattr(deps, "prophet_engine", None)
    if engine is not None and getattr(engine, "is_fitted", False):
        return _engine_info(engine)
    return {"engine": "genetic", "prophet_fitted": False, "prophet_models": 0}


@router.get("/forecast/observed")
@limiter.limit("30/minute")
async def get_observed_drops(
    request: Request,
    n: int = Query(25, ge=1, le=50),
    months: int = Query(3),
    sort: str = Query("growth", regex="^(growth|popular)$"),
    min_freq: float = Query(50, ge=0),
    direction: str = Query("declining", regex="^(declining|growing|all)$"),
):
    """Измеренное изменение спроса за окно — факт, не прогноз.

    months ∈ {1, 3, 6, 12}: нужно 2/3/6/12 снимков соответственно,
    иначе навык пропускается (в мете max_points — сколько есть максимум).
    direction режет по знаку: declining — только падения (< 0),
    growing — только рост (> 0), all — всё подряд. Ноль — ни то ни другое.
    """
    from src.predictors.skill_forecast import OBSERVED_MONTHS, compute_observed_drops

    if months not in OBSERVED_MONTHS:
        raise HTTPException(
            status_code=400,
            detail=f"months must be one of {list(OBSERVED_MONTHS)}",
        )
    match await _get_forecast_engine():
        case Ok(engine):
            observed = dict(getattr(engine, "_observed", {}) or {})
            max_points = max((len(v) for v in observed.values()), default=0)
            items = await asyncio.to_thread(
                compute_observed_drops, observed, months, min_freq
            )
            items = _select_observed_direction(items, direction)
            if sort == "popular":
                items = sorted(items, key=lambda x: x["last_frequency"], reverse=True)
            return {
                "direction": direction, "n": n, "months": months,
                "required_points": {1: 2, 3: 3, 6: 6, 12: 12}[months],
                "max_points": max_points,
                "forecasts": items[:n], **_engine_info(engine),
            }
        case Err(e):
            logger.warning("forecast_engine_unavailable", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=await user_error_detail(request, str(e), "Сервис прогнозов временно недоступен. Попробуйте позже."),
            )


@router.get("/forecast/popular")
@limiter.limit("30/minute")
async def get_popular_forecasts(
    request: Request,
    n: int = Query(25, ge=1, le=50),
    months: int = Query(12, ge=1, le=24),
):
    """Прогнозы популярных навыков."""
    match await _get_forecast_engine():
        case Ok(engine):
            meta = await _get_vacancy_meta()
            if isinstance(engine, ProphetForecastEngine):
                outcome = await asyncio.to_thread(engine.top_popular, n=n, months=months)
                match outcome:
                    case Ok(results):
                        results = [r for r in results if _is_usable_forecast(r)][:n]
                        _record_forecast_accuracy(engine, results)
                        items = [_serialize(r, "growing", "prophet") for r in results]
                    case Err(e):
                        logger.warning("forecast_compute_failed", error=str(e))
                        raise HTTPException(
                            status_code=500,
                            detail=await user_error_detail(request, str(e), "Не удалось построить прогноз. Попробуйте позже."),
                        )
            else:
                all_results = await asyncio.to_thread(engine.forecast_all, months)
                all_results = sorted(all_results.unwrap_or([]), key=lambda x: x.current_frequency, reverse=True)
                all_results = [r for r in all_results if _is_usable_forecast(r)][:n]
                _record_forecast_accuracy(engine, all_results)
                items = [_serialize(r, "growing", "genetic") for r in all_results]
            return {"direction": "popular", "n": n, "months": months, "forecasts": items, **_engine_info(engine), **meta}
        case Err(e):
            logger.warning("forecast_engine_unavailable", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=await user_error_detail(request, str(e), "Сервис прогнозов временно недоступен. Попробуйте позже."),
            )


@router.get("/forecast/{skill}")
@limiter.limit("60/minute")
async def get_skill_forecast(skill: str, request: Request, months: int = Query(12, ge=1, le=24)):
    """Прогноз по одному навыку."""
    match await _get_forecast_engine():
        case Ok(engine):
            if hasattr(engine, "forecast"):
                result = await asyncio.to_thread(engine.forecast, skill, months)
            else:
                result = await asyncio.to_thread(engine.predict, skill, months)
            match result:
                case Ok(r):
                    _record_forecast_accuracy(engine, [r])
                    return _serialize(r, method=_detect_method(engine, skill))
                case Err(e):
                    logger.warning("forecast_skill_failed", skill=skill, error=str(e))
                    raise HTTPException(status_code=404, detail=f"Навык не найден: {skill}")
        case Err(e):
            logger.warning("forecast_engine_unavailable", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=await user_error_detail(request, str(e), "Сервис прогнозов временно недоступен. Попробуйте позже."),
            )
