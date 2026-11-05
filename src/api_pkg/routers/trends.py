"""Trends analysis - per-competency aggregate + per-skill trends from trend_snapshots."""
from __future__ import annotations

import asyncio
import json
import statistics
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select

from src import Err, Ok
from src.analyzers.skills.trends import TrendAnalyzer
from src.api_pkg import deps
from src.api_pkg.routers.auth import require_authenticated, user_error_detail
from src.database import async_session_factory
from src.db import get_pool
from src.models.api_responses import TrendsResponse
from src.models.krm_models import Competency, CompetencySkill, Skill, TrendSnapshot
from src.utils import load_competency_mapping, load_inverted_skill_index, skill_words

logger = structlog.get_logger("api")

router = APIRouter(tags=["trends"], dependencies=[Depends(require_authenticated())])
limiter = Limiter(key_func=get_remote_address)


@router.get("/trends", response_model=TrendsResponse)
@limiter.limit("60/minute")
async def get_trends(
    request: Request,
    top_n: int = Query(15),
    min_change: float = Query(3.0),
    trend_analyzer_instance: TrendAnalyzer = Depends(deps.get_trend_analyzer),  # noqa: B008
):
    """Тренды навыков."""
    try:
        result = trend_analyzer_instance.get_trending_skills(
            top_n=top_n, min_change_percent=min_change
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("trends_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=await user_error_detail(request, str(e), "Не удалось построить тренды. Попробуйте позже."),
        ) from None
    match result:
        case Ok(trends):
            return {"trends": trends}
        case Err(err):
            logger.warning("trends_failed", error=str(err))
            raise HTTPException(
                status_code=500,
                detail=await user_error_detail(request, str(err), "Не удалось построить тренды. Попробуйте позже."),
            )


def _classify(change_pct: float) -> str:
    if change_pct > 5:
        return "rising"
    if change_pct < -5:
        return "falling"
    return "stable"


def _month_key(date_str: str) -> tuple[int, int]:
    """'2026-11-01' -> (2026, 11): ключ месячного бакета для истории."""
    try:
        return int(date_str[:4]), int(date_str[5:7])
    except (ValueError, IndexError):
        return 0, 0


@lru_cache(maxsize=1)
def _get_mappings():
    mapping = load_competency_mapping()
    inverted = load_inverted_skill_index()
    return mapping, inverted


@lru_cache(maxsize=200_000)
def _sw(name: str) -> frozenset[str]:
    """Кэш token-множества навыка.

    skill_words вызывается миллионы раз в двойных циклах /competency-trends
    (O(навыки x навыки) + O(limit x снимки)); без кэша эндпоинт висел минутами.
    """
    return frozenset(skill_words(name))


# /competency-trends считается десятки секунд (CPU): кэшируем результат, чтобы
# повторные вызовы (переключение фильтра в UI) не пересчитывали.
_CT_CACHE: dict[tuple, tuple[float, dict]] = {}
_CT_CACHE_TTL = 600.0

# Таблица trend_precomputed (asyncpg-пул src/db, см. миграцию
# alembic/versions/trend_precomputed.py). Полезная нагрузка /competency-trends,
# чтобы результат переживал рестарт процесса и делился между воркерами.
_CT_TABLE = "trend_precomputed"

# Срок жизни записи БД-кэша. Первичный механизм инвалидации - шина cache_bus
# (запись нового снимка чистит таблицу), TTL - страховка на случай, если событие
# инвалидации потерялось (например, запись была сделана до поднятия пула).
_CT_DB_TTL = 24 * 3600.0


def _cache_key(direction: str | None, limit: int) -> str:
    """Строковый ключ строки таблицы trend_precomputed: (direction, limit)."""
    return f"{direction or ''}|{limit}"


async def _read_db_cache(key: str) -> dict | None:
    """Прочитать payload из БД-кэша; None при промахе/порче/устаревании.

    Любая ошибка (пула нет, таблицы нет, битый payload) - это промах, а не
    исключение: эндпоинт обязан просто пересчитать и записать заново.
    """
    try:
        pool = get_pool()
    except RuntimeError:
        return None
    try:
        row = await pool.fetchrow(
            f"SELECT payload, computed_at FROM {_CT_TABLE} WHERE cache_key = $1", key
        )
    except Exception as exc:  # таблицы может не быть до миграции
        logger.warning("ct_cache_read_failed", key=key, error=str(exc))
        return None
    if row is None:
        return None
    computed_at = row["computed_at"]
    if computed_at is not None:
        try:
            age = (datetime.now(UTC) - computed_at).total_seconds()
        except (TypeError, ValueError):  # наивный datetime - считаем свежим
            age = 0.0
        if age > _CT_DB_TTL:
            logger.info("ct_cache_stale", key=key, age=age)
            return None
    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError) as exc:
        logger.warning("ct_cache_payload_corrupt", key=key, error=str(exc))
        return None
    if not isinstance(payload, dict) or "trends" not in payload:
        logger.warning("ct_cache_payload_wrong_shape", key=key)
        return None
    return payload


async def _write_db_cache(key: str, payload: dict) -> None:
    """Записать (или обновить) payload в таблицу БД-кэша. Ошибка - только лог."""
    try:
        pool = get_pool()
    except RuntimeError:
        return
    try:
        await pool.execute(
            f"INSERT INTO {_CT_TABLE} (cache_key, payload, computed_at) "
            f"VALUES ($1, $2::jsonb, now()) "
            f"ON CONFLICT (cache_key) DO UPDATE SET payload = EXCLUDED.payload, computed_at = now()",
            key,
            json.dumps(payload),
        )
    except Exception as exc:
        logger.warning("ct_cache_write_failed", key=key, error=str(exc))


async def _clear_db_cache(pool) -> None:
    """Удалить все строки trend_precomputed через переданный пул."""
    try:
        await pool.execute(f"DELETE FROM {_CT_TABLE}")
    except Exception as exc:
        logger.warning("ct_cache_clear_failed", error=str(exc))


# Отложенные задачи чистки БД-кэша: держим ссылки, чтобы task не собирался
# сборщиком мусора до завершения (asyncio предупреждает о pending-тасках).
_db_clear_tasks: set[asyncio.Task] = set()


def _schedule_db_cache_clear() -> None:
    """Запланировать асинхронную чистку таблицы в текущем цикле событий.

    Вызывается из синхронного колбэка шины cache_bus. Если цикла событий нет
    (тесты в sync-режиме) или пул ещё не поднят (старт приложения) - тихо
    пропускаем: данные не теряются, кэш обновится при следующем промахе.
    """
    try:
        loop = asyncio.get_running_loop()
        pool = get_pool()
    except RuntimeError:
        return
    task = loop.create_task(_clear_db_cache(pool))
    _db_clear_tasks.add(task)
    task.add_done_callback(_db_clear_tasks.discard)


def _clear_trend_caches() -> None:
    """Сбросить кэш трендов и token-множеств после изменения снимков."""
    _CT_CACHE.clear()
    _sw.cache_clear()
    _schedule_db_cache_clear()


from src.cache_bus import register as _register_cache  # noqa: E402

_register_cache("trends", _clear_trend_caches)


def _resolve_canonical_key(
    skill_name: str,
    vocab_keys: set[str],
    inverted_index: dict[str, set[str]] | None = None,
    mapping: dict[str, list[str]] | None = None,
) -> str | None:
    if skill_name in vocab_keys:
        return skill_name
    try:
        from src.parsing.skills.skill_normalizer import SkillNormalizer
        match SkillNormalizer.normalize(skill_name):
            case Ok(norm) if norm != skill_name and norm in vocab_keys:
                return norm
            case _:
                pass
    except Exception:
        logger.debug("swallowed_exception", exc_info=True)
    if len(skill_name) >= 3:
        try:
            from rapidfuzz import fuzz as rp_fuzz
            from rapidfuzz import process as rp_process
            matches = rp_process.extract(skill_name, list(vocab_keys), scorer=rp_fuzz.WRatio, limit=1)
            if matches and matches[0][1] >= 85:
                return matches[0][0]
        except ImportError:
            logger.debug("swallowed_exception", exc_info=True)
    if inverted_index and mapping:
        key = skill_name.lower().strip()
        if key in inverted_index:
            for comp_code in inverted_index[key]:
                for kw in mapping.get(comp_code, []):
                    if kw in vocab_keys:
                        return kw
    return None


@router.get("/competency-trends")
@limiter.limit("60/minute")
async def get_competency_trends(
    request: Request,
    direction: str | None = Query(None, pattern="^(rising|falling|stable)$"),
    limit: int = Query(200, ge=1, le=500),
):
    """Compute per-competency trends directly from trend_snapshots."""
    import time as _time

    cache_key = (direction, limit)
    entry = _CT_CACHE.get(cache_key)
    if entry and _time.monotonic() - entry[0] < _CT_CACHE_TTL:
        return entry[1]

    # Промах in-memory кэша: пробуем БД-кэш trend_precomputed. Он переживает
    # рестарт процесса: при первом запросе после старта _CT_CACHE пуст, но
    # последний расчёт можно поднять из БД, если он не старше TTL.
    db_key = _cache_key(direction, limit)
    db_payload = await _read_db_cache(db_key)
    if db_payload is not None:
        _CT_CACHE[cache_key] = (_time.monotonic(), db_payload)
        return db_payload

    async with async_session_factory() as session:
        # 1. find 2 snapshots with different data for meaningful comparison
        snap_rows = await session.execute(
            select(TrendSnapshot).order_by(TrendSnapshot.snapshot_date.desc())
        )
        all_snaps = snap_rows.scalars().all()
        if len(all_snaps) < 2:
            raise HTTPException(400, "Need at least 2 trend snapshots")

        def parse_freq(raw: Any) -> dict[str, int]:
            if isinstance(raw, str):
                return json.loads(raw)
            return {k: int(v) for k, v in (raw or {}).items()}

        cur = all_snaps[0]
        cur_freq = parse_freq(cur.skill_freq)
        snap_latest = str(cur.snapshot_date)

        # find a previous snapshot with different data (or at least 7 days older)
        prev = None
        for s in all_snaps[1:]:
            prev_freq = parse_freq(s.skill_freq)
            if prev_freq != cur_freq:
                prev = s
                break
            days_diff = (cur.snapshot_date - s.snapshot_date).days
            if days_diff >= 7:
                prev = s
                break
        if prev is None:
            prev = all_snaps[-1]
        prev_freq = parse_freq(prev.skill_freq)
        str(prev.snapshot_date)

        # Разбираем снимки ОДИН раз: в цикле по навыкам json.loads вызывался бы
        # O(навыки × снимки) раз (до 200 × 241).
        parsed_snaps = [
            (str(s.snapshot_date), parse_freq(s.skill_freq)) for s in all_snaps
        ]

        # --- Normalize cur_freq: merge linux + администрирование linux ---
        normalized_cur = dict(cur_freq)
        cur_merge = {"linux": ["администрирование linux"]}
        for target, sources in cur_merge.items():
            for src in sources:
                if src in normalized_cur:
                    normalized_cur[target] += normalized_cur.pop(src)

        # --- Normalize prev_freq: substring alias + overrides ---
        normalized_prev = dict(prev_freq)

        for ck in normalized_cur:
            if ck in normalized_prev or len(ck) < 3:
                continue
            ck_words = _sw(ck)
            if not ck_words:
                continue
            for ok in prev_freq:
                if ok != ck and ck_words <= _sw(ok):
                    normalized_prev[ck] = prev_freq[ok]
                    break

        override_prev = {
            "linux": 1674,
            "c": 1279,
            "huggingface": 15,
        }
        for skill, val in override_prev.items():
            normalized_prev[skill] = val

        # 2. all competencies + their skills
        comp_rows = await session.execute(
            select(
                Competency.id,
                Competency.code,
                Competency.name,
                Skill.name.label("skill_name"),
            )
            .outerjoin(CompetencySkill, CompetencySkill.competency_id == Competency.id)
            .outerjoin(Skill, Skill.id == CompetencySkill.skill_id)
            .order_by(Competency.code, Skill.name)
        )
        comp_map: dict[str, dict] = {}
        for row in comp_rows:
            code = row.code
            if code not in comp_map:
                comp_map[code] = {
                    "id": str(row.id),
                    "name": row.name or "",
                    "skills": [],
                }
            if row.skill_name:
                comp_map[code]["skills"].append(row.skill_name)

        # 3. load mapping for Level 4 fallback
        mapping, inverted = _get_mappings()

        # 4. resolve keys once against cur vocabulary
        cur_keys = set(normalized_cur.keys())

        # 4b. Канонический ключ навыка резолвится один раз на запрос: функция
        # чистая относительно cur_keys/inverted/mapping, а вызывалась бы на
        # каждую пару (компетенция x навык).
        canonical_by_skill: dict[str, str | None] = {}
        for data in comp_map.values():
            for sk in data["skills"]:
                if sk not in canonical_by_skill:
                    canonical_by_skill[sk] = _resolve_canonical_key(
                        sk, cur_keys, inverted, mapping
                    )

        # 4c. Вектор history считается один раз на канонический ключ.
        #
        # Раньше это был сырой ряд из ВСЕХ снимков (241 точка) для каждого навыка
        # каждой компетенции: ~11k навыков x 241 = 2.19 млн объектов, ответ 66 МБ,
        # 12-15 с на сериализацию. Для спарклайна столько не нужно, поэтому:
        #  * сворачиваем снимки в месячные бакеты (до 12 точек);
        #  * считаем только для навыков, которые есть на текущем рынке (cv > 0) -
        #    у отсутствующих история нулевая и график не рисуется;
        #  * значения нормализуем по объёму снимка (как в прогнозе): объём сбора
        #    между снимками различается в разы, сырые counts давали ложные скачки.
        all_dates = [sdate for sdate, _ in parsed_snaps]
        month_order = sorted({_month_key(d) for d in all_dates})[-12:]
        month_index = {m: i for i, m in enumerate(month_order)}
        ref_total = sum(parsed_snaps[0][1].values()) if parsed_snaps else 0
        _history_vectors: dict[str, list[int]] = {}

        # ключ снимка -> доля от объёма снимка
        snap_shares: list[tuple[str, dict[str, float]]] = []
        for sdate, sf in parsed_snaps:
            total = sum(sf.values())
            if total <= 0:
                continue
            snap_shares.append(
                (sdate, {k: v / total for k, v in sf.items()})
            )

        # месячные суммы долей + счётчик снимков в месяце
        month_sums: list[dict[str, float]] = [{} for _ in month_order]
        month_counts = [0] * len(month_order)
        for sdate, shares in snap_shares:
            idx = month_index.get(_month_key(sdate))
            if idx is None:
                continue
            month_counts[idx] += 1
            bucket = month_sums[idx]
            for k, share in shares.items():
                bucket[k] = bucket.get(k, 0.0) + share

        month_labels = [f"{m[0]}-{m[1]:02d}" for m in month_order]

        def _history_vector(canonical: str) -> list[int]:
            cached_vec = _history_vectors.get(canonical)
            if cached_vec is not None:
                return cached_vec
            cw = _sw(canonical)
            long_enough = len(canonical) >= 3
            vec: list[int] = []
            for idx in range(len(month_sums)):
                if not month_counts[idx]:
                    vec.append(0)
                    continue
                bucket = month_sums[idx]
                val = bucket.get(canonical)
                if val is None and long_enough and cw:
                    for ok, ov in bucket.items():
                        if _sw(ok) >= cw:
                            val = ov
                            break
                share = (val or 0.0) / month_counts[idx]
                vec.append(int(round(share * ref_total)))
            _history_vectors[canonical] = vec
            return vec

        # 5. compute trends per competency
        results = []
        for code, data in comp_map.items():
            skill_list = []
            changes: list[float] = []
            rising = 0
            falling = 0
            for sk in sorted(set(data["skills"])):
                canonical = canonical_by_skill[sk]
                cv = normalized_cur.get(canonical, 0) or 0 if canonical else 0
                pv = normalized_prev.get(canonical, 0) or 0 if canonical else 0

                if pv >= 10:
                    chg = round((cv - pv) / pv * 100, 1)
                    if chg > 200:
                        chg = 200
                    elif chg < -200:
                        chg = -200
                    changes.append(chg)
                    if chg >= 5:
                        rising += 1
                    elif chg <= -5:
                        falling += 1
                else:
                    chg = 0.0

                # История только для навыков, реально присутствующих на рынке:
                # у нулевых её не рисуют, а весит она больше всего.
                if canonical and cv > 0:
                    vals = _history_vector(canonical)
                    history = [
                        {"date": month_labels[i], "freq": vals[i]}
                        for i in range(len(month_labels))
                    ]
                else:
                    history = []

                skill_list.append({
                    "name": sk,
                    "direction": _classify(chg),
                    "change_pct": chg,
                    "frequency": cv,
                    "prev_frequency": pv,
                    "history": history,
                })

            if not changes:
                continue

            agg = round(statistics.median(changes), 1)
            total_ch = len(changes)
            stable = total_ch - rising - falling
            if rising > falling and rising > stable:
                comp_dir = "rising"
            elif falling > rising and falling > stable:
                comp_dir = "falling"
            else:
                comp_dir = "stable"

            results.append({
                "competency_id": data["id"],
                "code": code,
                "name": data["name"],
                "direction": comp_dir,
                "change_pct": agg,
                "skill_count": len(data["skills"]),
                "active_skills_count": len(changes),
                "snapshot_date": snap_latest,
                "skills": skill_list,
            })

        # 6. sort: rising first, then by |change_pct| desc
        results.sort(
            key=lambda r: (
                0 if r["direction"] == "rising" else 1 if r["direction"] == "falling" else 2,
                -abs(r["change_pct"]),
            )
        )

        # 7. filter
        if direction:
            results = [r for r in results if r["direction"] == direction]

        payload = {"total": len(results), "trends": results[:limit]}
        _CT_CACHE[cache_key] = (_time.monotonic(), payload)
        await _write_db_cache(db_key, payload)
        return payload
