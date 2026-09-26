"""Background scheduler: incremental hh.ru collection + chained daily gap-analysis.

Settings live in data/settings/scheduler.json (admin UI); env only seeds
defaults on first boot. The loop self-gates: it never runs while a pipeline
task is active, and waits for API warmup before the first cycle.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

from src import config
from src.circuit_breaker import CircuitBreaker
from src.utils import atomic_write_json

SCHEDULER_DIRNAME = "settings"
SCHEDULER_FILENAME = "scheduler.json"
SCHEDULER_DEFAULT_INTERVAL_HOURS = 12
SCHEDULER_MAX_INTERVAL_HOURS = 72
SCHEDULER_GAP_STALE_DAYS = 7
JSON_MAX_RECORDS = 150000

_scheduler_cache: dict | None = None
_scheduler_busy: str | None = None


def _scheduler_path() -> Path:
    return config.DATA_DIR / SCHEDULER_DIRNAME / SCHEDULER_FILENAME


def _seed_defaults() -> dict:
    try:
        enabled = bool(config.settings.BACKGROUND_COLLECTOR_ENABLED)
    except Exception:
        enabled = False
    try:
        interval = int(config.settings.SCHEDULER_COLLECT_INTERVAL_HOURS)
    except Exception:
        interval = SCHEDULER_DEFAULT_INTERVAL_HOURS
    try:
        daily = bool(config.settings.SCHEDULER_DAILY_GAP_ENABLED)
    except Exception:
        daily = False
    return {
        "collector_enabled": enabled,
        "collect_interval_hours": interval,
        "daily_gap_enabled": daily,
        "last_collect_ts": None,
        "last_gap_date": None,
    }


def load_scheduler_settings() -> dict:
    """Read persisted settings merged over seed defaults. Never raises."""
    merged = _seed_defaults()
    try:
        path = _scheduler_path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("collector_enabled", "daily_gap_enabled"):
                    if isinstance(data.get(key), bool):
                        merged[key] = data[key]
                iv = data.get("collect_interval_hours")
                if isinstance(iv, (int, float)) and 1 <= iv <= SCHEDULER_MAX_INTERVAL_HOURS:
                    merged["collect_interval_hours"] = int(iv)
                ts = data.get("last_collect_ts")
                if ts is None or isinstance(ts, (int, float)):
                    merged["last_collect_ts"] = ts
                gd = data.get("last_gap_date")
                if gd is None or (isinstance(gd, str) and len(gd) == 10):
                    merged["last_gap_date"] = gd
    except Exception as exc:
        logger.warning("scheduler_settings_load_failed", error=str(exc))
    global _scheduler_cache
    _scheduler_cache = dict(merged)
    return dict(merged)


def save_scheduler_settings(patch: dict) -> dict:
    """Merge a partial patch, persist atomically, return full settings."""
    current = load_scheduler_settings()
    patch = patch or {}
    for key in ("collector_enabled", "daily_gap_enabled"):
        if key in patch:
            current[key] = bool(patch[key])
    if "collect_interval_hours" in patch:
        try:
            iv = int(patch["collect_interval_hours"])
        except (TypeError, ValueError):
            iv = current["collect_interval_hours"]
        current["collect_interval_hours"] = max(1, min(SCHEDULER_MAX_INTERVAL_HOURS, iv))
    for key in ("last_collect_ts", "last_gap_date"):
        if key in patch:
            current[key] = patch[key]
    try:
        path = _scheduler_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(current, path)
    except Exception as exc:
        logger.warning("scheduler_settings_save_failed", error=str(exc))
    global _scheduler_cache
    _scheduler_cache = dict(current)
    return dict(current)


def is_scheduler_busy() -> bool:
    return _scheduler_busy is not None


def scheduler_status() -> dict:
    """Settings + liveness for the admin UI. Never raises."""
    try:
        st = load_scheduler_settings()
        now = time.time()
        interval_s = st["collect_interval_hours"] * 3600
        last = st["last_collect_ts"]
        nxt = (last + interval_s) if isinstance(last, (int, float)) else None
        return {
            **st,
            "busy": _scheduler_busy,
            "pipeline_running": _pipeline_running(),
        "next_collect_ts": nxt,
        "next_collect_in_s": max(0, int(nxt - now)) if nxt else 0,
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {"error": str(exc)[:200], "busy": _scheduler_busy}


def is_collect_due(now_ts: float, last_ts, interval_h: int) -> bool:
    if last_ts is None:
        return True
    try:
        return (float(now_ts) - float(last_ts)) >= int(interval_h) * 3600
    except (TypeError, ValueError):
        return True


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def is_gap_due(today: str, last_gap_date) -> bool:
    return last_gap_date != today


def is_gap_overdue(today: str, last_gap_date, max_days: int = SCHEDULER_GAP_STALE_DAYS) -> bool:
    if not last_gap_date:
        return True
    try:
        d0 = datetime.strptime(str(last_gap_date), "%Y-%m-%d").date()
        d1 = datetime.strptime(str(today), "%Y-%m-%d").date()
        return (d1 - d0).days >= max_days
    except (ValueError, TypeError):
        return True


def _pipeline_running() -> bool:
    try:
        from src.api_pkg.routers.pipeline import pipeline_tasks
        return any(getattr(t, "status", "") == "running" for t in pipeline_tasks.values())
    except Exception:
        return False


async def _wait_until_ready(timeout_s: int = 7200) -> bool:
    from src.api_pkg import deps as _deps
    waited = 0
    while waited < timeout_s:
        try:
            if _deps.is_ready:
                return True
        except Exception:
            pass
        await asyncio.sleep(15)
        waited += 15
    return False


async def start_background_collector():
    asyncio.create_task(_scheduler_loop())


async def _scheduler_loop() -> None:
    ready = await _wait_until_ready()
    if not ready:
        logger.warning("scheduler_startup_not_ready")
    collect_breaker = CircuitBreaker(fail_threshold=3, cooldown_s=12 * 3600)
    gap_breaker = CircuitBreaker(fail_threshold=2, cooldown_s=24 * 3600)
    logger.info("scheduler_started")
    while True:
        try:
            st = load_scheduler_settings()
            runnable = (
                not is_scheduler_busy()
                and not _pipeline_running()
                and st["collector_enabled"]
                and collect_breaker.allow()
                and is_collect_due(time.time(), st["last_collect_ts"], st["collect_interval_hours"])
            )
            if runnable:
                status, _new = await _run_collect_once()
                if status == "collected":
                    collect_breaker.record_success()
                    await _maybe_chained_gap(st, gap_breaker)
                elif status == "collected-empty":
                    collect_breaker.record_success()
                    if (
                        st["daily_gap_enabled"]
                        and gap_breaker.allow()
                        and is_gap_overdue(_today_utc(), st["last_gap_date"])
                    ):
                        await _run_chained_gap(gap_breaker)
                elif status == "failed":
                    collect_breaker.record_failure()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("scheduler_tick_error", error=str(exc))
        await asyncio.sleep(60)


async def _maybe_chained_gap(st: dict, gap_breaker) -> None:
    if not st["daily_gap_enabled"]:
        return
    if not gap_breaker.allow():
        return
    if not is_gap_due(_today_utc(), st["last_gap_date"]):
        return
    if _pipeline_running():
        logger.info("scheduled_gap_deferred_pipeline_running")
        return
    await _run_chained_gap(gap_breaker)


async def _run_chained_gap(gap_breaker) -> bool:
    ok = await _run_scheduled_gap()
    if ok:
        gap_breaker.record_success()
    else:
        gap_breaker.record_failure()
    return ok


async def _run_collect_once(*, force: bool = False, force_period_days: int | None = None) -> tuple:
    global _scheduler_busy
    if is_scheduler_busy():
        return "skipped", 0
    _scheduler_busy = "collect"
    try:
        status, new_count = await _try_collect(force_period_days=force_period_days, force=force)
        if status == "collected":
            save_scheduler_settings({"last_collect_ts": time.time()})
        return status, new_count
    except Exception as exc:
        logger.warning("scheduler_collect_failed", error=str(exc))
        return "failed", 0
    finally:
        _scheduler_busy = None


async def _run_scheduled_gap() -> bool:
    global _scheduler_busy
    if is_scheduler_busy():
        return False
    _scheduler_busy = "gap"
    try:
        from src.api_pkg import deps as _deps
        if _deps.evaluator is None:
            logger.warning("scheduled_gap_no_evaluator")
            return False
        profiles = dict(_deps.student_profiles or {})
        if not profiles:
            logger.warning("scheduled_gap_no_profiles")
            return False
        from src.api_pkg.routers.pipeline import PipelineAction, run_pipeline_task
        task_id = f"scheduled_gap_{_today_utc()}"
        await run_pipeline_task(
            PipelineAction.GAP_ANALYSIS, task_id,
            skip_collection=True, run_gap_analysis=True,
            profiles_override=profiles,
        )
        from src.api_pkg.routers.pipeline import pipeline_tasks
        final = pipeline_tasks.get(task_id)
        ok = bool(final is not None and getattr(final, "status", "") == "completed")
        if ok:
            save_scheduler_settings({"last_gap_date": _today_utc()})
        else:
            logger.warning("scheduled_gap_not_completed", task_id=task_id)
        return ok
    except Exception as exc:
        logger.warning("scheduled_gap_failed", error=str(exc))
        return False
    finally:
        _scheduler_busy = None


async def _refresh_history_snapshots() -> None:
    """backfill пропущенных месяцев + снимки по профессиям (не чаще раза в сутки)."""
    try:
        from src.cli.backfill_market_snapshots import main as backfill_main
        await asyncio.to_thread(backfill_main, force=False)
    except Exception as exc:
        logger.warning("collect_backfill_snapshots_error", error=str(exc))
    try:
        from src import config as _config
        prof_files = list(_config.HISTORY_DIR.glob("freq_profession_*.json"))
        too_recent = bool(prof_files) and (
            time.time() - max(f.stat().st_mtime for f in prof_files) < 24 * 3600
        )
        if not too_recent:
            from src.cli.snapshot_professions import main as prof_main
            await asyncio.to_thread(prof_main, force=False)
    except Exception as exc:
        logger.warning("collect_profession_snapshots_error", error=str(exc))


async def _try_collect(force_period_days: int | None = None, force: bool = False):
    """Direct HH API collection, saves JSON + DB.

    Defaults preserve background behavior (skip if recent, incremental period).
    force_period_days=30 + force=True performs a full monthly sweep.
    Returns (status, new_count): status is 'collected' | 'collected-empty' |
    "skipped" | "failed". Only "collected" chains the daily gap.
    """
    import asyncpg
    from src import Err, Ok, Result, config
    from src.parsing.api.hh_api import HeadHunterAPI
    from src.parsing.utils import IT_PROFESSIONAL_ROLES

    db_url = config.settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    try:
        conn = await asyncpg.connect(db_url)
        try:
            last_run = await conn.fetchval(
                "SELECT MAX(started_at) FROM pipeline_runs WHERE action IN ('full-cycle','data-collection') AND status='completed'"
            )
            before = await conn.fetchval("SELECT COUNT(*) FROM vacancies") or 0
            after = before
        finally:
            await conn.close()
    except Exception:
        logger.warning("collect_db_unavailable")
        return "failed", 0

    if last_run and not force:
        now_utc = datetime.now(timezone.utc)
        last_aware = last_run if last_run.tzinfo else last_run.replace(tzinfo=timezone.utc)
        elapsed = (now_utc - last_aware).total_seconds()
        gate_interval_s = load_scheduler_settings()["collect_interval_hours"] * 3600
        if elapsed < gate_interval_s:
            logger.debug("collect_skipped_recent", last_run=str(last_run)[:16])
            return "skipped", 0

    max_pages = 5
    period = 30

    if force_period_days is not None:
        period = max(1, min(int(force_period_days), 30))
        logger.info("collect_forced_period", days=period)
    elif last_run:
        now_utc = datetime.now(timezone.utc)
        last_aware = last_run if last_run.tzinfo else last_run.replace(tzinfo=timezone.utc)
        delta = (now_utc - last_aware).days
        if 1 <= delta <= 30:
            period = delta
            logger.info("collect_incremental", days=period)

    region_ids = list(range(1, 201))
    all_vacancies = []
    seen_ids: set[int] = set()
    _lock = asyncio.Lock()
    _last_req_time = 0.0
    _req_lock = asyncio.Lock()

    async def _rate_limit():
        """Ensure at most 3 requests per second to HH API."""
        nonlocal _last_req_time
        async with _req_lock:
            now = time.monotonic()
            wait = max(0.0, 1.0 / 3 - (now - _last_req_time))
            if wait > 0:
                await asyncio.sleep(wait)
            _last_req_time = time.monotonic()

    api = HeadHunterAPI()

    async def _collect_region(region_id: int):
        """Collect vacancies for a single region across all IT roles."""
        local = []
        for role_id in IT_PROFESSIONAL_ROLES:
            try:
                await _rate_limit()
                result = await asyncio.to_thread(
                    api.search_vacancies,
                    text="", area=region_id, period_days=period,
                    max_pages=max_pages, professional_role=role_id,
                )
                if result.is_ok():
                    for v in result.ok():
                        vid = v.get("id")
                        if vid:
                            async with _lock:
                                if vid not in seen_ids:
                                    seen_ids.add(vid)
                                    local.append(v)
            except Exception as exc:
                logger.debug("collect_region_role_failed", region=region_id, role=role_id, error=str(exc))
        if local:
            logger.info("collect_region_done", region=region_id, count=len(local))
        return local

    # All regions in parallel with rate limiting
    tasks = [_collect_region(rid) for rid in region_ids]
    region_results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in region_results:
        if isinstance(r, list):
            all_vacancies.extend(r)
        elif isinstance(r, BaseException):
            logger.warning("collect_region_exception", error=str(r))

    logger.info("collect_total_vacancies", count=len(all_vacancies))
    await _refresh_history_snapshots()



    if not all_vacancies:
        logger.info("collect_no_new_vacancies")
        return "collected-empty", 0

    # Save to JSON (async I/O — не блокируем event loop на 453 МБ файле)
    detailed_path = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    try:
        import json as j
        if detailed_path.exists():
            raw_text = await asyncio.to_thread(detailed_path.read_text, encoding="utf-8")
            existing = j.loads(raw_text)
        else:
            existing = []
        existing_ids = {v.get("id") for v in existing if v.get("id")}
        merged = existing + [v for v in all_vacancies if v.get("id") and v["id"] not in existing_ids]
        # prune oldest by published_at so the file cannot grow unbounded;
        # records without a date are always kept.
        dated = sorted(
            [v for v in merged if v.get("published_at")],
            key=lambda v: str(v.get("published_at")),
            reverse=True,
        )
        undated = [v for v in merged if not v.get("published_at")]
        merged = dated[:JSON_MAX_RECORDS] + undated
        await asyncio.to_thread(atomic_write_json, merged, detailed_path)
        logger.info("collect_json_saved", total=len(merged))
    except Exception as exc:
        logger.warning("collect_json_save_failed", error=str(exc))

    # Enrich with full details (description) from HH API
    logger.info("collect_enrich_details", count=len(all_vacancies))
    _detail_sem = asyncio.Semaphore(4)

    async def _enrich_one(vac: dict):
        async with _detail_sem:
            vid = vac.get("id")
            if not vid:
                return
            if vac.get("description"):
                return
            try:
                await _rate_limit()
                match await asyncio.to_thread(api.get_vacancy_details, str(vid)):
                    case Ok(details):
                        if details.get("description"):
                            vac["description"] = details["description"]
                        if details.get("snippet"):
                            existing_snippet = vac.get("snippet", {}) or {}
                            det_snippet = details.get("snippet", {}) or {}
                            vac["snippet"] = {
                                "requirement": existing_snippet.get("requirement")
                                or det_snippet.get("requirement", ""),
                                "responsibility": existing_snippet.get("responsibility")
                                or det_snippet.get("responsibility", ""),
                            }
                    case _:
                        pass
            except Exception as exc:
                logger.debug("collect_detail_failed", id=vid, error=str(exc))

    await asyncio.gather(*(_enrich_one(v) for v in all_vacancies))

    # Parse skills BEFORE converting IDs (Vacancy.from_api expects str id)
    parsed_count = 0
    skip_count = 0
    empty_ids: set[int] = set()
    try:
        from src.parsing.skills.vacancy_parser import VacancyParser
        from src.models.vacancy import Vacancy as VacModel
        import re as _re
        # Preload it_skills keywords for fast substring check
        _it_path = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "it_skills.json"
        _it_raw = await asyncio.to_thread(_it_path.read_text, encoding="utf-8")
        _it_kw = {s.strip().lower() for s in json.loads(_it_raw) if s.strip()}
        parser = VacancyParser()
        conn = await asyncpg.connect(db_url)
        try:
            for v in all_vacancies:
                vid = v.get("id")
                if not vid:
                    continue
                try:
                    vac_obj = VacModel.from_api(v)
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    logger.warning("collect_parse_skip_vacancy", id=vid, error=str(exc))
                    skip_count += 1
                    continue
                match parser.skill_parser.parse_vacancy(vac_obj):
                    case Ok(extracted):
                        texts = list(dict.fromkeys(s.text for s in extracted if s.text))
                        # Write-back for the later INSERT (it reads extracted_skills;
                        # the UPDATE below only matches already-stored rows).
                        v["extracted_skills"] = texts
                        if texts:
                            hh_id = int(vid)
                            await conn.execute(
                                "UPDATE vacancies SET parsed_skills = $1::jsonb WHERE hh_id = $2",
                                json.dumps(texts), hh_id,
                            )
                            parsed_count += 1
                        else:
                            # No skills parsed — check if vacancy is actually IT
                            desc = v.get("description", "") or ""
                            key_skills = [s.get("name", "") for s in v.get("key_skills", []) if s.get("name")]
                            has_it = bool(key_skills)
                            if not has_it and len(desc) > 50:
                                # Quick substring check: does description mention any it_skills keyword?
                                desc_lower = desc.lower()
                                has_it = any(kw in desc_lower for kw in _it_kw)
                            if not has_it:
                                empty_ids.add(int(vid))
                                skip_count += 1
                    case Err(e):
                        logger.warning("collect_parse_skill_failed", id=vid, error=str(e))
                        skip_count += 1
        finally:
            await conn.close()
        # Remove vacancies that had zero IT relevance
        if empty_ids:
            logger.info("collect_removing_non_it_vacancies", count=len(empty_ids))
            all_vacancies[:] = [v for v in all_vacancies if v.get("id") not in empty_ids]
        logger.info("collect_parse_done", parsed=parsed_count, skipped=skip_count, total=len(all_vacancies))
    except Exception as exc:
        logger.warning("collect_parse_failed", error=str(exc), parsed=parsed_count, skipped=skip_count)

    # Save to DB (ensure IDs are int)
    for v in all_vacancies:
        if isinstance(v.get("id"), str):
            try:
                v["id"] = int(v["id"])
            except (ValueError, TypeError):
                pass
    from src.pipeline.db_writer import save_vacancies_batch
    await save_vacancies_batch(all_vacancies)

    try:
        conn = await asyncpg.connect(db_url)
        try:
            after = await conn.fetchval("SELECT COUNT(*) FROM vacancies") or 0
        finally:
            await conn.close()
        logger.info("collect_done", before=before, after=after, new=after - before, collected=len(all_vacancies))
    except Exception:
        pass
    try:
        from src.pipeline.db_writer import complete_pipeline_run, create_pipeline_run
        _rid = await create_pipeline_run("data-collection")
        await complete_pipeline_run(
            _rid, status="completed",
            stats={"collected": len(all_vacancies), "new": after - before},
        )
    except Exception as exc:
        logger.warning("collect_run_record_failed", error=str(exc))

    # Авто-снимок рынка: обновляет freq_market_YYYY-MM.json и trend_snapshots,
    # чтобы Prophet всегда имел свежую точку истории (не только при полном pipeline).
    try:
        from collections import Counter
        from src.analyzers.skills.trends import TrendAnalyzer
        from src.parsing.skills.skill_normalizer import SkillNormalizer

        freq: Counter = Counter()
        for v in all_vacancies:
            ps = v.get("extracted_skills") or v.get("parsed_skills") or []
            if isinstance(ps, str):
                try:
                    ps = json.loads(ps)
                except Exception:
                    ps = []
            for s in ps:
                if not isinstance(s, str):
                    continue
                n = SkillNormalizer.normalize(s)
                if n.is_ok() and n.unwrap():
                    freq[n.unwrap()] += 1
        if freq:
            analyzer = TrendAnalyzer(dict(freq))
            res = analyzer.save_snapshot(dict(freq), apply_whitelist=True,
                                           source_type="full_market",
                                           vacancy_count=len(all_vacancies))
            if res.is_ok():
                logger.info("collect_snapshot_saved", path=str(res.unwrap()), skills=len(freq))
            else:
                logger.warning("collect_snapshot_save_failed", error=str(res.unwrap_err()))

        # Восполнить пропущенные месяцы из parsed_skills (например июль), чтобы
        # история прогнозов не теряла точки. Не перезаписывает существующие снимки.
    except Exception as exc:
        logger.warning("collect_snapshot_error", error=str(exc))
    return "collected", after - before
