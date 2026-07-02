"""Background incremental vacancy collector — direct hh.ru collection only."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_INTERVAL_HOURS = 6
_BASE_DIR = Path(__file__).resolve().parent.parent.parent


async def start_background_collector():
    asyncio.create_task(_collect_loop())


async def _collect_loop():
    await asyncio.sleep(30)
    logger.info("background_collector_started", interval_hours=_INTERVAL_HOURS)
    while True:
        try:
            await _try_collect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("background_collect_error", error=str(exc), exc_info=True)
        await asyncio.sleep(_INTERVAL_HOURS * 3600)


async def _try_collect():
    """Direct HH API collection, saves JSON + DB."""
    import asyncpg
    from src import config, Ok, Result
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
        finally:
            await conn.close()
    except Exception:
        logger.warning("collect_db_unavailable")
        return

    if last_run:
        elapsed = (datetime.now(timezone.utc) - last_run.replace(tzinfo=timezone.utc)).total_seconds()
        if elapsed < _INTERVAL_HOURS * 3600:
            logger.debug("collect_skipped_recent", last_run=str(last_run)[:16])
            return

    max_pages = 5
    period = 30

    if last_run:
        delta = (datetime.now(timezone.utc) - last_run.replace(tzinfo=timezone.utc)).days
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



    if not all_vacancies:
        logger.info("collect_no_new_vacancies")
        return

    # Save to JSON
    detailed_path = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    try:
        import json as j
        existing = j.loads(detailed_path.read_text(encoding="utf-8")) if detailed_path.exists() else []
        existing_ids = {v.get("id") for v in existing if v.get("id")}
        merged = existing + [v for v in all_vacancies if v.get("id") and v["id"] not in existing_ids]
        detailed_path.write_text(j.dumps(merged, ensure_ascii=False, default=str), encoding="utf-8")
        logger.info("collect_json_saved", total=len(merged))
    except Exception as exc:
        logger.warning("collect_json_save_failed", error=str(exc))

    # Enrich with full details (description) from HH API
    logger.info("collect_enrich_details", count=len(all_vacancies))
    for v in all_vacancies:
        vid = v.get("id")
        if not vid:
            continue
        if v.get("description"):
            continue
        try:
            match await asyncio.to_thread(api.get_vacancy_details, str(vid)):
                case Ok(details):
                    if details.get("description"):
                        v["description"] = details["description"]
                    if details.get("snippet"):
                        existing_snippet = v.get("snippet", {}) or {}
                        det_snippet = details.get("snippet", {}) or {}
                        v["snippet"] = {
                            "requirement": existing_snippet.get("requirement") or det_snippet.get("requirement", ""),
                            "responsibility": existing_snippet.get("responsibility") or det_snippet.get("responsibility", ""),
                        }
                case _:
                    pass
        except Exception as exc:
            logger.debug("collect_detail_failed", id=vid, error=str(exc))

    # Parse skills BEFORE converting IDs (Vacancy.from_api expects str id)
    parsed_count = 0
    skip_count = 0
    empty_ids: set[int] = set()
    try:
        from src.parsing.skills.vacancy_parser import VacancyParser
        from src.models.vacancy import Vacancy as VacModel
        import re as _re
        # Preload it_skills keywords for fast substring check
        _it_kw = {s.strip().lower() for s in json.loads(
            (Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "it_skills.json").read_text(encoding="utf-8")
        ) if s.strip()}
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
