"""Background incremental vacancy collector — direct hh.ru collection only."""
from __future__ import annotations

import asyncio
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
    from src import config
    from src.parsing.api.hh_api import HeadHunterAPI
    from src.parsing.utils import (
        get_areas, 
        IT_PROFESSIONAL_ROLES, 
        IT_PROFESSIONAL_QUERIES,
    )

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

    api = HeadHunterAPI()
    area_id = 2
    max_pages = 5
    period = 30

    all_vacancies = []
    seen_ids: set[int] = set()

    if last_run:
        delta = (datetime.now(timezone.utc) - last_run.replace(tzinfo=timezone.utc)).days
        if 1 <= delta <= 30:
            period = delta
            logger.info("collect_incremental", days=period)

    for role_id in IT_PROFESSIONAL_ROLES:
        try:
            result = api.search_vacancies(
                query="",
                area=area_id,
                period=period,
                max_pages=max_pages,
                professional_role=role_id,
            )
            match result:
                case Ok(vacs):
                    for v in vacs:
                        vid = v.get("id")
                        if vid and vid not in seen_ids:
                            seen_ids.add(vid)
                            all_vacancies.append(v)
                case _:
                    pass
        except Exception as exc:
            logger.warning("collect_role_failed", role=role_id, error=str(exc))

    for query in IT_PROFESSIONAL_QUERIES:
        try:
            result = api.search_vacancies(query=query, area=area_id, period=period, max_pages=max_pages)
            match result:
                case Ok(vacs):
                    for v in vacs:
                        vid = v.get("id")
                        if vid and vid not in seen_ids:
                            seen_ids.add(vid)
                            all_vacancies.append(v)
                case _:
                    pass
        except Exception as exc:
            logger.warning("collect_query_failed", query=query, error=str(exc))

    if not all_vacancies:
        logger.info("collect_no_new_vacancies")
        return

    # Save to JSON
    detailed_path = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    try:
        import json as j
        existing = json.loads(detailed_path.read_text(encoding="utf-8")) if detailed_path.exists() else []
        existing_ids = {v.get("id") for v in existing if v.get("id")}
        merged = existing + [v for v in all_vacancies if v.get("id") and v["id"] not in existing_ids]
        detailed_path.write_text(j.dumps(merged, ensure_ascii=False, default=str), encoding="utf-8")
        logger.info("collect_json_saved", total=len(merged))
    except Exception as exc:
        logger.warning("collect_json_save_failed", error=str(exc))

    # Save to DB
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
