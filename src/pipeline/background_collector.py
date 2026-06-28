"""Background incremental vacancy collector — runs main.py periodically."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_INTERVAL_HOURS = 6
_BASE_DIR = Path(__file__).resolve().parent.parent.parent


async def start_background_collector():
    """Start periodic vacancy collection as a background task."""
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
    """Check if due, then run main.py --it-sector."""
    import asyncpg
    from datetime import datetime, timezone
    from src import config

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

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(_BASE_DIR / "main.py"),
        "--it-sector", "--skip-gap-analysis",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=_BASE_DIR,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("collect_timeout")
        return

    if proc.returncode != 0:
        logger.warning("collect_failed", code=proc.returncode)
        return

    try:
        conn = await asyncpg.connect(db_url)
        try:
            after = await conn.fetchval("SELECT COUNT(*) FROM vacancies") or 0
        finally:
            await conn.close()
        logger.info("collect_done", before=before, after=after, new=after - before)
    except Exception:
        pass
