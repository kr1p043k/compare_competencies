"""Write pipeline results to PostgreSQL via asyncpg (raw SQL, no SQLAlchemy)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import structlog

from src import config
from src.db import create_pool, close_pool, get_pool

logger = structlog.get_logger(__name__)


async def _pool():
    """Ensure pool exists and return it."""
    try:
        return get_pool()
    except RuntimeError:
        await create_pool()
        return get_pool()


async def create_pipeline_run(action: str) -> str:
    pool = await _pool()
    rid = str(uuid.uuid4())
    await pool.execute(
        """INSERT INTO pipeline_runs (id, action, status, started_at)
           VALUES ($1, $2, 'started', NOW())""",
        rid, action,
    )
    return rid


async def complete_pipeline_run(run_id: str, status: str = "completed", error: str | None = None, stats: dict | None = None) -> None:
    pool = await _pool()
    await pool.execute(
        """UPDATE pipeline_runs SET status=$1, completed_at=NOW(),
           error_message=$2, stats=$3 WHERE id=$4""",
        status, error, json.dumps(stats) if stats else None, run_id,
    )


async def save_coverage_from_json(json_path: Path | None = None, run_id: str | None = None) -> int:
    if json_path is None:
        logger.debug("coverage_json_skipped", reason="no path provided")
        return 0
    if not json_path.exists():
        logger.warning("coverage_json_not_found", path=str(json_path))
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if not report:
        return 0

    pool = await _pool()
    count = 0

    async with pool.acquire() as conn:
        for disc_name, disc_data in report.items():
            dn = disc_name.lower().replace("_", " ")
            disc_row = await conn.fetchrow("SELECT id FROM disciplines WHERE LOWER(name)=$1", dn)
            if not disc_row:
                continue

            disc_id = disc_row["id"]
            skills = disc_data.get("skills", [])
            market_skills = disc_data.get("market_skills", [])

            total = len(skills)
            ms_lower = {m.lower() for m in market_skills}
            matched = sum(1 for s in skills if s.lower() in ms_lower)
            ratio = round(matched / total, 4) if total > 0 else 0.0

            await conn.execute(
                """INSERT INTO coverage_analyses
                   (discipline_id, total_skills, market_matched_skills, coverage_ratio, analysis_date)
                   VALUES ($1,$2,$3,$4,NOW())""",
                disc_id, total, matched, ratio,
            )

            for ms in market_skills:
                sk_row = await conn.fetchrow("SELECT id FROM skills WHERE name=$1", ms.lower())
                if sk_row:
                    await conn.execute(
                        """INSERT INTO market_skill_mappings
                           (skill_id, market_skill_name, frequency, weight, period, source)
                           VALUES ($1,$2,1,1.0,CURRENT_DATE,'gap_analysis')
                           ON CONFLICT DO NOTHING""",
                        sk_row["id"], ms,
                    )
            count += 1

    logger.info("coverage_saved", disciplines=count)
    return count


async def save_gap_analysis(data: dict, run_id: str | None = None) -> None:
    pool = await _pool()
    await pool.execute(
        """INSERT INTO analysis_results (pipeline_run_id, analysis_type, data)
           VALUES ($1,'gap',$2)""",
        run_id, json.dumps(data, ensure_ascii=False),
    )


async def save_to_analysis_results(run_id: str, analysis_type: str, data: dict) -> None:
    pool = await _pool()
    await pool.execute(
        """INSERT INTO analysis_results (pipeline_run_id, analysis_type, data)
           VALUES ($1, $2, $3)""",
        run_id, analysis_type, json.dumps(data, ensure_ascii=False, default=str),
    )


async def save_trend_snapshot(snapshot_date: datetime, skill_freq: dict, source: str = "hh_vacancies", run_id: str | None = None) -> None:
    pool = await _pool()
    await pool.execute(
        """INSERT INTO trend_snapshots (pipeline_run_id, snapshot_date, skill_freq, source)
           VALUES ($1,$2,$3::jsonb,$4)""",
        run_id, snapshot_date, json.dumps(skill_freq, ensure_ascii=False), source,
    )


async def export_history_trends_to_db() -> int:
    """Import all freq_*.json from data/history into trend_snapshots table."""
    total = 0
    for f in sorted(config.HISTORY_DIR.glob("freq_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            date_str = f.stem.split("_")[-1] if "_" in f.stem else f.stem
            try:
                d = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except ValueError:
                d = datetime.utcnow()
            await save_trend_snapshot(d, data)
            total += 1
        except Exception as e:
            logger.warning("trend_import_failed", file=f.name, error=str(e))
    return total
