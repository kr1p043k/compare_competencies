"""Write pipeline results to PostgreSQL (coverage, market mappings, analysis)."""

import json
import structlog
from datetime import datetime
from pathlib import Path
from sqlalchemy import select, text

from src import config
from src.database import async_session_factory
from src.models.krm_models import (
    AnalysisResult, CoverageAnalysis, Discipline,
    MarketSkillMapping, PipelineRun, Skill, TrendSnapshot,
)

logger = structlog.get_logger(__name__)


async def create_pipeline_run(action: str) -> str:
    """Create a pipeline_runs record, return its id."""
    async with async_session_factory() as session:
        run = PipelineRun(action=action, status="started", started_at=datetime.utcnow())
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run.id


async def complete_pipeline_run(run_id: str, status: str = "completed", error: str | None = None, stats: dict | None = None) -> None:
    """Mark a pipeline run as completed/failed."""
    from sqlalchemy import update as sa_update
    async with async_session_factory() as session:
        values = {"status": status, "completed_at": datetime.utcnow()}
        if error:
            values["error_message"] = error
        if stats:
            values["stats"] = stats
        await session.execute(
            sa_update(PipelineRun).where(PipelineRun.id == run_id).values(**values)
        )
        await session.commit()


async def save_coverage_from_json(json_path: Path | None = None, run_id: str | None = None) -> int:
    """Read coverage data from JSON and write to coverage_analyses + market_skill_mappings."""
    if json_path is None:
        json_path = config.DATA_RESULT_DIR / "gap_report.json"
    if not json_path.exists():
        logger.warning("coverage_json_not_found", path=str(json_path))
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    count = 0
    async with async_session_factory() as session:
        # Get direction for discipline lookup
        disciplines = await session.execute(select(Discipline))
        disc_map = {d.name.lower(): d for d in disciplines.scalars().all()}

        for disc_name, disc_data in report.items():
            disc = disc_map.get(disc_name.lower().replace("_", " "))
            if not disc:
                continue

            skills = disc_data.get("skills", [])
            market_skills = disc_data.get("market_skills", [])

            total = len(skills)
            matched = sum(1 for s in skills if s.lower() in {m.lower() for m in market_skills})
            ratio = round(matched / total, 4) if total > 0 else 0.0

            ca = CoverageAnalysis(
                discipline_id=disc.id,
                total_skills=total,
                market_matched_skills=matched,
                coverage_ratio=ratio,
                analysis_date=datetime.utcnow(),
            )
            session.add(ca)

            for ms in market_skills:
                skill = await session.execute(select(Skill).where(Skill.name == ms.lower()))
                sk = skill.scalar_one_or_none()
                if sk:
                    mapping = MarketSkillMapping(
                        skill_id=sk.id,
                        market_skill_name=ms,
                        frequency=1,
                        weight=1.0,
                        period=datetime.utcnow().date(),
                        source="gap_analysis",
                    )
                    session.add(mapping)
            count += 1

        await session.commit()
    logger.info("coverage_saved", disciplines=count)
    return count


async def save_gap_analysis(data: dict, run_id: str | None = None) -> None:
    """Write full gap analysis result to analysis_results."""
    async with async_session_factory() as session:
        result = AnalysisResult(
            pipeline_run_id=run_id,
            analysis_type="gap",
            data=data,
        )
        session.add(result)
        await session.commit()


async def save_trend_snapshot(snapshot_date: datetime, skill_freq: dict, source: str = "hh_vacancies", run_id: str | None = None) -> None:
    """Write trend snapshot to DB."""
    async with async_session_factory() as session:
        ts = TrendSnapshot(
            pipeline_run_id=run_id,
            snapshot_date=snapshot_date,
            skill_freq=skill_freq,
            source=source,
        )
        session.add(ts)
        await session.commit()


async def export_file_results_to_db() -> int:
    """Migrate all existing JSON results to DB. Returns count of records written."""
    total = 0
    total += await save_coverage_from_json()
    for f in sorted(config.DATA_RESULT_DIR.glob("*_report.json")):
        if f.name == "gap_report.json":
            continue
        total += await save_coverage_from_json(f)
    for f in sorted(config.HISTORY_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            date_str = f.stem.split("_")[-1] if "_" in f.stem else f.stem
            try:
                d = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except ValueError:
                d = datetime.utcnow()
            await save_trend_snapshot(d, data, run_id=None)
            total += 1
        except Exception as e:
            logger.warning("trend_import_failed", file=f.name, error=str(e))
    return total
