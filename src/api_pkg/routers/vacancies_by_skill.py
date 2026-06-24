"""GET /api/vacancies/by-skill — список вакансий с указанным навыком."""
import json

from fastapi import APIRouter, Query
from sqlalchemy import text

from src.database import async_session_factory

router = APIRouter(tags=["vacancies"])


@router.get("/vacancies/by-skill")
async def get_vacancies_by_skill(
    skill: str = Query(...),
    snapshot_date: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
):
    if len(skill) < 3:
        return {"vacancies": [], "total": 0}

    skill_arr = json.dumps([skill])

    async with async_session_factory() as session:
        rows = await session.execute(
            text("""
                SELECT v.alternate_url, v.name, v.employer_name
                FROM vacancies v
                JOIN trend_snapshots ts ON ts.pipeline_run_id = v.pipeline_run_id
                WHERE ts.snapshot_date = :date
                  AND v.parsed_skills @> :skill_arr::jsonb
                LIMIT :limit
            """),
            {"date": snapshot_date, "skill_arr": skill_arr, "limit": limit},
        )
        return {
            "vacancies": [
                {"url": r[0], "title": r[1], "employer": r[2]} for r in rows
            ],
            "total": rows.rowcount,
        }
