"""Vacancies: list, detail, stats — DB-backed."""

import json
import structlog
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.api_responses import (
    VacanciesResponse,
    VacancyDetailResponse,
    VacancyStatsResponse,
)

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["vacancies"])
limiter = Limiter(key_func=get_remote_address)


def _classify_experience(exp_str: str | None, name: str) -> str:
    if exp_str:
        el = exp_str.lower()
        if "junior" in el or "less1" in el or "no_experience" in el:
            return "junior"
        if "senior" in el or "morethan10" in el:
            return "senior"
        if "between3and6" in el or "between1and3" in el:
            return "middle"
    nl = name.lower()
    if "junior" in nl or "младший" in nl:
        return "junior"
    if "senior" in nl or "старший" in nl or "ведущий" in nl:
        return "senior"
    return "middle"


async def _get_db_pool():
    from src.db import get_pool
    return get_pool()


@router.get("/vacancies", response_model=VacanciesResponse)
@limiter.limit("60/minute")
async def get_vacancies(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Количество вакансий"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    experience: str | None = Query(
        None, description="Фильтр по опыту: junior, middle, senior"
    ),
    search: str | None = Query(None, description="Поиск по названию"),
    months: int | None = Query(None, ge=1, le=24, description="Период в месяцах"),
):
    pool = await _get_db_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    conditions: list[str] = []
    params: list = []

    if experience:
        exp_lower = experience.lower()
        conditions.append(
            f"(LOWER(v.experience) LIKE '%' || ${len(params) + 1} || '%'"
            f" OR LOWER(v.name) LIKE '%' || ${len(params) + 1} || '%')"
        )
        params.append(exp_lower)

    if search:
        conditions.append(f"LOWER(v.name) LIKE '%' || ${len(params) + 1} || '%'")
        params.append(search.lower())

    if months:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)
        conditions.append(f"v.published_at >= ${len(params) + 1}")
        params.append(cutoff)

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    count_sql = f"SELECT COUNT(*) FROM vacancies v WHERE {where_clause}"
    total = await pool.fetchval(count_sql, *params)

    sql = f"""
        SELECT v.hh_id, v.name, v.experience, v.salary_from, v.salary_to,
               v.salary_currency, v.employer_name, v.area_name,
               v.snippet_requirement, v.snippet_responsibility,
               v.published_at, v.alternate_url, v.parsed_skills, v.key_skills
        FROM vacancies v
        WHERE {where_clause}
        ORDER BY v.published_at DESC NULLS LAST
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
    """
    rows = await pool.fetch(sql, *params, limit, offset)

    items = []
    for r in rows:
        parsed = r["parsed_skills"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed) if parsed else []
        skills = (parsed[:10] if isinstance(parsed, list) else [])

        exp = _classify_experience(r["experience"], r["name"] or "")

        snippet = {}
        if r["snippet_requirement"] or r["snippet_responsibility"]:
            snippet = {
                "requirement": r["snippet_requirement"] or "",
                "responsibility": r["snippet_responsibility"] or "",
            }

        items.append({
            "id": str(r["hh_id"]),
            "name": r["name"] or "Без названия",
            "experience": exp,
            "salary_from": r["salary_from"],
            "salary_to": r["salary_to"],
            "salary_currency": r["salary_currency"] or "RUR",
            "employer_name": r["employer_name"] or "Не указано",
            "area": r["area_name"] or "Не указано",
            "published_at": r["published_at"].isoformat() if r["published_at"] else None,
            "alternate_url": r["alternate_url"],
            "skills": skills,
            "snippet": snippet,
        })

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
    }


@router.get("/vacancies/info")
async def get_vacancies_info():
    from src.api_pkg import deps
    pool = await _get_db_pool()

    info = {"count": 0, "file_modified": None, "date_range": None, "load_error": deps.vacancy_load_error}

    if pool:
        try:
            total = await pool.fetchval(
                "SELECT COUNT(*) FROM vacancies WHERE parsed_skills IS NOT NULL AND parsed_skills::text != '[]'"
            )
            info["total_vacancies"] = total or 0
            info["count"] = total or 0

            row = await pool.fetchrow(
                "SELECT MAX(published_at) AS max_p, MIN(published_at) AS min_p FROM vacancies WHERE published_at IS NOT NULL"
            )
            if row and row["min_p"]:
                info["date_range"] = {
                    "from": row["min_p"].isoformat()[:10],
                    "to": row["max_p"].isoformat()[:10],
                }

            last_run = await pool.fetchrow(
                "SELECT completed_at, stats FROM pipeline_runs "
                "WHERE action = 'full-cycle' AND status = 'completed' "
                "ORDER BY completed_at DESC LIMIT 1"
            )
            info["last_updated"] = str(last_run["completed_at"]) if last_run else None
            info["last_pipeline_stats"] = last_run["stats"] if last_run else None
            info["db_available"] = True
        except Exception as exc:
            logger.warning("vacancies_info_db_failed", error=str(exc))
            info["db_available"] = False
            info["total_vacancies"] = 0
    else:
        info["db_available"] = False

    return info


@router.get("/vacancies/{vacancy_id}", response_model=VacancyDetailResponse)
@limiter.limit("60/minute")
async def get_vacancy_detail(
    request: Request,
    vacancy_id: str,
):
    pool = await _get_db_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        hh_id = int(vacancy_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid vacancy ID format")

    row = await pool.fetchrow(
        """SELECT hh_id, name, description, experience, salary_from, salary_to,
                  salary_currency, employer_name, employer_id, area_name,
                  snippet_requirement, snippet_responsibility,
                  published_at, alternate_url, parsed_skills, key_skills
           FROM vacancies WHERE hh_id = $1""",
        hh_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Вакансия не найдена")

    def _load_jsonb(val):
        if isinstance(val, str):
            return json.loads(val) if val else []
        return list(val) if isinstance(val, list) else []

    parsed = _load_jsonb(row["parsed_skills"])
    skills = parsed[:20]

    ks = _load_jsonb(row["key_skills"])
    key_skills = ks

    snippet = {}
    if row["snippet_requirement"] or row["snippet_responsibility"]:
        snippet = {
            "requirement": row["snippet_requirement"] or "",
            "responsibility": row["snippet_responsibility"] or "",
        }

    return {
        "id": str(row["hh_id"]),
        "name": row["name"],
        "description": row["description"] or "",
        "experience": {"id": row["experience"], "name": row["experience"]} if row["experience"] else None,
        "salary": {
            "from": row["salary_from"],
            "to": row["salary_to"],
            "currency": row["salary_currency"] or "RUR",
        } if row["salary_from"] or row["salary_to"] else None,
        "employer": {
            "id": row["employer_id"],
            "name": row["employer_name"] or "Не указано",
        } if row["employer_name"] else None,
        "area": {"id": None, "name": row["area_name"]} if row["area_name"] else None,
        "published_at": row["published_at"].isoformat() if row["published_at"] else None,
        "alternate_url": row["alternate_url"],
        "skills": skills,
        "schedule": None,
        "employment": None,
        "key_skills": key_skills,
        "snippet": snippet,
    }


@router.get("/vacancies/stats/summary", response_model=VacancyStatsResponse)
@limiter.limit("30/minute")
async def get_vacancies_stats(
    request: Request,
):
    pool = await _get_db_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    total = await pool.fetchval("SELECT COUNT(*) FROM vacancies") or 0

    rows = await pool.fetch("SELECT experience, salary_from, salary_to FROM vacancies")
    junior = middle = senior = 0
    salaries = []
    for r in rows:
        exp = _classify_experience(r["experience"], "")
        if exp == "junior":
            junior += 1
        elif exp == "senior":
            senior += 1
        else:
            middle += 1
        if r["salary_from"]:
            salaries.append(r["salary_from"])
        if r["salary_to"]:
            salaries.append(r["salary_to"])

    avg_salary = sum(salaries) / len(salaries) if salaries else 0
    return {
        "total": total,
        "by_experience": {"junior": junior, "middle": middle, "senior": senior},
        "salary": {
            "average": round(avg_salary, 0),
            "min": min(salaries) if salaries else 0,
            "max": max(salaries) if salaries else 0,
            "count": len(salaries),
        },
    }
