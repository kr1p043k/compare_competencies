"""ZUN (знать/уметь/навыки) teacher analysis API — read + write over existing schema.

Endpoints:
    GET  /teacher/zun/directions
    GET  /teacher/zun/my-directions
    GET  /teacher/zun/stats
    GET  /teacher/zun/disciplines
    GET  /teacher/zun/disciplines/{discipline_id}
    GET  /teacher/zun/disciplines/name/{discipline_name}
    GET  /teacher/zun/search
    GET  /teacher/zun/search/semantic
    GET  /teacher/zun/filter
    GET  /teacher/zun/competencies/{competency_id}/skills
    GET  /teacher/zun/competencies/{competency_id}/coverage
    GET  /teacher/zun/analyze/results
    POST /teacher/zun/competencies/{competency_id}/entries
    PATCH /teacher/zun/entries/{ksa_id}
    DELETE /teacher/zun/entries/{ksa_id}
    POST /teacher/zun/analyze
    GET  /teacher/zun/analyze/status/{run_id}
    POST /teacher/zun/import/{dir_code}

All read endpoints are SELECT-only. Write endpoints (G1-G3, H1, H5) use the same
schema constraints as src/cli/fix_rpd_data.py and src/pipeline/db_writer.py —
no migrations are required.
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.api_pkg.routers.auth import require_any_role
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["zun"], dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
limiter = Limiter(key_func=get_remote_address)

_DIR_CODE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
KSA_TYPES = ("knowledge", "abilities", "skills")
MATCH_TYPES = ("exact", "fuzzy", "stem", "substring", "semantic", "explicit")
CATEGORIES = ("УК", "ОПК", "ПК", "ППК", "ИП", "ВПК")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULT_TEACHER = PROJECT_ROOT / "data" / "result" / "teacher"


class ZUNIn(BaseModel):
    ksa_type: str
    text: str


class ZUNPatch(BaseModel):
    text: str


# ---------- helpers ----------


def _validate_dir_code(dir_code: str) -> None:
    if not _DIR_CODE_RE.match(dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")


def _validate_uuid(value: str, name: str) -> None:
    if not _UUID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {name} format")


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


def _clean_text(cleaned: str | None, original: str) -> str:
    return cleaned if cleaned else original


async def _direction_ids(pool, dir_code: str) -> list[str]:
    rows = await pool.fetch("SELECT id FROM directions WHERE code=$1", dir_code)
    return [str(r["id"]) for r in rows]


async def _latest_coverage_subq(pool, dir_ids: list[str]) -> dict[str, dict]:
    """{discipline_id: {total_skills, matched_skills, coverage_ratio, analysis_date}} latest date."""
    if not dir_ids:
        return {}
    rows = await pool.fetch(
        """SELECT DISTINCT ON (discipline_id) discipline_id, total_skills,
                  market_matched_skills, coverage_ratio, analysis_date
           FROM coverage_analyses
           WHERE direction_id = ANY($1::uuid[])
           ORDER BY discipline_id, analysis_date DESC""",
        dir_ids,
    )
    return {
        str(r["discipline_id"]): {
            "total_skills": r["total_skills"],
            "matched_skills": r["market_matched_skills"],
            "coverage_ratio": round(float(r["coverage_ratio"]), 4),
            "analysis_date": r["analysis_date"].isoformat() if r["analysis_date"] else None,
        }
        for r in rows
    }


def _coverage_level(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    return "high" if ratio >= 0.5 else "medium" if ratio >= 0.2 else "low"


# semantic cache: {dir_code: (ts, items:list[dict], embs:np.ndarray, max_created)}
_SEM_CACHE: dict[str, tuple[float, list[dict], np.ndarray, str]] = {}
_SEM_TTL = 600


# ---------- A. metadata ----------


@router.get("/teacher/zun/directions")
@limiter.limit("60/minute")
async def zun_directions(request: Request):
    pool = get_pool()
    rows = await pool.fetch(
        """SELECT d.code, d.name, d.profile,
                  COUNT(DISTINCT disc.id) AS disciplines_count
           FROM directions d
           LEFT JOIN disciplines disc ON disc.direction_id = d.id
           GROUP BY d.code, d.name, d.profile
           ORDER BY d.code"""
    )
    ksa_flags = await pool.fetch(
        """SELECT DISTINCT d.code
           FROM directions d
           JOIN disciplines disc ON disc.direction_id = d.id
           JOIN competencies c ON c.discipline_id = disc.id
           JOIN ksa_entries k ON k.competency_id = c.id"""
    )
    has_ksa = {str(r["code"]) for r in ksa_flags}
    ksa_totals = await pool.fetch(
        """SELECT d.code, COUNT(k.id) AS ksa_total
           FROM directions d
           JOIN disciplines disc ON disc.direction_id = d.id
           JOIN competencies c ON c.discipline_id = disc.id
           JOIN ksa_entries k ON k.competency_id = c.id
           GROUP BY d.code"""
    )
    totals = {str(r["code"]): r["ksa_total"] for r in ksa_totals}
    return [
        {
            "dir_code": r["code"],
            "name": r["name"],
            "profile": r["profile"],
            "disciplines_count": r["disciplines_count"],
            "ksa_total": totals.get(r["code"], 0),
            "has_ksa": r["code"] in has_ksa,
        }
        for r in rows
    ]


@router.get("/teacher/zun/my-directions")
@limiter.limit("60/minute")
async def zun_my_directions(request: Request):
    from src.api_pkg.routers.auth import get_current_user
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    role = user.get("r")
    if role == "admin" or role == "teacher":
        resp = await zun_directions(request)
        return [{"dir_code": d["dir_code"], "name": d["name"], "profile": d["profile"]} for d in resp]
    pool = get_pool()
    rows = await pool.fetch(
        "SELECT dir_code FROM user_directions WHERE user_id=$1 ORDER BY dir_code",
        user.get("uid"),
    )
    return [{"dir_code": r["dir_code"]} for r in rows]


# ---------- B. summaries ----------


@router.get("/teacher/zun/stats")
@limiter.limit("60/minute")
async def zun_stats(request: Request, dir_code: str = "09.03.02"):
    _validate_dir_code(dir_code)
    pool = get_pool()
    dir_ids = await _direction_ids(pool, dir_code)
    if not dir_ids:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")

    dirs = await pool.fetch(
        "SELECT code, name, profile FROM directions WHERE code=$1 ORDER BY profile", dir_code
    )
    rows = await pool.fetch(
        """SELECT
                COUNT(DISTINCT disc.id) AS total_disciplines,
                COUNT(DISTINCT CASE WHEN c.parent_id IS NULL THEN c.id END) AS total_competencies,
                COUNT(DISTINCT CASE WHEN c.parent_id IS NOT NULL THEN c.id END) AS total_indicators,
                COUNT(DISTINCT CASE WHEN k.ksa_type='knowledge' THEN k.id END) AS ksa_knowledge,
                COUNT(DISTINCT CASE WHEN k.ksa_type='abilities' THEN k.id END) AS ksa_abilities,
                COUNT(DISTINCT CASE WHEN k.ksa_type='skills' THEN k.id END) AS ksa_skills,
                COUNT(DISTINCT k.id) AS ksa_total,
                COUNT(DISTINCT CASE WHEN k.id IS NOT NULL THEN c.id END) AS competencies_with_ksa,
                COUNT(DISTINCT CASE WHEN k.id IS NOT NULL THEN disc.id END) AS disciplines_with_ksa
           FROM directions d
           JOIN disciplines disc ON disc.direction_id = d.id
           JOIN competencies c ON c.discipline_id = disc.id
           LEFT JOIN ksa_entries k ON k.competency_id = c.id
           WHERE d.id = ANY($1::uuid[])""",
        dir_ids,
    )
    r = rows[0]
    linked = await pool.fetch(
        """SELECT COUNT(*) AS n FROM competency_skills cs
           JOIN competencies c ON c.id = cs.competency_id
           JOIN disciplines disc ON disc.id = c.discipline_id
           WHERE disc.direction_id = ANY($1::uuid[])""",
        dir_ids,
    )
    cov_rows = await pool.fetch(
        """SELECT COALESCE(AVG(coverage_ratio), 0) AS avg, COUNT(*) AS covered,
                  MAX(analysis_date) AS latest
           FROM coverage_analyses WHERE direction_id = ANY($1::uuid[])""",
        dir_ids,
    )
    cr = cov_rows[0]
    ksa_total = r["ksa_total"] or 0
    return {
        "dir_code": dir_code,
        "direction_names": [d["name"] for d in dirs],
        "profiles": [d["profile"] for d in dirs],
        "multiple_profiles": len(dirs) > 1,
        "total_disciplines": r["total_disciplines"],
        "total_competencies": r["total_competencies"],
        "total_indicators": r["total_indicators"],
        "ksa_counts": {
            "knowledge": r["ksa_knowledge"] or 0,
            "abilities": r["ksa_abilities"] or 0,
            "skills": r["ksa_skills"] or 0,
            "total": ksa_total,
        },
        "competencies_with_ksa": r["competencies_with_ksa"] or 0,
        "disciplines_with_ksa": r["disciplines_with_ksa"] or 0,
        "linked_skills": (linked[0]["n"] if linked else 0),
        "coverage": {
            "latest_analysis_date": cr["latest"].isoformat() if cr["latest"] else None,
            "avg_coverage_ratio": round(float(cr["avg"] or 0), 4),
            "covered_disciplines": cr["covered"] or 0,
        },
    }


@router.get("/teacher/zun/disciplines")
@limiter.limit("60/minute")
async def zun_disciplines(
    request: Request,
    dir_code: str = "09.03.02",
    q: str = "",
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    _validate_dir_code(dir_code)
    pool = get_pool()
    dir_ids = await _direction_ids(pool, dir_code)
    if not dir_ids:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")

    params: list = [dir_ids]
    q_filter = ""
    if q:
        q_filter = " AND disc.name ILIKE $" + str(len(params) + 1)
        params.append(f"%{q}%")

    rows = await pool.fetch(
        f"""SELECT disc.id, disc.name, disc.semester, disc.control_form,
                   COUNT(DISTINCT CASE WHEN c.parent_id IS NULL THEN c.id END) AS comps,
                   COUNT(DISTINCT CASE WHEN c.parent_id IS NOT NULL THEN c.id END) AS inds,
                   COUNT(DISTINCT k.id) AS ksa_total,
                   COUNT(DISTINCT CASE WHEN k.ksa_type='knowledge' THEN k.id END) AS k_knowledge,
                   COUNT(DISTINCT CASE WHEN k.ksa_type='abilities' THEN k.id END) AS k_abilities,
                   COUNT(DISTINCT CASE WHEN k.ksa_type='skills' THEN k.id END) AS k_skills,
                   COUNT(DISTINCT cs.id) AS linked_skills
            FROM disciplines disc
            JOIN competencies c ON c.discipline_id = disc.id
            LEFT JOIN ksa_entries k ON k.competency_id = c.id
            LEFT JOIN competency_skills cs ON cs.competency_id = c.id
            WHERE disc.direction_id = ANY($1::uuid[]){q_filter}
            GROUP BY disc.id, disc.name, disc.semester, disc.control_form
            ORDER BY disc.name
            LIMIT $2 OFFSET $3""",
        *params + [limit, offset],
    )
    cov = await _latest_coverage_subq(pool, dir_ids)
    return [
        {
            "id": str(r["id"]),
            "name": r["name"],
            "semester": r["semester"],
            "control_form": r["control_form"],
            "competencies_count": r["comps"],
            "indicators_count": r["inds"],
            "ksa_total": r["ksa_total"],
            "ksa_by_type": {
                "knowledge": r["k_knowledge"],
                "abilities": r["k_abilities"],
                "skills": r["k_skills"],
            },
            "linked_skills": r["linked_skills"],
            "coverage_ratio": cov.get(str(r["id"]), {}).get("coverage_ratio"),
            "coverage_level": _coverage_level(cov.get(str(r["id"]), {}).get("coverage_ratio")),
        }
        for r in rows
    ]


# ---------- C. tree ----------


async def _build_discipline_tree(pool, disc_id: str, dir_code: str | None = None) -> dict:
    disc = await pool.fetchrow(
        """SELECT disc.id, disc.name, disc.semester, disc.control_form,
                  d.code AS direction_code
           FROM disciplines disc
           JOIN directions d ON d.id = disc.direction_id
           WHERE disc.id = $1""",
        disc_id,
    )
    if not disc:
        raise HTTPException(status_code=404, detail="Discipline not found")

    comp_rows = await pool.fetch(
        """SELECT c.id, c.code, c.category, c.name, c.development_level,
                  c.sort_order, c.parent_id
           FROM competencies c
           WHERE c.discipline_id = $1
           ORDER BY c.sort_order, c.code""",
        disc_id,
    )
    comps: list[dict] = []
    root_by_id: dict[str, dict] = {}
    ind_by_id: dict[str, dict] = {}
    all_ids: list[str] = []
    for r in comp_rows:
        node: dict[str, Any] = {
            "id": str(r["id"]),
            "code": r["code"],
            "category": r["category"],
            "name": r["name"],
            "development_level": r["development_level"],
            "sort_order": r["sort_order"],
            "ksa": {"knowledge": [], "abilities": [], "skills": []},
            "linked_skills": [],
        }
        if r["parent_id"] is None:
            node["indicators"] = []
            root_by_id[str(r["id"])] = node
            comps.append(node)
        else:
            ind_by_id[str(r["id"])] = node
        all_ids.append(str(r["id"]))

    if all_ids:
        ksa_rows = await pool.fetch(
            """SELECT competency_id, ksa_type, original_text, cleaned_text
               FROM ksa_entries WHERE competency_id = ANY($1::uuid[])
               ORDER BY sort_order""",
            all_ids,
        )
        for r in ksa_rows:
            cid = str(r["competency_id"])
            target = root_by_id.get(cid) or ind_by_id.get(cid)
            if target is not None:
                target["ksa"][r["ksa_type"]].append(_clean_text(r["cleaned_text"], r["original_text"]))

        cs_rows = await pool.fetch(
            """SELECT cs.competency_id, s.name AS skill_name, cs.match_type
               FROM competency_skills cs
               JOIN skills s ON s.id = cs.skill_id
               WHERE cs.competency_id = ANY($1::uuid[])
               ORDER BY s.name""",
            all_ids,
        )
        for r in cs_rows:
            cid = str(r["competency_id"])
            target = root_by_id.get(cid) or ind_by_id.get(cid)
            if target is not None:
                target["linked_skills"].append({"name": r["skill_name"], "match_type": r["match_type"]})

    # attach indicators to their parents
    for r in comp_rows:
        if r["parent_id"] is not None:
            parent = root_by_id.get(str(r["parent_id"]))
            if parent is not None:
                parent["indicators"].append(ind_by_id[str(r["id"])])

    # coverage for the discipline (latest analysis)
    cov_row = await pool.fetchrow(
        """SELECT total_skills, market_matched_skills, coverage_ratio, analysis_date
           FROM coverage_analyses
           WHERE discipline_id = $1
           ORDER BY analysis_date DESC LIMIT 1""",
        disc_id,
    )
    coverage = None
    if cov_row:
        coverage = {
            "total_skills": cov_row["total_skills"],
            "matched_skills": cov_row["market_matched_skills"],
            "coverage_ratio": round(float(cov_row["coverage_ratio"]), 4),
            "coverage_level": _coverage_level(float(cov_row["coverage_ratio"])),
            "analysis_date": cov_row["analysis_date"].isoformat() if cov_row["analysis_date"] else None,
        }
    return {
        "id": str(disc["id"]),
        "name": disc["name"],
        "semester": disc["semester"],
        "control_form": disc["control_form"],
        "direction_code": disc["direction_code"],
        "coverage": coverage,
        "competencies": comps,
    }


@router.get("/teacher/zun/disciplines/{discipline_id}")
@limiter.limit("60/minute")
async def zun_discipline_tree(request: Request, discipline_id: str):
    _validate_uuid(discipline_id, "discipline_id")
    return await _build_discipline_tree(get_pool(), discipline_id)


@router.get("/teacher/zun/disciplines/name/{discipline_name:path}")
@limiter.limit("60/minute")
async def zun_discipline_by_name(request: Request, discipline_name: str, dir_code: str = "09.03.02"):
    _validate_dir_code(dir_code)
    pool = get_pool()
    dir_ids = await _direction_ids(pool, dir_code)
    if not dir_ids:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")
    row = await pool.fetchrow(
        """SELECT id FROM disciplines
           WHERE direction_id = ANY($1::uuid[]) AND name = $2
           ORDER BY name LIMIT 1""",
        dir_ids, discipline_name,
    )
    if not row:
        # fuzzy: drop spaces, lowercase
        norm = discipline_name.replace(" ", "").lower()
        rows = await pool.fetch(
            """SELECT id, name FROM disciplines
               WHERE direction_id = ANY($1::uuid[])
               ORDER BY name""",
            dir_ids,
        )
        for rr in rows:
            if rr["name"].replace(" ", "").lower() == norm:
                row = rr
                break
    if not row:
        raise HTTPException(status_code=404, detail=f"Discipline '{discipline_name}' not found")
    return await _build_discipline_tree(pool, str(row["id"]))


# ---------- D. search ----------


@router.get("/teacher/zun/search")
@limiter.limit("120/minute")
async def zun_search(
    request: Request,
    q: str,
    dir_code: str = "09.03.02",
    ksa_type: str = "",
    discipline_id: str = "",
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="q must be at least 2 characters")
    _validate_dir_code(dir_code)
    pool = get_pool()
    dir_ids = await _direction_ids(pool, dir_code)
    if not dir_ids:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")

    params: list = [dir_ids, f"%{q.strip()}%"]
    filters = " AND disc.direction_id = ANY($1::uuid[]) AND (k.original_text ILIKE $2 OR k.cleaned_text ILIKE $2)"
    if ksa_type:
        if ksa_type not in KSA_TYPES:
            raise HTTPException(status_code=400, detail="Invalid ksa_type")
        filters += " AND k.ksa_type = $" + str(len(params) + 1)
        params.append(ksa_type)
    if discipline_id:
        _validate_uuid(discipline_id, "discipline_id")
        filters += " AND disc.id = $" + str(len(params) + 1)
        params.append(discipline_id)

    rows = await pool.fetch(
        f"""SELECT k.id, k.ksa_type, k.original_text, k.cleaned_text,
                   c.id AS competence_id, c.code AS competence_code,
                   c.parent_id, parent.code AS parent_code,
                   disc.id AS discipline_id, disc.name AS discipline_name
            FROM ksa_entries k
            JOIN competencies c ON c.id = k.competency_id
            LEFT JOIN competencies parent ON parent.id = c.parent_id
            JOIN disciplines disc ON disc.id = c.discipline_id
            WHERE 1=1{filters}
            ORDER BY disc.name, c.code, k.sort_order
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}""",
        *params + [limit, offset],
    )
    return [
        {
            "ksa_id": str(r["id"]),
            "ksa_type": r["ksa_type"],
            "original_text": r["original_text"],
            "cleaned_text": _clean_text(r["cleaned_text"], r["original_text"]),
            "competence_id": str(r["competence_id"]),
            "competence_code": r["competence_code"],
            "is_indicator": r["parent_id"] is not None,
            "parent_code": r["parent_code"],
            "discipline_id": str(r["discipline_id"]),
            "discipline_name": r["discipline_name"],
        }
        for r in rows
    ]


async def _semantic_cache_items(pool, dir_ids: list[str]) -> tuple[list[dict], np.ndarray]:
    rows = await pool.fetch(
        """SELECT k.id, k.ksa_type, k.original_text, k.created_at,
                  c.id AS competence_id, c.code AS competence_code, c.parent_id,
                  parent.code AS parent_code, disc.id AS discipline_id, disc.name AS discipline_name
           FROM ksa_entries k
           JOIN competencies c ON c.id = k.competency_id
           LEFT JOIN competencies parent ON parent.id = c.parent_id
           JOIN disciplines disc ON disc.id = c.discipline_id
           WHERE disc.direction_id = ANY($1::uuid[])
           ORDER BY disc.name, c.code, k.sort_order""",
        dir_ids,
    )
    items = [
        {
            "ksa_id": str(r["id"]),
            "ksa_type": r["ksa_type"],
            "original_text": r["original_text"],
            "competence_id": str(r["competence_id"]),
            "competence_code": r["competence_code"],
            "is_indicator": r["parent_id"] is not None,
            "parent_code": r["parent_code"],
            "discipline_id": str(r["discipline_id"]),
            "discipline_name": r["discipline_name"],
        }
        for r in rows
    ]
    if not items:
        return items, np.zeros((0, 0))
    try:
        from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory
        prov = EmbeddingProviderFactory.get()
        texts = [it["original_text"] for it in items]
        embs = prov.encode(texts, show_progress_bar=False)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        return items, embs
    except Exception as exc:
        logger.warning("zun_semantic_encode_failed", error=str(exc))
        return items, np.zeros((0, 0))


@router.get("/teacher/zun/search/semantic")
@limiter.limit("60/minute")
async def zun_search_semantic(
    request: Request,
    q: str,
    dir_code: str = "09.03.02",
    ksa_type: str = "",
    limit: int = Query(20, ge=1, le=200),
):
    if len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="q must be at least 2 characters")
    _validate_dir_code(dir_code)
    pool = get_pool()
    dir_ids = await _direction_ids(pool, dir_code)
    if not dir_ids:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")

    max_created = await pool.fetchval(
        """SELECT MAX(k.created_at)::text
           FROM ksa_entries k
           JOIN competencies c ON c.id = k.competency_id
           JOIN disciplines disc ON disc.id = c.discipline_id
           WHERE disc.direction_id = ANY($1::uuid[])""",
        dir_ids,
    ) or ""

    cache = _SEM_CACHE.get(dir_code)
    if cache is not None and (time.time() - cache[0]) <= _SEM_TTL and cache[3] == max_created:
        _, items, embs, _ = cache
    else:
        items, embs = await _semantic_cache_items(pool, dir_ids)
        if embs.shape[0] > 0:
            _SEM_CACHE[dir_code] = (time.time(), items, embs, max_created)
        else:
            _SEM_CACHE.pop(dir_code, None)
            return {"mode": "fallback_substring", "matches": await _fallback_semantic(pool, q, dir_code, ksa_type, limit), "note": "embeddings unavailable"}

    try:
        from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory
        prov = EmbeddingProviderFactory.get()
        qemb = prov.encode([q.strip()], show_progress_bar=False)[0]
        qn = qemb / (np.linalg.norm(qemb) + 1e-9)
    except Exception as exc:
        logger.warning("zun_semantic_query_encode_failed", error=str(exc))
        return {"mode": "fallback_substring", "matches": await _fallback_semantic(pool, q, dir_code, ksa_type, limit), "note": "embeddings unavailable"}

    sims = embs @ qn
    order = np.argsort(sims)[::-1][:limit]
    matches = []
    for idx in order:
        item = items[int(idx)]
        if ksa_type and item["ksa_type"] != ksa_type:
            continue
        matches.append({**item, "similarity": round(float(sims[idx]), 4)})
        if len(matches) >= limit:
            break
    if not matches and ksa_type:
        # filtered everything out — return fallback to be helpful
        return {"mode": "fallback_substring", "matches": await _fallback_semantic(pool, q, dir_code, ksa_type, limit), "note": "no semantic matches for filter"}
    return {"mode": "semantic", "query": q.strip(), "matches": matches}


async def _fallback_semantic(pool, q: str, dir_code: str, ksa_type: str, limit: int) -> list[dict]:
    params: list = [f"%{q.strip()}%"]
    filters = " AND (k.original_text ILIKE $1 OR k.cleaned_text ILIKE $1)"
    if ksa_type:
        filters += " AND k.ksa_type = $" + str(len(params) + 1)
        params.append(ksa_type)
    rows = await pool.fetch(
        f"""SELECT k.id, k.ksa_type, k.original_text, c.code AS competence_code,
                   c.parent_id, disc.name AS discipline_name
            FROM ksa_entries k
            JOIN competencies c ON c.id = k.competency_id
            JOIN disciplines disc ON disc.id = c.discipline_id
            JOIN directions d ON d.id = disc.direction_id
            WHERE d.code = $2{filters}
            ORDER BY disc.name, k.sort_order
            LIMIT $3""",
        *params + [dir_code, limit],
    )
    return [
        {
            "ksa_id": str(r["id"]),
            "ksa_type": r["ksa_type"],
            "original_text": r["original_text"],
            "competence_code": r["competence_code"],
            "is_indicator": r["parent_id"] is not None,
            "discipline_name": r["discipline_name"],
            "similarity": None,
        }
        for r in rows
    ]


# ---------- E. filter ----------


@router.get("/teacher/zun/filter")
@limiter.limit("120/minute")
async def zun_filter(
    request: Request,
    dir_code: str = "09.03.02",
    category: str = "",
    code_prefix: str = "",
    competence_id: str = "",
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    _validate_dir_code(dir_code)
    if category and category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    pool = get_pool()
    dir_ids = await _direction_ids(pool, dir_code)
    if not dir_ids:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")

    params: list = [dir_ids]
    filters = " AND disc.direction_id = ANY($1::uuid[])"
    if category:
        filters += " AND c.category = $" + str(len(params) + 1)
        params.append(category)
    if code_prefix:
        filters += " AND c.code ILIKE $" + str(len(params) + 1)
        params.append(code_prefix + "%")
    if competence_id:
        _validate_uuid(competence_id, "competence_id")
        filters += " AND c.id = $" + str(len(params) + 1)
        params.append(competence_id)

    rows = await pool.fetch(
        f"""SELECT c.id, c.code, c.category, c.name, c.development_level,
                   disc.id AS discipline_id, disc.name AS discipline_name,
                   COUNT(k.id) AS ksa_total
            FROM competencies c
            JOIN disciplines disc ON disc.id = c.discipline_id
            LEFT JOIN ksa_entries k ON k.competency_id = c.id
            WHERE 1=1{filters}
            GROUP BY c.id, c.code, c.category, c.name, c.development_level,
                     disc.id, disc.name
            ORDER BY disc.name, c.code
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}""",
        *params + [limit, offset],
    )
    return [
        {
            "competence_id": str(r["id"]),
            "code": r["code"],
            "category": r["category"],
            "name": r["name"],
            "development_level": r["development_level"],
            "discipline_id": str(r["discipline_id"]),
            "discipline_name": r["discipline_name"],
            "ksa_total": r["ksa_total"],
        }
        for r in rows
    ]


# ---------- F. competency -> market ----------


@router.get("/teacher/zun/competencies/{competency_id}/skills")
@limiter.limit("120/minute")
async def zun_competency_skills(request: Request, competency_id: str, match_type: str = ""):
    _validate_uuid(competency_id, "competency_id")
    if match_type and match_type not in MATCH_TYPES:
        raise HTTPException(status_code=400, detail="Invalid match_type")
    pool = get_pool()
    comp = await pool.fetchrow("SELECT id FROM competencies WHERE id=$1", competency_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Competency not found")

    params: list = [competency_id]
    filters = " WHERE cs.competency_id = $1"
    if match_type:
        filters += " AND cs.match_type = $" + str(len(params) + 1)
        params.append(match_type)
    rows = await pool.fetch(
        f"""SELECT s.id AS skill_id, s.name AS skill_name, cs.match_type,
                   cs.source_text, cs.ksa_type
            FROM competency_skills cs
            JOIN skills s ON s.id = cs.skill_id
            {filters}
            ORDER BY s.name""",
        *params,
    )
    return [
        {
            "skill_id": str(r["skill_id"]),
            "skill_name": r["skill_name"],
            "match_type": r["match_type"],
            "source_text": r["source_text"],
            "ksa_type": r["ksa_type"],
        }
        for r in rows
    ]


@router.get("/teacher/zun/competencies/{competency_id}/coverage")
@limiter.limit("120/minute")
async def zun_competency_coverage(request: Request, competency_id: str):
    _validate_uuid(competency_id, "competency_id")
    pool = get_pool()
    comp = await pool.fetchrow(
        """SELECT c.id, c.code, disc.id AS discipline_id, disc.name AS discipline_name
           FROM competencies c
           JOIN disciplines disc ON disc.id = c.discipline_id
           WHERE c.id = $1""",
        competency_id,
    )
    if not comp:
        raise HTTPException(status_code=404, detail="Competency not found")
    cov = await pool.fetchrow(
        """SELECT total_skills, market_matched_skills, coverage_ratio, analysis_date
           FROM coverage_analyses
           WHERE discipline_id = $1
           ORDER BY analysis_date DESC LIMIT 1""",
        comp["discipline_id"],
    )
    if not cov:
        return {
            "competence_id": competency_id,
            "competence_code": comp["code"],
            "discipline_id": str(comp["discipline_id"]),
            "discipline_name": comp["discipline_name"],
            "coverage": None,
        }
    return {
        "competence_id": competency_id,
        "competence_code": comp["code"],
        "discipline_id": str(comp["discipline_id"]),
        "discipline_name": comp["discipline_name"],
        "coverage": {
            "total_skills": cov["total_skills"],
            "matched_skills": cov["market_matched_skills"],
            "coverage_ratio": round(float(cov["coverage_ratio"]), 4),
            "analysis_date": cov["analysis_date"].isoformat() if cov["analysis_date"] else None,
        },
    }


# ---------- H3. results from disk ----------


@router.get("/teacher/zun/analyze/results")
@limiter.limit("60/minute")
async def zun_analyze_results(
    request: Request,
    dir_code: str = "09.03.02",
    discipline_id: str = "",
):
    _validate_dir_code(dir_code)
    base = (RESULT_TEACHER / dir_code).resolve()
    if RESULT_TEACHER.resolve() not in base.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"No analysis results for {dir_code}")
    if discipline_id:
        _validate_uuid(discipline_id, "discipline_id")
        if not base.resolve().is_relative_to(RESULT_TEACHER.resolve()):
            raise HTTPException(status_code=400, detail="Invalid path")
        disc = await get_pool().fetchrow(
            "SELECT name FROM disciplines WHERE id=$1", discipline_id
        )
        if not disc:
            raise HTTPException(status_code=404, detail="Discipline not found")
        safe = _safe_filename(disc["name"])
        fpath = (base / safe / (safe + ".json")).resolve()
        if not fpath.exists():
            raise HTTPException(status_code=404, detail=f"Analysis not found for discipline '{disc['name']}'")
        return json.loads(fpath.read_text(encoding="utf-8"))
    summary = (base / "_summary.json").resolve()
    if not summary.exists():
        raise HTTPException(status_code=404, detail=f"Summary not found for {dir_code}")
    return json.loads(summary.read_text(encoding="utf-8"))


# ---------- G. write KSA (no migrations) ----------


@router.post("/teacher/zun/competencies/{competency_id}/entries", status_code=201)
@limiter.limit("30/minute")
async def zun_add_entry(request: Request, competency_id: str, body: ZUNIn):
    _validate_uuid(competency_id, "competency_id")
    if body.ksa_type not in KSA_TYPES:
        raise HTTPException(status_code=400, detail="Invalid ksa_type")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text too long (max 2000)")

    pool = get_pool()
    comp = await pool.fetchrow(
        """SELECT c.id FROM competencies c
           JOIN disciplines disc ON disc.id = c.discipline_id
           JOIN directions d ON d.id = disc.direction_id
           WHERE c.id = $1
           LIMIT 1""",
        competency_id,
    )
    if not comp:
        raise HTTPException(status_code=404, detail="Competency not found")

    dup = await pool.fetchrow(
        """SELECT 1 FROM ksa_entries
           WHERE competency_id = $1 AND ksa_type = $2 AND original_text = $3""",
        competency_id, body.ksa_type, text,
    )
    if dup:
        raise HTTPException(status_code=409, detail="KSA entry already exists")

    sort = await pool.fetchval(
        "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM ksa_entries WHERE competency_id = $1",
        competency_id,
    )
    ksa_id = await pool.fetchval(
        """INSERT INTO ksa_entries (competency_id, ksa_type, original_text, cleaned_text, sort_order, parse_version_id)
           VALUES ($1, $2::ksa_type, $3, $3, $4, NULL)
           RETURNING id""",
        competency_id, body.ksa_type, text, sort,
    )
    logger.info("zun_entry_added", ksa_id=str(ksa_id), competency_id=competency_id, ksa_type=body.ksa_type)
    return {"ksa_id": str(ksa_id), "ksa_type": body.ksa_type, "text": text}


@router.patch("/teacher/zun/entries/{ksa_id}")
@limiter.limit("30/minute")
async def zun_patch_entry(request: Request, ksa_id: str, body: ZUNPatch):
    _validate_uuid(ksa_id, "ksa_id")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text too long (max 2000)")
    pool = get_pool()
    row = await pool.fetchrow(
        """UPDATE ksa_entries SET original_text = $1, cleaned_text = $1
           WHERE id = $2 RETURNING id, ksa_type""",
        text, ksa_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="KSA entry not found")
    logger.info("zun_entry_updated", ksa_id=ksa_id)
    return {"ksa_id": ksa_id, "ksa_type": row["ksa_type"], "text": text}


@router.delete("/teacher/zun/entries/{ksa_id}")
@limiter.limit("30/minute")
async def zun_delete_entry(request: Request, ksa_id: str):
    _validate_uuid(ksa_id, "ksa_id")
    pool = get_pool()
    res = await pool.execute("DELETE FROM ksa_entries WHERE id = $1", ksa_id)
    if res == "DELETE 0":
        raise HTTPException(status_code=404, detail="KSA entry not found")
    logger.info("zun_entry_deleted", ksa_id=ksa_id)
    return {"status": "deleted"}


# ---------- H1/H2. analyze ----------


@router.post("/teacher/zun/analyze")
@limiter.limit("2/minute")
async def zun_analyze(request: Request, background_tasks: BackgroundTasks, dir_code: str = "09.03.02"):
    _validate_dir_code(dir_code)

    async def _run():
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "src.cli", "teacher-analysis",
                "--direction", dir_code,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=PROJECT_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=2400)
            logger.info("zun_teacher_analysis_done", returncode=proc.returncode,
                        stderr=stderr.decode("utf-8", errors="ignore")[-500:])
        except asyncio.TimeoutError:
            logger.error("zun_teacher_analysis_timeout", dir_code=dir_code)
            if proc and proc.returncode is None:
                proc.kill()
        except Exception as exc:
            logger.error("zun_teacher_analysis_error", error=str(exc))

    background_tasks.add_task(_run)
    return {"status": "started", "dir_code": dir_code}


@router.get("/teacher/zun/analyze/status/{run_id}")
@limiter.limit("120/minute")
async def zun_analyze_status(request: Request, run_id: str):
    _validate_uuid(run_id, "run_id")
    pool = get_pool()
    row = await pool.fetchrow(
        """SELECT id, action, status, started_at, completed_at, error_message, stats
           FROM pipeline_runs WHERE id = $1""",
        run_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": str(row["id"]),
        "action": row["action"],
        "status": row["status"],
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
        "error": row["error_message"],
        "stats": row["stats"] or {},
    }


# ---------- H5. import from KRM JSON (idempotent, mirror of fix_rpd_data) ----------


async def _import_direction(pool, dir_code: str, dry_run: bool) -> dict:
    krm_path = config.REFERENCE_DIR / f"krm_disciplines_{dir_code}.json"
    if not krm_path.exists():
        raise HTTPException(status_code=404, detail=f"KRM file not found for {dir_code}")
    krm = json.loads(krm_path.read_text(encoding="utf-8"))
    container = krm.get(dir_code)
    if container is None or not isinstance(container, dict) or "disciplines" not in container:
        container = next(
            (v for v in krm.values() if isinstance(v, dict) and "disciplines" in v), None
        )
    if container is None:
        raise HTTPException(status_code=400, detail="No disciplines container in KRM file")
    disciplines_raw = container.get("disciplines", {}) or {}

    dirs = await pool.fetch("SELECT id, name FROM directions WHERE code=$1", dir_code)
    if not dirs:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found in DB")
    dir_ids = [str(d["id"]) for d in dirs]
    disc_rows = await pool.fetch(
        "SELECT id, name FROM disciplines WHERE direction_id = ANY($1::uuid[])", dir_ids
    )
    disc_map = {r["name"]: str(r["id"]) for r in disc_rows}
    comp_rows = await pool.fetch(
        """SELECT id, discipline_id, code FROM competencies
           WHERE discipline_id = ANY($1::uuid[])""",
        list(disc_map.values()),
    )
    comp_map = {(str(r["discipline_id"]), r["code"]): str(r["id"]) for r in comp_rows}

    inserted = skipped_dup = skipped_gap = dropped = 0
    disciplines_processed = 0
    for disc_name, disc_data in sorted(disciplines_raw.items()):
        disc_id = disc_map.get(disc_name)
        if not disc_id:
            continue
        ksa_data = disc_data.get("ksa", {}) or {}
        disciplines_processed += 1
        for comp_code, sections in ksa_data.items():
            comp_id = comp_map.get((disc_id, comp_code))
            if not comp_id:
                continue
            for kt in ("knowledge", "abilities", "skills"):
                texts = sections.get(kt, []) or []
                # gap-fill: if category already has entries for this competency in DB — skip
                existing_type = await pool.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM ksa_entries WHERE competency_id=$1 AND ksa_type=$2::ksa_type)",
                    comp_id, kt,
                )
                if existing_type and not dry_run:
                    skipped_gap += len(texts)
                    continue
                for raw in texts:
                    text = (raw or "").strip()
                    if not text:
                        continue
                    dup = await pool.fetchval(
                        """SELECT EXISTS (SELECT 1 FROM ksa_entries
                                         WHERE competency_id=$1 AND ksa_type=$2::ksa_type AND original_text=$3)""",
                        comp_id, kt, text,
                    )
                    if dup:
                        skipped_dup += 1
                        continue
                    if dry_run:
                        inserted += 1
                        continue
                    sort = await pool.fetchval(
                        "SELECT COALESCE(MAX(sort_order),0)+1 FROM ksa_entries WHERE competency_id=$1",
                        comp_id,
                    )
                    await pool.execute(
                        """INSERT INTO ksa_entries (competency_id, ksa_type, original_text, cleaned_text, sort_order, parse_version_id)
                           VALUES ($1, $2::ksa_type, $3, $3, $4, NULL)""",
                        comp_id, kt, text, sort,
                    )
                    inserted += 1

    return {
        "dir_code": dir_code,
        "inserted": inserted,
        "skipped_dups": skipped_dup,
        "skipped_gap": skipped_gap,
        "disciplines_processed": disciplines_processed,
    }


@router.post("/teacher/zun/import/{dir_code}")
@limiter.limit("5/minute")
async def zun_import(request: Request, dir_code: str, dry_run: bool = False):
    _validate_dir_code(dir_code)
    pool = get_pool()
    return await _import_direction(pool, dir_code, dry_run=dry_run)