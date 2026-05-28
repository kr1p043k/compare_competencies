"""Taxonomy: coverage, professions, KRM."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import Err, Ok
from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
from src.models.api_responses import (
    KRMCoverageResponse,
    ProfessionDetailResponse,
    ProfessionsResponse,
    TaxonomyCoverageResponse,
)

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["taxonomy"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/taxonomy/coverage", response_model=TaxonomyCoverageResponse)
@limiter.limit("20/minute")
async def taxonomy_coverage(
    request: Request,
    taxonomy_instance: SkillTaxonomy | None = Depends(deps.get_taxonomy),
):
    if not taxonomy_instance:
        raise HTTPException(status_code=503, detail="Таксономия не загружена")
    match taxonomy_instance.get_all_categories():
        case Ok(categories):
            cat_ids = categories
        case Err(err):
            raise HTTPException(status_code=500, detail=str(err))
    coverage = {}
    for cat_id in cat_ids:
        match taxonomy_instance.get_skills_in_category(cat_id):
            case Ok(skills):
                cat_skills = set(s.lower() for s in skills)
            case Err(err):
                raise HTTPException(status_code=500, detail=str(err))
        covered = cat_skills & deps.current_skills_set
        coverage[cat_id] = {
            "label": taxonomy_instance.get_category_label_by_id(cat_id),
            "icon": taxonomy_instance.get_category_icon_by_id(cat_id),
            "total": len(cat_skills),
            "covered": len(covered),
            "percent": round(len(covered) / len(cat_skills) * 100, 1)
            if cat_skills
            else 0,
        }
    return {"coverage": coverage}


@router.get("/api/taxonomy/professions", response_model=ProfessionsResponse)
@limiter.limit("60/minute")
async def get_professions(request: Request):
    try:
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy

        taxonomy = ProfessionTaxonomy()
        professions = []
        for name in taxonomy.professions:
            info = taxonomy.get_profession_info(name)
            professions.append(
                {
                    "name": name,
                    "domains": info.get("domains", []),
                    "competency_codes": info.get("competency_codes", []),
                    "hh_queries": info.get("hh_queries", []),
                    "aliases": info.get("aliases", []),
                }
            )
        return {"professions": professions, "total": len(professions)}
    except Exception as e:
        logger.error("get_professions_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/taxonomy/profession/{profession_name}",
    response_model=ProfessionDetailResponse,
)
@limiter.limit("60/minute")
async def get_profession_detail(request: Request, profession_name: str):
    try:
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy

        taxonomy = ProfessionTaxonomy()
        info = taxonomy.get_profession_info(profession_name)
        if not info:
            raise HTTPException(
                status_code=404, detail=f"Profession '{profession_name}' not found"
            )

        skills = list(taxonomy.get_profession_skills(profession_name))
        krm_codes = taxonomy.get_profession_competency_codes(profession_name)
        krm_skills = {}
        for code in krm_codes:
            krm_skills[code] = taxonomy.get_competency_skills(code)

        return {
            "name": profession_name,
            "domains": info.get("domains", []),
            "skill_count": len(skills),
            "skills": skills[:100],
            "competency_codes": krm_codes,
            "krm_competencies": {
                code: {"skill_count": len(s), "skills": s[:20]}
                for code, s in krm_skills.items()
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_profession_detail_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/taxonomy/profession/{profession_name}/krm-coverage",
    response_model=KRMCoverageResponse,
)
@limiter.limit("30/minute")
async def get_profession_krm_coverage(
    request: Request, profession_name: str, skills: str = Query("")
):
    try:
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy

        taxonomy = ProfessionTaxonomy()
        user_skills = (
            [s.strip() for s in skills.split(",") if s.strip()] if skills else []
        )

        coverage = taxonomy.compute_krm_coverage(profession_name, user_skills)
        if not coverage:
            raise HTTPException(
                status_code=404, detail=f"No KRM data for '{profession_name}'"
            )

        return {
            "profession": profession_name,
            "user_skills": user_skills,
            "competency_coverage": coverage,
            "avg_coverage": round(
                sum(v["coverage"] for v in coverage.values()) / len(coverage)
                if coverage
                else 0,
                4,
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_krm_coverage_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
