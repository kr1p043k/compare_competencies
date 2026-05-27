"""Results summary, recommendations files, images."""

import json

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.models.student import StudentProfile

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["results"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/results/summary", response_model=dict)
@limiter.limit("30/minute")
async def get_results_summary(
    request: Request,
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    summary_path = config.DATA_PROCESSED_DIR / "profiles_comparison_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {
        "message": "Результаты анализа не найдены. Запустите gap-анализ.",
        "profiles": list(profiles.keys()),
    }


@router.get("/api/results/recommendations/{profile}", response_model=dict)
@limiter.limit("30/minute")
async def get_recommendations_result(
    request: Request,
    profile: str,
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")

    result_path = (
        config.DATA_DIR / "result" / profile / f"full_recommendations_{profile}.json"
    )
    if result_path.exists():
        try:
            with open(result_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "profile": profile,
        "message": "Рекомендации не найдены. Запустите gap-анализ.",
        "recommendations": [],
    }


@router.get("/api/results/images/{profile}/{image_type}")
@limiter.limit("60/minute")
async def get_profile_image(
    request: Request,
    profile: str,
    image_type: str,
):
    safe_types = ["radar", "ml_importance", "cluster_insights", "deficits", "skills_heatmap", "skill_correlation"]
    if image_type not in safe_types:
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_path = config.DATA_DIR / "result" / profile / f"{image_type}_{profile}.png"
    if not image_path.exists():
        image_path = config.DATA_DIR / "result" / f"{image_type}_{profile}.png"
    if not image_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Image not found: {image_type}_{profile}.png"
        )

    return FileResponse(image_path, media_type="image/png")


@router.get("/api/results/images/coverage-comparison")
@limiter.limit("30/minute")
async def get_coverage_comparison_image(request: Request):
    image_path = config.REPORTS_DIR / "coverage_comparison.png"
    if not image_path.exists():
        raise HTTPException(
            status_code=404, detail="Coverage comparison image not found"
        )
    return FileResponse(image_path, media_type="image/png")


@router.get("/api/results/images/skills-heatmap")
@limiter.limit("30/minute")
async def get_skills_heatmap_image(request: Request):
    image_path = config.REPORTS_DIR / "skills_heatmap.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Skills heatmap not found")
    return FileResponse(image_path, media_type="image/png")


@router.get("/api/results/images/skill-correlation")
@limiter.limit("30/minute")
async def get_skill_correlation_image(request: Request):
    image_path = config.REPORTS_DIR / "skill_correlation_heatmap.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Skill correlation not found")
    return FileResponse(image_path, media_type="image/png")
