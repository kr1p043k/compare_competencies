"""Profile, recommendation, profession evaluation endpoints."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.models.api_responses import (
    MissingSkillsResponse,
    DeadSkillsResponse,
    ProfessionEvalResponse,
    ProfilesCompareResponse,
    ProfileShort,
)
from src.models.student import StudentProfile
from src import Err, Ok
from src.predictors.recommendation_engine import RecommendationEngine
from src.parsing.skills.skill_validator import SkillValidator

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["profiles"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/profiles/compare", response_model=ProfilesCompareResponse)
@limiter.limit("20/minute")
async def compare_profiles(
    request: Request,
    eval_instance: ProfileEvaluator = Depends(deps.get_evaluator),
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    evaluations = {}
    for pname, student in profiles.items():
        try:
            eval_result = eval_instance.evaluate_profile(student)
            evaluations[pname] = {
                "market_coverage_score": eval_result.get("market_coverage_score"),
                "skill_coverage": eval_result.get("skill_coverage"),
                "domain_coverage_score": eval_result.get("domain_coverage_score"),
                "readiness_score": eval_result.get("readiness_score"),
                "real_coverage": eval_result.get("market_skill_coverage"),
            }
        except Exception as e:
            logger.error("Ошибка оценки профиля", profile=pname, error=str(e))
            evaluations[pname] = {"error": str(e)}
    return {"profiles": evaluations}


@router.get("/api/profiles/{profile}", response_model=ProfileShort)
@limiter.limit("60/minute")
async def get_profile(
    request: Request,
    profile: str,
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    student = profiles[profile]
    return {
        "profile_name": student.profile_name,
        "target_level": student.target_level,
        "skills_count": len(student.skills),
        "skills": student.skills[:50],
        "competencies_count": len(student.competencies),
        "competencies": student.competencies[:50],
    }


@router.get(
    "/api/profiles/{profile}/profession-evaluation",
    response_model=ProfessionEvalResponse,
)
@limiter.limit("30/minute")
async def get_profile_profession_evaluation(request: Request, profile: str):
    if profile not in deps.student_profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")

    from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy

    taxonomy = ProfessionTaxonomy()
    profile_config = taxonomy.get_profile_target(profile)
    if not profile_config:
        raise HTTPException(
            status_code=404, detail=f"No profession target for '{profile}'"
        )

    student = deps.student_profiles[profile]
    result = deps.evaluator.evaluate_profile(
        student,
        user_type="student",
        target_domains=profile_config.get("target_domains", []),
        taxonomy=taxonomy,
    )

    return {
        "profile": profile,
        "target_profession": profile_config.get("target_profession", ""),
        "target_domains": profile_config.get("target_domains", []),
        "profession_coverage": result.get("profession_coverage", 0),
        "krm_coverage": result.get("krm_coverage", {}),
        "readiness_score": result.get("readiness_score", 0),
        "skill_coverage": result.get("skill_coverage", 0),
        "domain_coverage_score": result.get("domain_coverage_score", 0),
    }


@router.get("/api/recommendations/{profile}", response_model=dict)
@limiter.limit("30/minute")
async def get_recommendations(
    request: Request,
    profile: str,
    engine: RecommendationEngine = Depends(deps.get_recommendation_engine),
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    student = profiles[profile]
    match engine.generate_recommendations(student):
        case Ok(full_rec):
            return full_rec
        case Err(err):
            raise HTTPException(status_code=500, detail=str(err))


@router.get("/api/skills/missing", response_model=MissingSkillsResponse)
@limiter.limit("30/minute")
async def missing_skills(
    request: Request,
    min_frequency: int = Query(1),
    freq: dict[str, int] = Depends(deps.get_skill_freq),
):
    validator = SkillValidator(whitelist=None)
    extracted = {}
    for skill, freq_val in freq.items():
        if (
            skill.lower() not in deps.current_skills_set
            and freq_val >= min_frequency
            and validator.validate(skill).is_valid
        ):
            extracted[skill] = freq_val
    sorted_skills = sorted(extracted.items(), key=lambda x: x[1], reverse=True)
    return {"missing_skills": [{"skill": s, "frequency": f} for s, f in sorted_skills]}


@router.get("/api/skills/dead", response_model=DeadSkillsResponse)
@limiter.limit("30/minute")
async def dead_skills(
    request: Request,
    freq: dict[str, int] = Depends(deps.get_skill_freq),
):
    extracted_lower = {s.lower() for s in freq}
    dead = sorted(
        s for s in deps.current_skills_set if s.lower() not in extracted_lower
    )
    return {"dead_skills": dead}
