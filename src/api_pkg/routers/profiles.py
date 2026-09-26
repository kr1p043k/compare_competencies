"""Profile, recommendation, profession evaluation endpoints."""

from typing import Any

import asyncio
import numpy as np
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel

from src import Err, Ok
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.api_pkg import deps
from src.models.api_responses import (
    DeadSkillsResponse,
    MissingSkillsResponse,
    ProfessionEvalResponse,
    ProfilesCompareResponse,
    ProfileShort,
)
from src.api_pkg.routers.auth import require_any_role, user_error_detail
from src.models.student import StudentProfile
from src.models.enums import ExperienceLevel
from src import config
from src.parsing.skills.skill_validator import SkillValidator
from src.predictors.recommendation_engine import RecommendationEngine

logger = structlog.get_logger("api")

router = APIRouter(tags=["profiles"])
limiter = Limiter(key_func=get_remote_address)


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy types to native JSON-safe Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@router.get("/profiles/compare", response_model=ProfilesCompareResponse)
@limiter.limit("20/minute")
async def compare_profiles(
    request: Request,
    eval_instance: ProfileEvaluator = Depends(deps.get_evaluator),
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """Сравнение профилей студентов."""
    from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
    prof_taxonomy = ProfessionTaxonomy()
    evaluations = {}
    for pname, student in profiles.items():
        # Stage 4: паритет с CLI — фильтруем по целевым доменам профиля.
        cfg = prof_taxonomy.get_profile_target(pname) or {}
        domains = cfg.get("target_domains", [])
        if domains and not student.target_profession:
            try:
                student.target_profession = cfg.get("target_profession", "")
            except Exception:
                pass
        match eval_instance.evaluate_profile(
            student,
            target_domains=domains or None,
            taxonomy=prof_taxonomy if domains else None,
        ):
            case Ok(eval_result):
                evaluations[pname] = {
                    "market_coverage_score": eval_result.get("market_coverage_score"),
                    "skill_coverage": eval_result.get("skill_coverage"),
                    "domain_coverage_score": eval_result.get("domain_coverage_score"),
                    "readiness_score": eval_result.get("readiness_score"),
                    "real_coverage": eval_result.get("market_skill_coverage"),
                }
            case Err(err):
                logger.error("Ошибка оценки профиля", profile=pname, error=str(err))
                evaluations[pname] = {"error": str(err)}
    return {"profiles": evaluations}


@router.get("/profiles")
@limiter.limit("60/minute")
async def list_profiles(request: Request):
    """Profile names available in memory (built-ins + custom created via POST /profiles/custom)."""
    return {"profiles": sorted(deps.student_profiles.keys())}


class CustomProfileIn(BaseModel):
    name: str
    target_level: str
    competencies: list[str] = []
    skills: list[str] = []


_CUSTOM_NAME_RE = "^[a-z0-9_]{2,32}$"


@router.post("/profiles/custom", status_code=201, dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("10/minute")
async def create_custom_profile(request: Request, body: CustomProfileIn):
    """Create your own competency profile (Data tab): persists to
    data/students/<name>_competency.json and registers it in memory."""
    import json
    import re
    from pathlib import Path

    name = (body.name or "").strip().lower()
    if not re.match(_CUSTOM_NAME_RE, name):
        raise HTTPException(status_code=400,
                            detail="name must match [a-z0-9_]{2,32}")
    if name in deps.student_profiles:
        raise HTTPException(status_code=409, detail="Profile already exists")
    try:
        level = ExperienceLevel(body.target_level.strip().lower())
    except ValueError:
        raise HTTPException(status_code=400,
                            detail="target_level must be junior|middle|senior")
    codes = list(dict.fromkeys(c.strip() for c in (body.competencies or [])
                               if c and c.strip()))[:200]
    skills = list(dict.fromkeys(s.strip() for s in (body.skills or [])
                                if s and s.strip()))[:500]
    if not codes and not skills:
        raise HTTPException(status_code=400,
                            detail="provide competencies or skills")
    if not skills and codes:
        # same code->skills mapping as startup loader
        try:
            mp = json.loads((Path(config.DATA_DIR) / "processed" / "competency_mapping.json")
                            .read_text(encoding="utf-8"))
        except Exception:
            mp = {}
        mapped: set[str] = set()
        for code in codes:
            cn = "".join(c for c in code if c.isalnum()).upper()
            for key, value in (mp.items() if isinstance(mp, dict) else []):
                kn = "".join(c for c in str(key) if c.isalnum()).upper()
                if cn and cn == kn:
                    mapped.update(value if isinstance(value, list) else [value])
                    break
        skills = sorted(mapped)[:500]
    students_dir = Path(config.DATA_DIR) / "students"
    students_dir.mkdir(parents=True, exist_ok=True)
    fpath = (students_dir / f"{name}_competency.json").resolve()
    if students_dir.resolve() not in fpath.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    fpath.write_text(json.dumps({"competencies": codes, "skills": skills,
                                "target_level": level.value},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    deps.student_profiles[name] = StudentProfile(
        profile_name=name, competencies=codes, skills=skills, target_level=level)
    logger.info("custom_profile_created", profile=name, competencies=len(codes),
                skills=len(skills))
    return {"profile": name, "target_level": level.value,
            "competencies_count": len(codes), "skills_count": len(skills)}


@router.get(
    "/profiles/{profile}/profession-evaluation",
    response_model=dict,
)
@limiter.limit("30/minute")
async def get_profile_profession_evaluation(
    request: Request,
    profile: str,
    profession: str | None = Query(None),
    engine: Any = Depends(deps.get_recommendation_engine),
):
    """Оценка профиля под профессию. Без ?profession= — фиксированная цель профиля
    (краткий ответ). С ?profession=<имя> — полный фокус-режим: строгие метрики,
    KRM по выбранной профессии и Топ-10, пересчитанный движком под её домены."""
    if profile not in deps.student_profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")
    if deps.evaluator is None:
        raise HTTPException(status_code=503, detail="Анализатор ещё не загружен")

    from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy

    taxonomy = ProfessionTaxonomy()
    if profession:
        info = taxonomy.get_profession_info(profession)
        if not info:
            raise HTTPException(status_code=404, detail=f"Profession '{profession}' not found")
        profile_config = {
            "target_profession": profession,
            "target_domains": info.get("domains", []),
        }
    else:
        profile_config = taxonomy.get_profile_target(profile)
    if not profile_config:
        raise HTTPException(
            status_code=404, detail=f"No profession target for '{profile}'"
        )

    student = deps.student_profiles[profile]
    match deps.evaluator.evaluate_profile(
        student,
        user_type="student",
        target_domains=profile_config.get("target_domains", []),
        taxonomy=taxonomy,
    ):
        case Ok(result):
            base = {
                "profile": profile,
                "target_profession": profile_config.get("target_profession", ""),
                "target_domains": profile_config.get("target_domains", []),
                "profession_coverage": result.get("profession_coverage", 0),
                "krm_coverage": result.get("krm_coverage", {}),
                "readiness_score": result.get("readiness_score", 0),
                "skill_coverage": result.get("skill_coverage", 0),
                "domain_coverage_score": result.get("domain_coverage_score", 0),
            }
            if not profession:
                return base
            return await _build_focused_profession_view(
                request, profile, student, taxonomy, profile_config, result, engine, base
            )
        case Err(err):
            logger.warning("profile_compare_failed", profile=profile, error=str(err))
            raise HTTPException(
                status_code=500,
                detail=await user_error_detail(request, str(err), "Не удалось сравнить профиль. Попробуйте позже."),
            )


async def _build_focused_profession_view(
    request: Request,
    profile: str,
    student: Any,
    taxonomy: Any,
    profile_config: dict,
    eval_result: dict,
    engine: Any,
    base: dict,
) -> dict:
    """Полный фокус-режим под выбранную профессию: строгие метрики, KRM по ней
    и Топ-10, пересчитанный движком. Тяжёлая часть — в thread pool, не блокируем loop."""
    profession = profile_config.get("target_profession", "")
    domains = profile_config.get("target_domains", [])

    # 1. Строгое покрытие: пересечение навыков профиля с навыками профессии.
    # Штатный skill_coverage считает взвешенно по demand и засчитывает
    # нулевые веса как «покрыто» (отсюда 91.57 при profession 0) — здесь честно.
    prof_skills = {s.lower().strip() for s in taxonomy.get_profession_skills(profession)}
    user_set = {s.lower().strip() for s in (student.skills or [])}
    strict_has = sorted(prof_skills & user_set)
    strict_total = len(prof_skills)
    strict_cov = round(len(strict_has) / strict_total * 100, 2) if strict_total else 0.0

    # 2. KRM по ВЫБРАННОЙ профессии (evaluator считает по своей цели профиля).
    # KRM-маппинг в таксономии есть не для всех профессий — тогда честно говорим.
    krm_cov: dict = {}
    try:
        krm_cov = taxonomy.compute_krm_coverage(profession, student.skills or []) or {}
    except Exception as e:
        logger.warning("focused_krm_failed", profession=profession, error=str(e))
    krm_available = bool(krm_cov)
    krm_note = (
        ""
        if krm_available
        else f"KRM-маппинг в таксономии есть только для части профессий — для «{profession}» данных нет"
    )

    # 3. Полный пересчёт Топ-10 движком под домены профессии.
    try:
        gen = await asyncio.to_thread(
            engine.generate_recommendations,
            student,
            "student",
            eval_result,
            domains,
            taxonomy,
        )
    except Exception as e:
        logger.warning("focused_recommend_failed", profession=profession, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=await user_error_detail(request, str(e), "Не удалось пересчитать рекомендации. Попробуйте позже."),
        )
    match gen:
        case Ok(rec_result):
            pass
        case Err(err):
            logger.warning("focused_recommend_err", profession=profession, error=str(err))
            raise HTTPException(
                status_code=500,
                detail=await user_error_detail(request, str(err), "Не удалось пересчитать рекомендации. Попробуйте позже."),
            )
        case _:
            raise HTTPException(status_code=500, detail="Не удалось пересчитать рекомендации. Попробуйте позже.")
    payload = rec_result.model_dump()

    # Подменяем сводку фокусными цифрами, старые оставляем для прозрачности.
    # R2: строгое — в coverage_strict (scope profession), взвешенное не затираем.
    summary = payload.get("summary", {}) or {}
    summary["skill_coverage"] = strict_cov
    summary["skill_coverage_market"] = base.get("skill_coverage", 0)
    summary["coverage_strict"] = strict_cov
    summary["coverage_weighted"] = base.get("skill_coverage", 0)
    summary["coverage_strict_scope"] = "profession"
    payload["summary"] = summary

    return {
        **base,
        "focus_mode": True,
        "skill_coverage": strict_cov,
        "skill_coverage_market": base.get("skill_coverage", 0),
        "coverage_strict": strict_cov,
        "coverage_weighted": base.get("skill_coverage", 0),
        "coverage_strict_scope": "profession",
        "skill_strict_has": len(strict_has),
        "skill_strict_total": strict_total,
        "skill_strict_missing": sorted(prof_skills - user_set)[:50],
        "krm_coverage": krm_cov,
        "krm_available": krm_available,
        "krm_note": krm_note,
        "recommendations": payload.get("recommendations", []),
        "closest_roles": payload.get("closest_roles", []),
        "gaps": payload.get("gaps", {}),
        "domain_coverage": payload.get("domain_coverage", {}),
        "summary": summary,
    }


@router.get("/recommendations/{profile}", response_model=dict)
@limiter.limit("30/minute")
async def get_recommendations(
    request: Request,
    profile: str,
    engine: RecommendationEngine = Depends(deps.get_recommendation_engine),
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """Рекомендации навыков профилю."""
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    student = profiles[profile]
    try:
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
        prof_taxonomy = ProfessionTaxonomy()
        cfg = prof_taxonomy.get_profile_target(profile) or {}
        domains = cfg.get("target_domains", [])
        precomputed = None
        if domains:
            if not student.target_profession:
                try:
                    student.target_profession = cfg.get("target_profession", "")
                except Exception:
                    pass
            match deps.get_evaluator().evaluate_profile(
                student, target_domains=domains, taxonomy=prof_taxonomy,
            ):
                case Ok(ev):
                    precomputed = ev
                case Err(err):
                    logger.warning("profile_recommendations_eval_failed",
                                   profile=profile, error=str(err))
        match engine.generate_recommendations(
            student,
            precomputed_eval=precomputed,
            taxonomy=prof_taxonomy if domains else None,
        ):
            case Ok(full_rec):
                return _json_safe(full_rec.model_dump())
            case Err(err):
                logger.warning("profile_recommendations_failed", profile=profile, error=str(err))
                raise HTTPException(
                    status_code=500,
                    detail=await user_error_detail(request, str(err), "Не удалось построить рекомендации. Попробуйте позже."),
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("recommendations_endpoint_failed", profile=profile, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=await user_error_detail(request, str(e), "Не удалось построить рекомендации. Попробуйте позже."),
        ) from None


@router.get("/skills/missing", response_model=MissingSkillsResponse)
@limiter.limit("30/minute")
async def missing_skills(
    request: Request,
    min_frequency: int = Query(1),
    freq: dict[str, int] = Depends(deps.get_skill_freq),
):
    """Отсутствующие навыки (по частоте)."""
    validator = SkillValidator(whitelist=None)
    extracted = {}
    for skill, freq_val in freq.items():
        if skill.lower() in deps.current_skills_set or freq_val < min_frequency:
            continue
        match validator.validate(skill):
            case Ok(result) if result.is_valid:
                extracted[skill] = freq_val
            case _:
                pass
    sorted_skills = sorted(extracted.items(), key=lambda x: x[1], reverse=True)
    return {"missing_skills": [{"skill": s, "frequency": f} for s, f in sorted_skills]}


@router.get("/skills/dead", response_model=DeadSkillsResponse)
@limiter.limit("30/minute")
async def dead_skills(
    request: Request,
    freq: dict[str, int] = Depends(deps.get_skill_freq),
):
    """Мёртвые навыки (нулевой спрос)."""
    extracted_lower = {s.lower() for s in freq}
    dead = sorted(
        s for s in deps.current_skills_set if s.lower() not in extracted_lower
    )
    return {"dead_skills": dead}


def _self_profile_name(email: str) -> tuple[str, str]:
    """Личный профиль пользователя: (имя профиля, путь к файлу)."""
    import re
    from pathlib import Path

    safe = re.sub(r"[^a-z0-9_]", "_", (email or "anon").strip().lower())[:40] or "anon"
    name = f"self_{safe}"
    return name, str(Path(config.DATA_DIR) / "students" / f"{name}_competency.json")


def _load_self_profile(name: str, fpath: str) -> dict:
    import json

    try:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"skills": [], "target_level": "middle", "user_added": [], "competencies": []}


def _save_self_profile(name: str, fpath: str, data: dict) -> None:
    import json
    from pathlib import Path

    p = Path(fpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.resolve().parent != Path(config.DATA_DIR).resolve() / "students":
        raise HTTPException(status_code=400, detail="Invalid path")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _sync_self_to_memory(name: str, data: dict) -> None:
    try:
        level = ExperienceLevel(str(data.get("target_level", "middle")).strip().lower())
    except ValueError:
        level = ExperienceLevel.MIDDLE
        data["target_level"] = level.value
    deps.student_profiles[name] = StudentProfile(
        profile_name=name, competencies=list(data.get("competencies", [])),
        skills=list(data.get("skills", [])), target_level=level)


@router.get("/profiles/self", response_model=dict)
@limiter.limit("60/minute")
async def get_self_profile(request: Request):
    """Личный профиль текущего пользователя (создаётся пустым при первом чтении)."""
    from src.api_pkg.routers.auth import get_current_user

    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    name, fpath = _self_profile_name(str(user.get("u") or ""))
    data = _load_self_profile(name, fpath)
    _sync_self_to_memory(name, data)
    return {"profile": name, "target_level": data.get("target_level", "middle"),
            "skills": data.get("skills", []), "user_added": data.get("user_added", []),
            "competencies": data.get("competencies", []),
            "skills_count": len(data.get("skills", []))}


class SelfProfilePatch(BaseModel):
    target_level: str | None = None
    add_skills: list[str] = []
    remove_skills: list[str] = []


@router.patch("/profiles/self", response_model=dict)
@limiter.limit("30/minute")
async def patch_self_profile(request: Request, body: SelfProfilePatch):
    """Свой профиль: смена уровня + добавление своих навыков; удаление —
    только собою добавленных. Чужие профили недоступны по построению."""
    from src.api_pkg.routers.auth import get_current_user

    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    email = str(user.get("u") or "")
    name, fpath = _self_profile_name(email)
    data = _load_self_profile(name, fpath)

    if body.target_level is not None:
        try:
            level = ExperienceLevel(body.target_level.strip().lower())
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="target_level must be junior|middle|senior")
        data["target_level"] = level.value

    skills = list(data.get("skills", []))
    owned = list(data.get("user_added", []))
    owned_lower = {s.lower() for s in owned}
    have_lower = {s.lower() for s in skills}
    added, removed, refused = [], [], []
    for s in (body.add_skills or [])[:100]:
        s = (s or "").strip()[:200]
        if not s:
            continue
        if s.lower() not in have_lower:
            skills.append(s)
            have_lower.add(s.lower())
        if s.lower() not in owned_lower:
            owned.append(s)
            owned_lower.add(s.lower())
        added.append(s)
    for s in (body.remove_skills or [])[:100]:
        key = (s or "").strip().lower()
        if not key:
            continue
        if key in owned_lower:
            skills = [x for x in skills if x.lower() != key]
            owned = [x for x in owned if x.lower() != key]
            owned_lower.discard(key)
            removed.append(s)
        else:
            refused.append(s)
    data["skills"] = skills
    data["user_added"] = owned
    _save_self_profile(name, fpath, data)
    _sync_self_to_memory(name, data)

    try:
        from src.api_pkg.student_actions import log_action
        log_action(username=email, action_type="profile_edit", profile=name,
                   result_ref=f"added={len(added)} removed={len(removed)} refused={len(refused)}")
    except Exception:
        pass
    logger.info("self_profile_patched", profile=name,
                added=len(added), removed=len(removed), refused=len(refused))
    return {"profile": name, "target_level": data.get("target_level"),
            "skills": skills, "user_added": owned,
            "added": added, "removed": removed, "refused": refused}


@router.get("/profiles/{profile}", response_model=ProfileShort)
@limiter.limit("60/minute")
async def get_profile(
    request: Request,
    profile: str,
    full: bool = Query(False),
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """Профиль студента по имени. По умолчанию первые 50 навыков (для UI),
    ?full=true — полный список (для графиков покрытия)."""
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    student = profiles[profile]
    return {
        "profile_name": student.profile_name,
        "target_level": student.target_level,
        "skills_count": len(student.skills),
        "skills": student.skills if full else student.skills[:50],
        "competencies_count": len(student.competencies),
        "competencies": student.competencies[:50],
    }
