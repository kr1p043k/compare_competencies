"""Results summary, recommendations files, images."""

import asyncio
import copy
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.api_pkg import deps
from src.api_pkg.routers.auth import require_any_role, user_error_detail
from src.api_pkg.summary_builder import build_summary_payload, load_recommendations_from_disk
from src.models.student import StudentProfile

logger = structlog.get_logger("api")

router = APIRouter(tags=["results"])
limiter = Limiter(key_func=get_remote_address)


# Кэш распарсенных result-файлов по (path, mtime_ns, size). result/*/_full_...json
# перечитываются на каждый запрос; кэшируем по версии файла и отдаём копии,
# чтобы per-user мутации не попали в кэш и не утекли между пользователями.
_json_cache: dict[str, tuple[int, int, Any]] = {}
_json_cache_lock = threading.Lock()


def _reset_results_cache() -> None:
    with _json_cache_lock:
        _json_cache.clear()


def _read_json(path: Path) -> Any:
    """Прочитать+распарсить JSON, кэшируя по (path, mtime_ns, size).

    Каждый вызов возвращает глубокую копию: вызывающие мутируют результат
    (фильтр студента `data["evaluations"]`, `payload["generated_at"]`), и копия
    не даёт протечь правкам одного запроса в кэш и в чужие ответы.
    """
    key = str(path)
    st = path.stat()  # прежняя семантика: отсутствие файла - ошибка, выше есть exists-проверки
    sig = (st.st_mtime_ns, st.st_size)
    with _json_cache_lock:
        cached = _json_cache.get(key)
    if cached is None or cached[0] != sig[0] or cached[1] != sig[1]:
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
        with _json_cache_lock:
            _json_cache[key] = (sig[0], sig[1], parsed)
        cached = (sig[0], sig[1], parsed)
    return copy.deepcopy(cached[2])


@router.get(
    "/results/summary",
    dependencies=[Depends(require_any_role("admin", "teacher", "rop", "student"))],
    response_model=dict,
)
@limiter.limit("30/minute")
async def get_results_summary(
    request: Request,
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """Сводка результатов анализа.

    Студенту отдаём только его собственный профиль: `base`/`dc`/`top_dc` - это
    архетипы, и «58% соответствия» от архетипа студент читает как оценку себя,
    хотя к нему она отношения не имеет. Преподавателю нужны все профили.
    """
    from src.api_pkg.routers.auth import get_current_user

    user = await get_current_user(request) or {}
    role = user.get("r") or user.get("role")

    evaluations = build_summary_payload(load_recommendations_from_disk())

    self_key: str | None = None
    if role == "student":
        username = str(user.get("u") or user.get("username") or "")
        safe = "".join(ch if ch.isalnum() else "_" for ch in username.lower())
        for candidate in (f"self_{safe}", f"self_{safe.replace('.', '_')}"):
            if candidate in evaluations:
                self_key = candidate
                break
        if self_key:
            evaluations = {self_key: evaluations[self_key]}
        else:
            # Своего профиля ещё нет: показывать чужие архетипы нельзя, иначе
            # студент примет их за собственную оценку.
            evaluations = {}

    if evaluations:
        return {
            "evaluations": evaluations,
            "profiles": list(evaluations.keys()),
            "self_profile": self_key,
        }

    for summary_path in (
        config.DATA_RESULT_DIR / "profiles_comparison_summary.json",
        config.DATA_PROCESSED_DIR / "profiles_comparison_summary.json",
    ):
        if summary_path.exists():
            try:
                data = await asyncio.to_thread(_read_json, summary_path)
                if role == "student" and isinstance(data, dict):
                    evals = data.get("evaluations")
                    if isinstance(evals, dict):
                        data["evaluations"] = (
                            {self_key: evals[self_key]} if self_key in evals else {}
                        )
                    # Список профилей тоже не должен раскрывать архетипы:
                    # иначе UI нарисует выбор base/dc/top_dc.
                    data["profiles"] = [self_key] if self_key else []
                return data
            except Exception as e:
                logger.warning("results_summary_read_failed", path=str(summary_path), error=str(e))
                raise HTTPException(
                    status_code=500,
                    detail=await user_error_detail(
                        request, str(e), "Не удалось прочитать результаты анализа. Попробуйте позже."
                    ),
                )

    return {
        "message": "Результаты анализа не найдены. Запустите gap-анализ.",
        "profiles": ([self_key] if self_key else []) if role == "student" else list(profiles.keys()),
    }


@router.get(
    "/results/recommendations/{profile}",
    dependencies=[Depends(require_any_role("admin", "teacher", "rop", "student"))],
    response_model=dict,
)
@limiter.limit("30/minute")
async def get_recommendations_result(
    request: Request,
    profile: str,
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """Сохранённые рекомендации профиля."""
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")

    result_path = (
        config.DATA_DIR / "result" / profile / f"full_recommendations_{profile}.json"
    )
    if result_path.exists():
        try:
            payload = await asyncio.to_thread(_read_json, result_path)
            try:
                mtime = result_path.stat().st_mtime
                payload["generated_at"] = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
            except Exception:
                pass
            return payload
        except Exception as e:
            logger.warning("results_recommendations_read_failed", path=str(result_path), error=str(e))
            raise HTTPException(
                status_code=500,
                detail=await user_error_detail(request, str(e), "Не удалось прочитать рекомендации. Попробуйте позже."),
            )

    return {
        "profile": profile,
        "message": "Рекомендации не найдены. Запустите gap-анализ.",
        "recommendations": [],
    }


@router.get("/results/report/{profile}", dependencies=[Depends(require_any_role("admin", "teacher", "rop", "student"))])
@limiter.limit("10/minute")
async def get_report_pdf(
    request: Request,
    profile: str,
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """PDF-отчёт профиля: метрики, роли, рекомендации + PNG-графики."""
    from fastapi.responses import Response

    from src.reports.pdf_report import build_profile_pdf

    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    result_path = (
        config.DATA_DIR / "result" / profile / f"full_recommendations_{profile}.json"
    )
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Рекомендации не найдены. Запустите gap-анализ.")
    try:
        payload = await asyncio.to_thread(_read_json, result_path)
        pdf_bytes = await asyncio.to_thread(
            build_profile_pdf, profile, payload, config.REPORTS_DIR)
    except Exception as e:
        logger.warning("results_report_pdf_failed", profile=profile, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=await user_error_detail(request, str(e), "Не удалось собрать PDF. Попробуйте позже."),
        )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{profile}.pdf"'},
    )


# FROZEN v1: PNG-графики (радар/heatmap/coverage). Зафиксированы как v1 API:
# доступны и под /api/..., и под /api/v1/... (общий _mount). Интерактивные
# графики фронта ходят по данным (/market, /taxonomy, /profiles) и эти
# эндпоинты не используют. НЕ удалять и НЕ менять контракт без новой версии.
@router.get(
    "/results/images/{profile}/{image_type}",
    dependencies=[Depends(require_any_role("admin", "teacher", "rop", "student"))],
)
@limiter.limit("60/minute")
async def get_profile_image(
    request: Request,
    profile: str,
    image_type: str,
):
    """PNG-график профиля по типу."""
    import re
    if not re.match(r"^[a-zA-Z0-9_-]+$", profile):
        raise HTTPException(status_code=400, detail="Invalid profile name")

    safe_types = ["radar", "ml_importance", "cluster_insights", "skills_heatmap", "skill_correlation"]
    if image_type not in safe_types:
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_path = config.DATA_DIR / "result" / profile / f"{image_type}_{profile}.png"
    if not image_path.exists():
        image_path = config.DATA_DIR / "result" / f"{image_type}_{profile}.png"
    if not image_path.exists():
        image_path = config.REPORTS_DIR / profile / f"{image_type}_{profile}.png"
    if not image_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Image not found: {image_type}_{profile}.png"
        )

    resolved = image_path.resolve()
    allowed = config.DATA_DIR.resolve() / "result"
    if allowed not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid image path")

    return FileResponse(resolved, media_type="image/png")


@router.get(
    "/results/images/coverage-comparison",
    dependencies=[Depends(require_any_role("admin", "teacher", "rop", "student"))],
)
@limiter.limit("30/minute")
async def get_coverage_comparison_image(request: Request):
    """PNG сравнения покрытия."""
    image_path = config.REPORTS_DIR / "coverage_comparison.png"
    if not image_path.exists():
        raise HTTPException(
            status_code=404, detail="Coverage comparison image not found"
        )
    return FileResponse(image_path, media_type="image/png")


@router.get(
    "/results/images/skills-heatmap",
    dependencies=[Depends(require_any_role("admin", "teacher", "rop", "student"))],
)
@limiter.limit("30/minute")
async def get_skills_heatmap_image(request: Request):
    """PNG тепловой карты навыков."""
    image_path = config.REPORTS_DIR / "skills_heatmap.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Skills heatmap not found")
    return FileResponse(image_path, media_type="image/png")


@router.get(
    "/results/images/skill-correlation",
    dependencies=[Depends(require_any_role("admin", "teacher", "rop", "student"))],
)
@limiter.limit("30/minute")
async def get_skill_correlation_image(request: Request):
    """PNG корреляций навыков."""
    image_path = config.REPORTS_DIR / "skill_correlation_heatmap.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Skill correlation not found")
    return FileResponse(image_path, media_type="image/png")
