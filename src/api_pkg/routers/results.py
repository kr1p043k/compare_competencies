"""Results summary, recommendations files, images."""

import asyncio
import copy
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import Err, Ok, config
from src.api_pkg import deps
from src.api_pkg.routers.auth import require_any_role, user_error_detail
from src.api_pkg.summary_builder import build_summary_payload, load_recommendations_from_disk
from src.api_pkg.user_keys import self_profile_keys
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


async def _require_student_profile(request: Request, profile: str) -> None:
    """Студент читает только рекомендации/отчёт своего профиля; чужой - 403.

    Ключ своего профиля строится так же, как в get_results_summary
    (self_<username>); admin/teacher/rop не ограничиваются.
    """
    from src.api_pkg.routers.auth import get_current_user

    user = await get_current_user(request) or {}
    if (user.get("r") or user.get("role")) != "student":
        return
    username = str(user.get("u") or user.get("username") or "")
    if profile in self_profile_keys(username):
        return
    raise HTTPException(status_code=403, detail="Доступ к чужому профилю запрещён")


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
        for candidate in self_profile_keys(username):
            if candidate in evaluations:
                self_key = candidate
                break
        # Своего профиля ещё нет: показывать чужие архетипы нельзя, иначе
        # студент примет их за собственную оценку.
        evaluations = {self_key: evaluations[self_key]} if self_key else {}

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
                ) from None

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
    await _require_student_profile(request, profile)
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
            ) from None

    return {
        "profile": profile,
        "message": "Рекомендации не найдены. Запустите gap-анализ.",
        "recommendations": [],
    }


# ---------------------------------------------------------------- ML #78: SHAP
# Ленивая LTRRecommendationEngine для ML-объяснений: артефакт + эмбеддинги
# тяжёлые, создаём при первом запросе и держим один инстанс на процесс.
# Read-only: инференс (predict + SHAP) состояние движка не меняет.
_explanation_engine: Any | None = None
_explanation_engine_lock = threading.Lock()

# Сколько недостающих навыков-кандидатов ранжировать перед SHAP. Как в
# teacher_analysis_runner: co-occurrence + SHAP на всех ~4000 навыках
# неприемлемо медленны, а топ рекомендаций почти всегда среди частых.
_EXPLANATION_CANDIDATE_LIMIT = 200


def _get_or_load_explanation_engine() -> Any:
    """Вернуть LTR-движок; подгрузить артефакт, если он ещё не загружен.

    Незагруженный артефакт перепроверяется при каждом запросе: после
    обучения модели пайплайном объяснения появляются без рестарта API.
    Возвращает unfit-движок, если артефакта нет - эндпоинт ответит
    graceful-сообщением.
    """
    global _explanation_engine
    with _explanation_engine_lock:
        if _explanation_engine is None:
            # Импорт внутри функции: xgboost/shap не нужны остальным роутам.
            from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine

            _explanation_engine = LTRRecommendationEngine()
        if not getattr(_explanation_engine, "is_fitted", False):
            from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine

            match _explanation_engine.load_model(LTRRecommendationEngine._resolve_model_path()):
                case Ok(_):
                    logger.info("ml_explanations_model_loaded")
                case Err(e):
                    logger.warning("ml_explanations_model_not_ready", error=str(e))
        return _explanation_engine


def _shap_contributions(
    explain_result: tuple[Any, Any, Any],
    row: int,
    top: int,
) -> list[dict[str, Any]]:
    """Топ-|SHAP|-вкладов строки: feature, value, direction, feature_value.

    Строки shap_values/X выровнены с impacts контрактом
    predict_skill_impact_with_shap (индекс i = навык impacts[i]).
    """
    _impacts, shap_vals, x = explain_result
    if shap_vals is None or x is None or row >= len(shap_vals):
        return []
    columns = list(x.columns)
    row_values = shap_vals[row]
    # Точные нули не несут сигнала в топ-списке: не тратим на них слоты top.
    candidates = [
        j
        for j in range(len(row_values))
        if j < len(columns) and float(row_values[j]) != 0.0
    ]
    order = sorted(candidates, key=lambda j: abs(float(row_values[j])), reverse=True)[:top]
    contributions: list[dict[str, Any]] = []
    for j in order:
        value = float(row_values[j])
        feature = str(columns[j])
        try:
            feature_value = round(float(x.iloc[row][feature]), 4)
        except Exception:  # нечисловое значение столбца - не ронять объяснение
            feature_value = None
        contributions.append(
            {
                "feature": feature,
                "value": round(value, 4),
                "direction": "positive" if value > 0 else "negative" if value < 0 else "zero",
                "feature_value": feature_value,
            }
        )
    return contributions


@router.get(
    "/results/ml-explanations/{profile}",
    dependencies=[Depends(require_any_role("admin", "teacher", "rop"))],
    response_model=dict,
)
@limiter.limit("10/minute")
async def get_ml_explanations(
    request: Request,
    profile: str,
    top: int = Query(default=5, ge=1, le=8),
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """ML #78: per-skill объяснение рекомендаций (SHAP-вклады признаков).

    Read-only view: живой инференс LTR-модели по навыкам профиля, без
    записи на диск. SHAP считаем для топ-навыков (дорого), поэтому
    rate-limit жёстче, чем у файловых эндпоинтов.
    """
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")

    student_profile = profiles[profile]
    student_skills = (
        student_profile.skills
        or [*student_profile.competencies, *student_profile.technologies]
    )

    engine = _get_or_load_explanation_engine()
    if engine is None or not getattr(engine, "is_fitted", False):
        return {
            "profile": profile,
            "model": "ltr_xgboost_shap",
            "message": "ML-модель не обучена - объяснения недоступны. Запустите обучение пайплайна.",
            "explanations": [],
        }

    market_all = list(engine.skill_metadata.keys())
    known_lower = {s.lower().strip() for s in student_skills}
    missing = [s for s in market_all if s.lower().strip() not in known_lower]
    # Ограничение по частоте - то же, что в teacher_analysis_runner.
    missing = sorted(
        missing,
        key=lambda s: engine.skill_metadata.get(s, {}).get("freq_normalized", 0.0),
        reverse=True,
    )[:_EXPLANATION_CANDIDATE_LIMIT]

    if not missing:
        return {
            "profile": profile,
            "model": "ltr_xgboost_shap",
            "message": "Профиль уже покрывает все навыки рынка.",
            "explanations": [],
        }

    match await asyncio.to_thread(
        engine.predict_skill_impact_with_shap, student_skills, missing, True, 8
    ):
        case Ok((impacts, shap_vals, x_out)):
            explanations: list[dict[str, Any]] = []
            for i, (skill, score, text) in enumerate(impacts[:10]):
                contributions = _shap_contributions(
                    (impacts, shap_vals, x_out), i, top
                )
                explanations.append(
                    {
                        "skill": skill,
                        "score": score,
                        "explanation": text,
                        "contributions": contributions,
                    }
                )
            return {
                "profile": profile,
                "model": "ltr_xgboost_shap",
                "explanations": explanations,
            }
        case Err(e):
            logger.warning("ml_explanations_predict_failed", profile=profile, error=str(e))
            return {
                "profile": profile,
                "model": "ltr_xgboost_shap",
                "message": "Не удалось посчитать объяснения ML-модели.",
                "explanations": [],
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

    await _require_student_profile(request, profile)
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
        ) from None
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
