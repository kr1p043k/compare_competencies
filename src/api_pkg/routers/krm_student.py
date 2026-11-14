"""KRM для студентов: программа (КРМ) + свои компетенции в одном виде, без дублей.

Студент выбирает направление -> видит дисциплины/компетенции КРМ с отметками
что у него уже есть + свои навыки вне КРМ. Совпадения считаются один раз
(source: krm|student|both). Матчинг: нормализация + вхождение (тексты КРМ -
развёрнутые описания, навыки студента - короткие каноники).
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api_pkg import deps
from src.api_pkg.rate_limit import limiter
from src.api_pkg.routers.auth import require_any_role
from src.api_pkg.routers.krm_teacher import (
    _list_all_dirs,
    _load_krm_direction,
    _validate_dir_code,
)
from src.api_pkg.user_keys import self_profile_keys
from src.models.student import StudentProfile
from src.parsing.skills.skill_normalizer import SkillNormalizer

logger = structlog.get_logger("api")

router = APIRouter(tags=["krm_student"])

_STUDENT_ROLES = ("admin", "teacher", "rop", "student")


def _norm(text: str) -> str:
    try:
        r = SkillNormalizer.normalize(text or "")
        if r.is_ok() and r.ok():
            return str(r.ok()).lower().strip()
    except Exception:
        pass
    return (text or "").lower().strip()


@router.get("/krm/directions", dependencies=[Depends(require_any_role(*_STUDENT_ROLES))])
@limiter.limit("60/minute")
async def krm_directions(request: Request):
    """Все направления КРМ (для выбора студентом)."""
    return {"directions": _list_all_dirs()}


@router.get("/krm/student/competencies", dependencies=[Depends(require_any_role(*_STUDENT_ROLES))])
@limiter.limit("30/minute")
async def krm_student_competencies(
    request: Request,
    direction: str | None = Query(None, description="Код направления; по умолчанию - доступное студенту"),
    profile: str | None = Query(None, description="Профиль; по умолчанию - свой"),
    profiles: dict[str, StudentProfile] = Depends(deps.get_student_profiles),
):
    """Сводка: компетенции КРМ направления + компетенции студента, дубли - один раз.

    `direction` и `profile` необязательны. Раньше без них эндпоинт отвечал 422:
    студент не знает свой код направления (09.03.02) и имя профиля
    (self_<email>), а именно он и открывает эту страницу первым.
    """
    from src.api_pkg.routers.auth import get_current_user

    if not direction:
        user = await get_current_user(request) or {}
        allowed: list[str] = []
        try:
            from src.api_pkg.routers.krm_teacher import _user_directions

            allowed = await _user_directions(user)
        except Exception:
            allowed = []
        if not allowed:
            allowed = [d["dir_code"] for d in _list_all_dirs()]
        if not allowed:
            raise HTTPException(status_code=404, detail="Нет доступных направлений")
        direction = allowed[0]

    user = await get_current_user(request) or {}
    username = str(user.get("u") or user.get("username") or "")
    own_profiles = self_profile_keys(username)

    if not profile:
        # Профиль студента выводится self_<username в нижнем регистре>.
        profile = next((c for c in own_profiles if c in profiles), None) or ""
        if not profile:
            # Раньше здесь брался произвольный профиль (next(iter(profiles))):
            # при отсутствии своего студент видел чужие компетенции.
            raise HTTPException(status_code=404, detail="Профиль не найден")

    _validate_dir_code(direction)
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")

    # Студент читает только свой профиль; чужой существующий -> 403.
    if (user.get("r") or user.get("role")) == "student" and profile not in own_profiles:
        raise HTTPException(status_code=403, detail="Доступ к чужому профилю запрещён")
    student = profiles[profile]
    student_norms = {_norm(s) for s in (student.skills or []) if _norm(s)}

    data = _load_krm_direction(direction)
    disciplines = (data.get("disciplines") or {}) if isinstance(data, dict) else {}

    disc_out = []
    krm_items: dict[str, dict] = {}
    for disc_name, disc in disciplines.items():
        comp_out = []
        for code in (disc.get("competencies") or []):
            texts = ((disc.get("skills") or {}).get(code)) or []
            skills_out = []
            for t in texts:
                nt = _norm(str(t))
                if not nt:
                    continue
                # Студент "имеет" пункт КРМ: точное совпадение или вхождение.
                has = nt in student_norms or any(
                    s in nt or nt in s for s in student_norms if len(s) > 3 and len(nt) > 3
                )
                skills_out.append({"text": str(t), "student_has": bool(has)})
                prev = krm_items.get(nt)
                krm_items[nt] = {
                    "text": str(t),
                    "code": code,
                    "discipline": disc_name,
                    "student_has": bool(has) or bool(prev and prev["student_has"]),
                }
            comp_out.append({"code": code, "skills": skills_out})
        disc_out.append({"discipline": disc_name, "competencies": comp_out})

    # Свои навыки вне КРМ: нормализованный навык студента не покрыт ни одним пунктом.
    krm_covered = set()
    for nt in student_norms:
        if nt in krm_items or any(nt in k or k in nt for k in krm_items if len(nt) > 3 and len(k) > 3):
            krm_covered.add(nt)
    # Пустые/пробельные навыки пропускаем (как в student_norms выше), иначе они
    # попадают в merged как source="student" при counts.student=0.
    # Дедуп - по нормализованной форме: «Python» и «python» один навык, а не два.
    seen_only: set[str] = set()
    student_only: list[str] = []
    for raw in (student.skills or []):
        norm = _norm(raw)
        if not norm or norm in krm_covered or norm in seen_only:
            continue
        seen_only.add(norm)
        student_only.append(raw)
    student_only.sort()

    merged = (
        [{"skill": v["text"], "source": "both" if v["student_has"] else "krm",
          "code": v["code"], "discipline": v["discipline"]}
         for _, v in sorted(krm_items.items())]
        + [{"skill": s, "source": "student", "code": "", "discipline": ""} for s in student_only]
    )
    n_both = sum(1 for m in merged if m["source"] == "both")
    return {
        "direction": direction,
        "direction_name": data.get("direction_name", ""),
        "profile": profile,
        "disciplines": disc_out,
        "merged": merged,
        "counts": {
            "krm": len(krm_items),
            "student": len(student_norms),
            "overlap": n_both,
            "merged": len(merged),
            "student_only": len(student_only),
        },
    }
