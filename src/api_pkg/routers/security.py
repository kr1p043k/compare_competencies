"""Управление двухфакторной аутентификацией администраторов.

Главное здесь - выключатель. TOTP-коды, QR-настройка и recovery-коды живут в
``auth.py`` под ``/auth/2fa/*``; этот роутер отвечает только за политику:
включена ли 2FA для роли admin и в каком состоянии находятся администраторы.

Эндпоинты отдельные от ``admin.py`` намеренно: политика безопасности не должна
зависеть от общего административного модуля и его истории изменений.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api_pkg import twofa_policy
from src.api_pkg.request_logger import audit_action
from src.api_pkg.routers.auth import get_current_user, require_any_role
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["security"])
limiter = Limiter(key_func=get_remote_address)


class TwofaAdminState(BaseModel):
    email: str
    role: str
    enrolled: bool


@router.get(
    "/admin/security/2fa",
    dependencies=[Depends(require_any_role("admin"))],
)
@limiter.limit("30/minute")
async def get_twofa_settings(request: Request):
    """Текущее состояние переключателя и сколько админов уже настроили TOTP."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    enabled = await twofa_policy.is_twofa_enabled(pool)
    enrolled, total = await _admin_enrolment_counts(pool)
    return {
        "enabled": enabled,
        "role": "admin",
        "enrolled_admins": enrolled,
        "total_admins": total,
        "note": (
            "Пока выключено, вход по паролю и TOTP не запрашивается."
            if not enabled
            else "Включено: администраторы с настроенным TOTP вводят код при входе."
        ),
    }


@router.put(
    "/admin/security/2fa",
    dependencies=[Depends(require_any_role("admin"))],
)
@limiter.limit("10/minute")
async def set_twofa_settings(
    request: Request,
    enabled: bool = Body(..., embed=True),
):
    """Включить или выключить 2FA для администраторов.

    Выключение не удаляет секреты TOTP: если включить снова, настройки
    останутся на месте и код просто опять начнут спрашивать.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    user = await get_current_user(request) or {}
    actor = str(user.get("u") or user.get("username") or "unknown")

    await twofa_policy.set_twofa_enabled(pool, enabled, actor=actor)
    await audit_action(
        request,
        "admin.security.2fa.toggle",
        f"enabled={enabled}",
    )
    enrolled, total = await _admin_enrolment_counts(pool)
    logger.info("twofa_settings_updated", enabled=enabled, actor=actor)
    return {
        "enabled": enabled,
        "enrolled_admins": enrolled,
        "total_admins": total,
    }


@router.get(
    "/admin/security/2fa/admins",
    dependencies=[Depends(require_any_role("admin"))],
)
@limiter.limit("30/minute")
async def list_admin_twofa_state(request: Request):
    """Кто из администраторов уже настроил TOTP, а кто ещё нет."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    try:
        rows = await pool.fetch(
            """
            SELECT u.email,
                   u.role,
                   (t.user_id IS NOT NULL AND t.confirmed) AS enrolled
              FROM users u
              LEFT JOIN admin_totp t ON t.user_id = u.id
             WHERE lower(u.role) = 'admin' AND u.is_active = true
             ORDER BY u.email
            """
        )
    except Exception as exc:
        # Нет таблицы admin_totp (миграция не применена) - значит никто не настроил.
        logger.warning("admin_totp_table_unavailable", error=str(exc)[:200])
        rows = await pool.fetch(
            """SELECT email, role, false AS enrolled
                 FROM users WHERE lower(role) = 'admin' AND is_active = true
                ORDER BY email"""
        )

    return {
        "admins": [
            TwofaAdminState(
                email=r["email"], role=r["role"], enrolled=bool(r["enrolled"])
            ).model_dump()
            for r in rows
        ]
    }


async def _admin_enrolment_counts(pool) -> tuple[int, int]:
    """(настроили TOTP, всего активных администраторов)."""
    try:
        row = await pool.fetchrow(
            """
            SELECT count(*) FILTER (WHERE t.user_id IS NOT NULL AND t.confirmed) AS enrolled,
                   count(*) AS total
              FROM users u
              LEFT JOIN admin_totp t ON t.user_id = u.id
             WHERE lower(u.role) = 'admin' AND u.is_active = true
            """
        )
    except Exception as exc:
        logger.warning("admin_totp_table_unavailable", error=str(exc)[:200])
        row = await pool.fetchrow(
            """SELECT 0 AS enrolled, count(*) AS total
                 FROM users WHERE lower(role) = 'admin' AND is_active = true"""
        )
    if row is None:
        return 0, 0
    return int(row["enrolled"] or 0), int(row["total"] or 0)
