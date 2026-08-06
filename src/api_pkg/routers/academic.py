"""Academic API (ЮФУ): SSO-вход по токену хаба + прокси get-competencies / analyze-gap.

Токен с hub.sfedu.ru (JWT, payload.username) приходит в ?token=<jwt>.
- /auth/sso — проверяет JWT через academic-api /verify-token, маппит username@sfedu.ru
  на users, создаёт локальную сессию с нашим токеном и сохраняет hub-JWT (sso_token).
- /academic/* — проксирует запросы в academic-api с Authorization: Bearer <hub-jwt>.
"""

import base64
import json
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src import config
from src.api_pkg.routers.auth import _hash_token, _make_token, get_current_user
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["academic"])

ACADEMIC_TIMEOUT = 60.0


class SsoRequest(BaseModel):
    token: str


class CompetenciesRequest(BaseModel):
    topic: str
    broad_top_k: int | None = 10
    final_top_k: int | None = 5


class CurrentCompetency(BaseModel):
    code: str
    description: str | None = None


class GapRequest(BaseModel):
    topic: str
    current_competencies: list[CurrentCompetency]
    broad_top_k: int | None = 10
    final_top_k: int | None = 5


async def _require_user(request: Request) -> dict[str, Any]:
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _jwt_username(token: str) -> str | None:
    """Достаёт username из payload JWT (часть [1]) без проверки подписи."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload = json.loads(base64.urlsafe_b64decode(b64).decode())
        username = payload.get("username") or payload.get("sub") or payload.get("user_name")
        return str(username) if username else None
    except Exception:
        return None


def _request_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("token", "") or request.query_params.get("token", "")


async def _get_sso_token(request: Request) -> str:
    """Находит hub-JWT, сохранённый в сессии нашего токена."""
    our_token = _request_token(request)
    if not our_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT sso_token FROM sessions WHERE token_hash = $1 AND logged_out_at IS NULL",
        _hash_token(our_token),
    )
    if row is None or not row["sso_token"]:
        raise HTTPException(
            status_code=403,
            detail="Сессия не привязана к хабу ЮФУ. Войдите через hub.sfedu.ru",
        )
    return row["sso_token"]


async def _academic_post(path: str, sso_token: str, payload: dict) -> Any:
    try:
        async with httpx.AsyncClient(timeout=ACADEMIC_TIMEOUT) as client:
            resp = await client.post(
                f"{config.ACADEMIC_API_BASE}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {sso_token}"},
            )
    except httpx.HTTPError as exc:
        logger.warning("academic_api_unreachable", path=path, error=str(exc))
        raise HTTPException(status_code=503, detail="Сервис ЮФУ недоступен, попробуйте позже") from None
    if resp.status_code == 401:
        raise HTTPException(
            status_code=419,
            detail="Доступ к сервису ЮФУ истёк (токен действует 1 час). Войдите через hub.sfedu.ru",
        )
    if resp.status_code != 200:
        logger.warning("academic_api_error", path=path, status=resp.status_code, body=resp.text[:300])
        raise HTTPException(status_code=502, detail=resp.text[:500] or "Ошибка сервиса ЮФУ")
    return resp.json()


@router.post("/auth/sso")
async def sso_login(body: SsoRequest, request: Request):
    if not config.ACADEMIC_SSO_ENABLED:
        raise HTTPException(status_code=503, detail="Вход через хаб ЮФУ отключён")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            verify = await client.post(
                f"{config.ACADEMIC_API_BASE}/verify-token",
                json={"token": body.token},
            )
    except httpx.HTTPError as exc:
        logger.warning("sso_verify_unreachable", error=str(exc))
        raise HTTPException(status_code=503, detail="Сервис ЮФУ недоступен, попробуйте позже") from None
    if verify.status_code != 200:
        logger.warning(
            "sso_verify_rejected",
            status=verify.status_code,
            body=verify.text[:300],
            username=_jwt_username(body.token),
        )
        raise HTTPException(status_code=401, detail="Недействительный или истёкший токен хаба")

    username = _jwt_username(body.token)
    if not username:
        raise HTTPException(status_code=401, detail="В токене хаба отсутствует username")
    email = f"{username}@{config.ACADEMIC_EMAIL_DOMAIN}".lower()

    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT id, email, role, full_name FROM users WHERE email = $1 AND is_active = true",
        email,
    )
    if row is None:
        logger.warning("sso_user_not_found", username=username, email=email)
        raise HTTPException(
            status_code=403,
            detail="Учётная запись ЮФУ не найдена в системе. Обратитесь к администратору.",
        )

    token = _make_token(str(row["id"]), row["email"], row["role"])
    token_hash = _hash_token(token)
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    ua = request.headers.get("User-Agent", "")
    await pool.execute(
        """INSERT INTO sessions (user_id, token_hash, ip_address, user_agent, sso_token)
           VALUES ($1, $2, $3, $4, $5)""",
        row["id"], token_hash, ip, ua, body.token,
    )

    logger.info("sso_user_logged_in", username=username, role=row["role"])
    return {
        "token": token,
        "role": row["role"],
        "name": row["full_name"],
        "username": username,
    }


@router.get("/academic/krm-competencies")
async def academic_krm_competencies(
    request: Request,
    user: dict[str, Any] = Depends(_require_user),
):
    """Все коды компетенций ОП из KRM-файла (для current_competencies в analyze-gap)."""
    try:
        data = json.loads(config.KRM_DISCIPLINES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("krm_competencies_load_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="KRM-данные недоступны") from None
    codes: set[str] = set()
    for disc in data.get("09.03.02", {}).get("disciplines", {}).values():
        codes.update(disc.get("competencies", []))
    return {"codes": sorted(codes), "count": len(codes)}


@router.post("/academic/get-competencies")
async def academic_get_competencies(
    body: CompetenciesRequest,
    request: Request,
    user: dict[str, Any] = Depends(_require_user),
):
    sso_token = await _get_sso_token(request)
    payload = {
        "topic": body.topic,
        "broad_top_k": body.broad_top_k,
        "final_top_k": body.final_top_k,
    }
    return await _academic_post("/get-competencies", sso_token, payload)


@router.post("/academic/analyze-gap")
async def academic_analyze_gap(
    body: GapRequest,
    request: Request,
    user: dict[str, Any] = Depends(_require_user),
):
    sso_token = await _get_sso_token(request)
    payload = {
        "topic": body.topic,
        "current_competencies": [
            item.model_dump(exclude_none=True) for item in body.current_competencies
        ],
        "broad_top_k": body.broad_top_k,
        "final_top_k": body.final_top_k,
    }
    return await _academic_post("/analyze-gap", sso_token, payload)
