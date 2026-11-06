"""Academic API (ЮФУ): SSO-вход по токену хаба + прокси get-competencies / analyze-gap.

Токен с hub.sfedu.ru (JWT, payload.username) приходит в ?token=<jwt>.
- /auth/sso - проверяет JWT через academic-api /verify-token, маппит username@sfedu.ru
  на users, создаёт локальную сессию с нашим токеном и сохраняет hub-JWT (sso_token).
- /academic/* - проксирует запросы в academic-api с Authorization: Bearer <hub-jwt>.
"""

import asyncio
import base64
import hashlib
import json
import time
from typing import Any

import httpx
import structlog
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.api_pkg.client_ip import client_ip
from src.api_pkg.routers.auth import (
    _hash_token,
    _make_token,
    get_current_user,
    require_any_role,
)
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["academic"])
limiter = Limiter(key_func=get_remote_address)

ACADEMIC_TIMEOUT = 60.0
# analyze-gap - тяжёлый LLM-запрос (анализ разрыва всей КРМ), может идти минуты.
ACADEMIC_GAP_TIMEOUT = 300.0

# academic-api (nginx/WAF) блокирует запросы от скриптов с python-httpx User-Agent (403).
# Ходим с браузерным UA, иначе все вызовы падают с 403 Forbidden.
_ACADEMIC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    )
}


# /auth/sso принимает hub-JWT в теле; cap защищает _jwt_username (base64-декод)
# и внешний verify-token от патологически длинных токенов. JWT хаба ~1-2 КБ.
_MAX_SSO_TOKEN_LEN = 4096


class SsoRequest(BaseModel):
    # JWT не бывает длиннее нескольких килобайт: ограничение снимает
    # бесплатный CPU-усилитель на разборе и логировании гигантской строки.
    token: str = Field(min_length=1, max_length=4096)


class CompetenciesRequest(BaseModel):
    topic: str
    broad_top_k: int | None = 10
    final_top_k: int | None = None


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

# Общий объект зависимости: ruff B008 не любит вызов Depends() прямо в
# значении по умолчанию, а FastAPI нормально переиспользует один и тот же.
_REQUIRE_USER = Depends(_require_user)


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


def _jwt_exp(token: str) -> int | None:
    """Возвращает exp (unix) из payload JWT или None, если поле отсутствует."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload = json.loads(base64.urlsafe_b64decode(b64).decode())
        exp = payload.get("exp")
        return int(exp) if isinstance(exp, (int, float)) else None
    except Exception:
        return None


def _request_token(request: Request) -> str:
    """Токен только из заголовка или cookie.

    Раньше принимался ?token= - он попадает в логи, Referer и историю браузера.
    Hub передаёт JWT во фрагменте (#token=...), который до сервера не доходит,
    поэтому query-вариант не нужен.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("token", "")


# ---------- hub-JWT: шифрование в покое ----------
# sessions.sso_token - это JWT хаба ЮФУ, живущий час. В открытом виде он
# бесполезен только до первого чтения БД: любой, кто видит таблицу, получает
# рабочий токен. Шифруем Fernet-ключом, выведенным из SECRET_KEY (HKDF-подобно:
# sha256(secret + соль)). Свои сессионные токены мы храним хешем, тут хеш не
# годится - hub-JWT нужно предъявлять в academic-api.

_FERNET_SALT = b"compare-online/sso-token/v1"


def _fernet() -> Fernet:
    secret = config.get_secret_key()
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode() + _FERNET_SALT).digest())
    return Fernet(key)


def _encrypt_sso(token: str) -> str:
    return _fernet().encrypt(token.encode()).decode()


def _decrypt_sso(stored: str) -> str | None:
    """Расшифровать hub-JWT. None - если шифр не наш (legacy plaintext)."""
    try:
        return _fernet().decrypt(stored.encode()).decode()
    except (InvalidToken, ValueError):
        return None


async def _get_sso_token(request: Request) -> tuple[str, str]:
    """Находит hub-JWT, сохранённый в сессии нашего токена.

    Возвращает (sso_token, token_hash). Если hub-JWT уже истёк (exp <= now) -
    помечает сессию logged_out_at и кидает 419, чтобы фронт перешёл на хаб.
    """
    our_token = _request_token(request)
    if not our_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    token_hash = _hash_token(our_token)
    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT sso_token FROM sessions WHERE token_hash = $1 AND logged_out_at IS NULL",
        token_hash,
    )
    if row is None or not row["sso_token"]:
        raise HTTPException(
            status_code=403,
            detail="Сессия не привязана к хабу ЮФУ. Войдите через hub.sfedu.ru",
        )
    sso_token = _decrypt_sso(row["sso_token"]) or row["sso_token"]
    exp = _jwt_exp(sso_token)
    if exp is not None and exp <= time.time():
        await pool.execute(
            "UPDATE sessions SET logged_out_at = now() WHERE token_hash = $1 AND logged_out_at IS NULL",
            token_hash,
        )
        raise HTTPException(
            status_code=419,
            detail="Доступ к сервису ЮФУ истёк (токен действует 1 час). Войдите через hub.sfedu.ru",
        )
    return sso_token, token_hash


async def _user_direction_codes(user: dict[str, Any]) -> list[str]:
    """Коды направлений пользователя (admin - все KRM-файлы, иначе user_directions)."""
    role = user.get("r")
    pool = get_pool()
    if role != "admin" and pool is not None:
        rows = await pool.fetch(
            "SELECT dir_code FROM user_directions WHERE user_id=$1 ORDER BY dir_code",
            user.get("uid"),
        )
        codes = [r["dir_code"] for r in rows]
        if codes:
            return codes
    dirs: list[str] = []
    if config.REFERENCE_DIR.exists():
        for path in sorted(config.REFERENCE_DIR.glob("krm_disciplines_*.json")):
            if "_clean" in path.name:
                continue
            dir_code = path.name[len("krm_disciplines_"):-len(".json")]
            if dir_code:
                dirs.append(dir_code)
    return dirs


def _local_competencies(topic: str, final_top_k: int | None, dir_codes: list[str]) -> dict[str, Any]:
    """Локальные рекомендации компетенций (AcademicGapAnalyzer.recommend)."""
    from src.analyzers.academic_gap import AcademicGapAnalyzer

    lag = [d for d in dir_codes or []]
    lag = lag or ["09.03.02"]
    result: dict[str, Any] = {}
    for dir_code in lag:
        analyzer = AcademicGapAnalyzer(dir_code=dir_code)
        result = analyzer.recommend(topic, final_top_k=final_top_k)
        if result.get("recommended_competencies") or result.get("found_trends"):
            break
    return result


async def _academic_post(
    path: str,
    sso_token: str,
    payload: dict,
    token_hash: str | None = None,
    timeout: float | None = None,
) -> Any:
    timeout = timeout or ACADEMIC_TIMEOUT
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_ACADEMIC_HEADERS) as client:
            resp = await client.post(
                f"{config.ACADEMIC_API_BASE}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {sso_token}"},
            )
    except httpx.ReadTimeout as exc:
        logger.warning("academic_api_timeout", path=path, timeout=timeout, error=str(exc))
        raise HTTPException(
            status_code=504,
            detail="Сервис ЮФУ обрабатывает запрос дольше обычного, попробуйте позже",
        ) from None
    except httpx.HTTPError as exc:
        logger.warning("academic_api_unreachable", path=path, error=str(exc))
        raise HTTPException(status_code=503, detail="Сервис ЮФУ недоступен, попробуйте позже") from None
    if resp.status_code == 401:
        if token_hash:
            pool = get_pool()
            await pool.execute(
                "UPDATE sessions SET logged_out_at = now() WHERE token_hash = $1 AND logged_out_at IS NULL",
                token_hash,
            )
        raise HTTPException(
            status_code=419,
            detail="Доступ к сервису ЮФУ истёк (токен действует 1 час). Войдите через hub.sfedu.ru",
        )
    if resp.status_code != 200:
        logger.warning("academic_api_error", path=path, status=resp.status_code, body=resp.text[:300])
        raise HTTPException(status_code=502, detail=resp.text[:500] or "Ошибка сервиса ЮФУ")
    return resp.json()


@router.post("/auth/sso")
@limiter.limit("10/minute")
async def sso_login(body: SsoRequest, request: Request, response: Response):
    """SSO-вход через академический провайдер.

    Лимит 10/мин: эндпоинт ходит во внешний academic-api и создаёт сессию,
    без лимита его можно долбить перебором hub-JWT.
    """
    if not config.ACADEMIC_SSO_ENABLED:
        raise HTTPException(status_code=503, detail="Вход через хаб ЮФУ отключён")

    try:
        async with httpx.AsyncClient(timeout=30, headers=_ACADEMIC_HEADERS) as client:
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

    # 2FA обязательна и для SSO: иначе владелец hub-JWT админа получал бы
    # полный токен, не зная TOTP (второй фактор обходился стороной).
    # Мастер-переключатель действует одинаково для пароля и SSO: выключен -
    # код не спрашиваем ни у кого, включён - у всех подключённых.
    if row["role"] == "admin":
        from src.api_pkg.routers.auth import (
            _issue_challenge,
            _issue_enroll,
            _totp_state,
            get_two_factor_enabled,
        )

        if await get_two_factor_enabled(pool):
            state = await _totp_state(pool, row["id"])
            if state and state.get("confirmed"):
                challenge = _issue_challenge(str(row["id"]))
                logger.info("sso_totp_required", username=username)
                return {
                    "status": "totp_required",
                    "challenge_id": challenge,
                    "expires_in": 300,
                }
            logger.info("sso_totp_setup_required", username=username)
            return {
                "status": "totp_setup_required",
                "enroll_id": _issue_enroll(str(row["id"])),
                "expires_in": 600,
            }

    token = _make_token(str(row["id"]), row["email"], row["role"])
    token_hash = _hash_token(token)
    ip = client_ip(request)
    ua = request.headers.get("User-Agent", "")
    await pool.execute(
        """INSERT INTO sessions (user_id, token_hash, ip_address, user_agent, sso_token)
           VALUES ($1, $2, $3, $4, $5)""",
        row["id"], token_hash, ip, ua, _encrypt_sso(body.token),
    )

    logger.info("sso_user_logged_in", username=username, role=row["role"])
    # SSO тоже ставит HttpOnly-cookie: иначе фронту пришлось бы держать токен
    # в localStorage только ради этого пути входа.
    from src.api_pkg.routers.auth import _set_session_cookie

    _set_session_cookie(response, token)
    return {
        "token": token,
        "role": row["role"],
        "name": row["full_name"],
        "username": username,
    }


@router.get("/academic/krm-competencies")
@limiter.limit("60/minute")
async def academic_krm_competencies(
    request: Request,
    dir_code: str | None = None,
    user: dict[str, Any] = _REQUIRE_USER,
):
    """Все коды компетенций ОП из БД (для current_competencies в analyze-gap).

    Параметр dir_code (опционально): фильтр по направлению.
    Без параметра - все уникальные коды компетенций.
    """
    pool = get_pool()
    if dir_code:
        rows = await pool.fetch(
            """SELECT DISTINCT c.code
               FROM competencies c
               JOIN disciplines disc ON disc.id = c.discipline_id
               JOIN directions d ON d.id = disc.direction_id
               WHERE d.code = $1
               ORDER BY c.code""",
            dir_code,
        )
    else:
        rows = await pool.fetch(
            "SELECT DISTINCT code FROM competencies ORDER BY code"
        )
    codes = [r["code"] for r in rows]
    return {"codes": codes, "count": len(codes)}


@router.post("/academic/get-competencies")
@limiter.limit("10/minute")
async def academic_get_competencies(
    body: CompetenciesRequest,
    request: Request,
    user: dict[str, Any] = _REQUIRE_USER,
):
    """Компетенции направления из БД.

    При недоступности SSO (нет привязки к хабу, истёк токен) или сбое сервиса
    ЮФУ отдаются локальные рекомендации по направлению пользователя.
    """
    fallback_codes = await _user_direction_codes(user)
    try:
        sso_token, token_hash = await _get_sso_token(request)
    except HTTPException as exc:
        if exc.status_code in (403, 419):
            logger.info("academic_fallback_local", reason="sso_unavailable", status=exc.status_code)
            return await asyncio.to_thread(
                _local_competencies, body.topic, body.final_top_k, fallback_codes
            )
        raise
    payload = {
        "topic": body.topic,
        "broad_top_k": body.broad_top_k,
        "final_top_k": body.final_top_k,
    }
    try:
        return await _academic_post("/get-competencies", sso_token, payload, token_hash)
    except HTTPException as exc:
        if exc.status_code in (502, 503, 504):
            logger.info("academic_fallback_local", reason="academic_api_error", status=exc.status_code)
            return await asyncio.to_thread(
                _local_competencies, body.topic, body.final_top_k, fallback_codes
            )
        raise


@router.post("/academic/analyze-gap", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("10/minute")
async def academic_analyze_gap(
    body: GapRequest,
    request: Request,
    user: dict[str, Any] = _REQUIRE_USER,
):
    """Анализ разрыва по теме (AcademicGapAnalyzer)."""
    sso_token, token_hash = await _get_sso_token(request)
    payload = {
        "topic": body.topic,
        "current_competencies": [
            item.model_dump(exclude_none=True) for item in body.current_competencies
        ],
        "broad_top_k": body.broad_top_k,
        "final_top_k": body.final_top_k,
    }
    return await _academic_post(
        "/analyze-gap", sso_token, payload, token_hash,
        timeout=ACADEMIC_GAP_TIMEOUT,
    )


@router.post("/academic/analyze-gap-local", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("10/minute")
async def academic_analyze_gap_local(
    body: CompetenciesRequest,
    request: Request,
    user: dict[str, Any] = _REQUIRE_USER,
):
    """Собственный анализ разрыва (без сервиса ЮФУ).

    Тема → навыки → сравнение с рынком (it_skills) и компетенциями КРМ.
    Выполняется в фоне в потоке (эмбеддинги). Формат ответа - как у ЮФУ.
    """
    from src.analyzers.academic_gap import AcademicGapAnalyzer

    try:
        result = await asyncio.to_thread(AcademicGapAnalyzer().analyze, body.topic)
    except Exception as exc:
        logger.error("analyze_gap_local_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Не удалось выполнить локальный анализ разрыва",
        ) from None
    return result
