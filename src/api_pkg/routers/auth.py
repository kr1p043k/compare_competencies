"""Auth: login/logout against DB users + sessions (asyncpg)."""

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.api_pkg.client_ip import client_ip
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["auth"])
limiter = Limiter(key_func=get_remote_address)

SECRET_KEY = config.get_secret_key()
TOKEN_TTL = 86400 * config.TOKEN_TTL_DAYS

#: Сколько неудачных попыток подряд допускается до блокировки аккаунта.
_LOCKOUT_THRESHOLD = 5
#: Потолок блокировки: дальше время не растёт, чтобы не запереть человека навсегда.
_LOCKOUT_MAX_MINUTES = 15
#: Счётчик неудач сбрасывается, если последняя была больше часа назад.
_FAILURE_WINDOW = "1 hour"
#: Целевая стоимость bcrypt. Исторические хеши в БД — cost 6 (подбираются
#: на GPU тысячами в секунду), при успешном входе молча перехешируем.
_TARGET_BCRYPT_COST = 12


def _humanize_seconds(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} сек"
    minutes = (seconds + 59) // 60
    return f"{minutes} мин" if minutes < 60 else f"{minutes // 60} ч"


def _lockout_minutes(failed_count: int) -> int:
    """Экспоненциальная задержка: 1, 2, 4, 8, ... минут (с потолком)."""
    over = max(0, failed_count - _LOCKOUT_THRESHOLD)
    return min(2 ** over, _LOCKOUT_MAX_MINUTES)


async def _register_failed_login(pool, email: str, ip: str) -> None:
    """Увеличивает счётчик неудач и при необходимости блокирует аккаунт."""
    try:
        count = await pool.fetchval(
            f"""UPDATE users
                   SET failed_login_count = CASE
                         WHEN last_failed_login_at IS NULL
                           OR last_failed_login_at < NOW() - INTERVAL '{_FAILURE_WINDOW}'
                         THEN 1
                         ELSE failed_login_count + 1
                       END,
                       last_failed_login_at = NOW()
                 WHERE lower(email) = $1
             RETURNING failed_login_count""",
            email,
        )
    except Exception as exc:  # noqa: BLE001 — сбой учёта не должен ломать ответ
        logger.error("login_failure_counter_error", email=email, error=str(exc))
        return

    if count is None:
        return

    logger.warning("login_failed", email=email, ip=ip, failed_count=count)

    if count < _LOCKOUT_THRESHOLD:
        return

    minutes = _lockout_minutes(count)
    try:
        await pool.execute(
            f"""UPDATE users
                   SET locked_until = NOW() + INTERVAL '{minutes} minutes'
                 WHERE lower(email) = $1""",
            email,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("login_lockout_error", email=email, error=str(exc))
        return

    logger.warning(
        "login_account_locked_by_bruteforce",
        email=email,
        ip=ip,
        failed_count=count,
        minutes=minutes,
    )


class LoginRequest(BaseModel):
    email: str
    password: str


def _make_token(user_id: str, email: str, role: str) -> str:
    payload = json.dumps(
        {"uid": user_id, "u": email, "r": role, "t": int(time.time()) + TOKEN_TTL},
        separators=(",", ":"),
    )
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        + "."
        + base64.urlsafe_b64encode(sig).decode().rstrip("=")
    )


def _decode_token(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64 = parts[0] + "=" * ((4 - len(parts[0]) % 4) % 4)
        sig_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload = base64.urlsafe_b64decode(payload_b64).decode()
        expected_sig = base64.urlsafe_b64decode(sig_b64)
        actual_sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        data = json.loads(payload)
        if data.get("t", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _request_token(request: Request) -> str:
    """Только заголовок Authorization или cookie.

    Раньше токен принимался ещё и из ?token= - он утекает в логи, Referer и
    историю браузера. Ни фронтенд, ни n8n так не ходят (проверено), поэтому
    параметр убран.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("token", "")


async def resolve_session(token: str) -> dict[str, Any] | None:
    """Проверить токен И активную сессию в БД; вернуть данные пользователя.

    Общий помощник для HTTP (get_current_user) и WebSocket-хендшейков: WS раньше
    проверял только подпись токена, поэтому logout/деактивация/понижение в правах
    не действовали на открытом сокете до истечения токена.
    """
    if not token:
        return None
    data = _decode_token(token)
    if data is None:
        return None
    pool = get_pool()
    if pool is None:
        return None
    try:
        row = await pool.fetchrow(
            """SELECT u.id, u.email, u.role, u.is_active
               FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = $1 AND s.logged_out_at IS NULL""",
            _hash_token(token),
        )
    except Exception:
        logger.error("resolve_session_db_error", exc_info=True)
        return None
    if row is None or not row["is_active"]:
        return None
    return {"uid": str(row["id"]), "u": row["email"], "r": row["role"], "t": data.get("t")}


async def get_current_user(request: Request) -> dict[str, Any] | None:
    token = _request_token(request)
    if not token:
        logger.warning("get_current_user_no_token", path=request.url.path)
        return None
    user = await resolve_session(token)
    if user is None:
        logger.warning("get_current_user_rejected", path=request.url.path, token_sha=_hash_token(token)[:16])
    return user


def require_role(role: str):
    async def dependency(request: Request):
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if user.get("r") != role and user.get("r") != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dependency


def require_any_role(*roles: str):
    """Allow access if user has any of the given roles (admin always allowed)."""
    async def dependency(request: Request):
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if user.get("r") != "admin" and user.get("r") not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dependency


def require_authenticated():
    """Любой залогиненный пользователь, роль не важна.

    Для read-эндпоинтов (прогнозы, вакансии, таксономия, кластеры), которые
    раньше отдавались анонимно - как выгрузка датасета в аудите gigachess.
    """
    async def dependency(request: Request):
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user
    return dependency


async def user_error_detail(request: Request | None, technical: str, user_msg: str) -> str:
    """Внутренности - только админу.

    Технические детали (тексты исключений, пути) показываем только роли admin;
    остальным - безопасное сообщение. request=None (нет контекста запроса) -
    всегда безопасное. Ошибка уже должна быть залогирована вызывающей стороной.
    """
    try:
        if request is None:
            return user_msg
        user = await get_current_user(request)
        if isinstance(user, dict) and user.get("r") == "admin" and technical:
            return technical
    except Exception:
        pass
    return user_msg


@router.post("/auth/login")
@limiter.limit("10/minute")
async def login(body: LoginRequest, request: Request):
    """Вход по email/паролю, выдача токена.

    Защита от перебора: лимит по IP (slowapi) + блокировка по аккаунту.
    После `_LOCKOUT_THRESHOLD` неудач подряд аккаунт блокируется на
    экспоненциально растущее время (до `_LOCKOUT_MAX_MINUTES`).
    Блокировка по аккаунту не зависит от IP, поэтому не обходится
    подменой заголовков или ботнетом.
    """
    pool = get_pool()
    if pool is None:
        logger.error("login_no_pool")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    ip = client_ip(request)
    ua = request.headers.get("User-Agent", "")
    email = body.email.strip().lower()

    try:
        lock = await pool.fetchrow(
            """SELECT locked_until,
                      GREATEST(0, CEIL(EXTRACT(EPOCH FROM (locked_until - NOW()))))::int AS retry_in
                 FROM users WHERE lower(email) = $1""",
            email,
        )
        if lock and lock["locked_until"] is not None and lock["retry_in"] and lock["retry_in"] > 0:
            logger.warning("login_account_locked", email=email, ip=ip, retry_in=lock["retry_in"])
            raise HTTPException(
                status_code=429,
                detail=(
                    "Слишком много неудачных попыток входа. "
                    f"Повторите через {_humanize_seconds(lock['retry_in'])}."
                ),
                headers={"Retry-After": str(lock["retry_in"])},
            )

        row = await pool.fetchrow(
            """SELECT id, email, role, full_name,
                      password_hash = crypt($2, password_hash) AS pw_match,
                      COALESCE(split_part(password_hash, '$', 3)::int, 0) AS pw_cost
                 FROM users
                WHERE lower(email) = $1 AND is_active = true""",  # noqa: E501 — SQL string must not be split
            email, body.password,
        )

        # Выравниваем время ответа. Для несуществующего email и для аккаунтов
        # со старым слабым хешем (cost 6) считаем ещё и bcrypt целевой
        # стоимости — иначе 401 для них приходит в ~9 раз быстрее и выдаёт,
        # какие адреса зарегистрированы (перечисление пользователей).
        if row is None or row["pw_cost"] < _TARGET_BCRYPT_COST:
            await pool.execute(
                f"SELECT crypt($1, gen_salt('bf', {_TARGET_BCRYPT_COST}))", body.password
            )

        if row is None or not row["pw_match"]:
            await _register_failed_login(pool, email, ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = _make_token(str(row["id"]), row["email"], row["role"])

        token_hash = _hash_token(token)
        await pool.execute(
            """INSERT INTO sessions (user_id, token_hash, ip_address, user_agent)
               VALUES ($1, $2, $3, $4)""",
            row["id"], token_hash, ip, ua,
        )

        # Успешный вход снимает блокировку и счётчик неудач.
        # Заодно обновляем слабый хеш (cost 6 -> 12), пока пароль есть в памяти.
        if row["pw_cost"] < _TARGET_BCRYPT_COST:
            await pool.execute(
                f"""UPDATE users
                       SET password_hash = crypt($1, gen_salt('bf', {_TARGET_BCRYPT_COST})),
                           failed_login_count = 0,
                           locked_until = NULL,
                           last_failed_login_at = NULL
                     WHERE id = $2""",
                body.password, row["id"],
            )
            logger.info("user_password_rehashed", email=row["email"], old_cost=row["pw_cost"])
        else:
            await pool.execute(
                """UPDATE users
                      SET failed_login_count = 0, locked_until = NULL, last_failed_login_at = NULL
                    WHERE id = $1""",
                row["id"],
            )

        logger.info("user_logged_in", email=row["email"], role=row["role"], ip=ip)
        return {
            "token": token,
            "role": row["role"],
            "name": row["full_name"],
            "username": row["email"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("login_db_error", error=str(exc))
        raise HTTPException(
            status_code=503, detail="Service temporarily unavailable"
        ) from exc


@router.post("/auth/logout")
@limiter.limit("30/minute")
async def logout(request: Request):
    """Выход (инвалидация сессии)."""
    user_data = await get_current_user(request)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = _request_token(request)
    token_hash = _hash_token(token)

    pool = get_pool()
    if pool is None:
        logger.error("logout_no_pool")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    await pool.execute(
        "UPDATE sessions SET logged_out_at = NOW() WHERE token_hash = $1 AND logged_out_at IS NULL",
        token_hash,
    )

    logger.info("user_logged_out", email=user_data.get("u"))
    return {"status": "ok", "message": "Logged out"}


@router.get("/auth/me")
@limiter.limit("60/minute")
async def me(request: Request):
    """Текущий пользователь по токену."""
    user_data = await get_current_user(request)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    pool = get_pool()
    if pool is None:
        logger.error("me_no_pool")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    row = await pool.fetchrow(
        "SELECT email, role, full_name FROM users WHERE email = $1",
        user_data["u"],
    )
    return {
        "username": user_data["u"],
        "role": user_data["r"],
        "name": row["full_name"] if row else "",
    }
