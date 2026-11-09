"""Auth: login/logout against DB users + sessions (asyncpg)."""

import base64
import hashlib
import hmac
import json
import secrets
import struct
import time
import urllib.parse
from typing import Any

import structlog
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from slowapi import Limiter

from src import config
from src.api_pkg.client_ip import client_ip
from src.api_pkg.request_logger import audit_action
from src.api_pkg.security_settings import get_two_factor_enabled
from src.api_pkg.twofa_token_store import (
    consume_challenge,
    consume_enroll,
    issue_challenge,
    issue_enroll,
    peek_enroll,
)
from src.db import get_pool

try:
    import pyotp
    _PYOTP_AVAILABLE = True
except ImportError:  # pragma: no cover - RFC 6238 stdlib fallback
    pyotp = None  # type: ignore[assignment]
    _PYOTP_AVAILABLE = False

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["auth"])


def _login_rate_key(request: Request) -> str:
    """Ключ лимита входа: реальный IP клиента.

    `get_remote_address` берёт TCP-peer, а за nginx это один и тот же адрес на
    всех — лимит становится общим и превращается в DoS. `client_ip` не доверяет
    заголовку от недоверенных соединений, поэтому его нельзя обойти.
    """
    return f"login:{client_ip(request)}"


limiter = Limiter(key_func=_login_rate_key)

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




# ---------------- TOTP 2FA для роли admin ----------------
#
# Секрет TOTP - такой же секрет, как пароль: без шифрования он даёт полный
# доступ к аккаунту. Шифруем Fernet-ключом, производным от SECRET_KEY (как
# hub-JWT в academic.py). Recovery-коды храним только хэшами.

_TOTP_ISSUER = "Compare Competencies"
_TOTP_FERNET_SALT = b"compare-online/admin-totp/v1"
_TOTP_CHALLENGE_TTL = 300  # 5 минут на ввод кода
_TOTP_MAX_FAILURES = 5
_TOTP_LOCK_MINUTES = 15
_RECOVERY_CODE_COUNT = 10
_ENROLL_TTL = 600  # 10 минут на первичную настройку при включённом 2FA
_TOTP_STEP_SECONDS = 30  # RFC 6238: длина шага TOTP (окно кода)


def _totp_fernet() -> Fernet:
    key = base64.urlsafe_b64encode(
        hashlib.sha256(SECRET_KEY.encode() + _TOTP_FERNET_SALT).digest()
    )
    return Fernet(key)


def _encrypt_totp_secret(secret: str) -> str:
    return _totp_fernet().encrypt(secret.encode()).decode()


def _decrypt_totp_secret(stored: str) -> str | None:
    try:
        return _totp_fernet().decrypt(stored.encode()).decode()
    except (InvalidToken, ValueError):
        return None


def _hash_recovery_code(code: str) -> str:
    """Хеш резервного кода для сравнения с сохранённым.

    Коды это hex-строки; на мобильной клавиатуре их часто вводят в верхнем
    регистре, поэтому нормализуем регистр и обрезаем пробелы, иначе верный код
    не совпадает с хешем и вход не проходит.
    """
    return hashlib.sha256(f"{code.strip().lower()}:{SECRET_KEY}".encode()).hexdigest()


def _generate_totp_secret() -> str:
    if _PYOTP_AVAILABLE:
        return pyotp.random_base32()
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _totp_now(secret: str) -> str:
    """Текущий TOTP-код (для тестов и восстановления)."""
    if _PYOTP_AVAILABLE:
        return pyotp.TOTP(secret).now()
    # RFC 6238: 30-секундные окна, 6 цифр, SHA-1.
    counter = int(time.time()) // _TOTP_STEP_SECONDS
    digest = hmac.new(
        base64.b32decode(secret + "=" * (-len(secret) % 8)),
        struct.pack(">Q", counter),
        hashlib.sha1,
    ).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return f"{binary % 1_000_000:06d}"


def _verify_totp(secret: str, code: str) -> bool:
    code = (code or "").strip().replace(" ", "")
    if not code:
        return False
    if _PYOTP_AVAILABLE:
        # valid_window=1: допускаем соседнее окно на рассинхрон часов.
        return bool(pyotp.TOTP(secret).verify(code, valid_window=1))
    return hmac.compare_digest(_totp_now(secret), code)


def _two_factor_disabled() -> HTTPException:
    """Рубильник выключен: эндпоинт есть, но настройкой запрещён."""
    return HTTPException(
        status_code=403,
        detail="Двухфакторная аутентификация отключена администратором",
    )


async def _issue_login_response(
    pool, user_row, request: Request, response: Response
) -> dict:
    """Выдать сессию и токен (общий финал login и 2FA-verify).

    Cookie ставится здесь же: иначе вход через 2FA выдавал бы токен только в
    теле ответа, и фронт снова держал бы его в localStorage.
    """
    token = _make_token(str(user_row["id"]), user_row["email"], user_row["role"])
    await pool.execute(
        """INSERT INTO sessions (user_id, token_hash, ip_address, user_agent)
           VALUES ($1, $2, $3, $4)""",
        user_row["id"], _hash_token(token), client_ip(request),
        request.headers.get("User-Agent", ""),
    )
    _set_session_cookie(response, token)
    return {
        "token": token,
        "role": user_row["role"],
        "name": user_row["full_name"],
        "username": user_row["email"],
    }


async def _totp_state(pool, user_id) -> dict | None:
    """Состояние 2FA пользователя.

    Различаем два случая:
      * строки нет -> None (2FA не настроена, вход по паролю);
      * ошибка БД/таблицы -> TotpUnavailable (fail-closed).

    Раньше любая ошибка трактовалась как «2FA не настроена», и сбой БД или
    неприменённая миграция снимали второй фактор со всех админов.
    """
    try:
        row = await pool.fetchrow(
            """SELECT secret, confirmed, recovery_codes, failed_attempts, locked_until,
                      GREATEST(0, CEIL(EXTRACT(EPOCH FROM (locked_until - NOW()))))::int AS retry_in
                 FROM admin_totp WHERE user_id = $1""",
            user_id,
        )
    except Exception:
        # Сбой чтения 2FA-состояния не должен молча выключать второй фактор:
        # "записи нет" и "запрос упал" - разные исходы. Падаем закрыто: токен
        # не выдаём, клиенту - 503. Только реально отсутствующая строка значит
        # "2FA не настроена".
        logger.exception("admin_totp_query_failed", user_id=user_id)
        raise HTTPException(
            status_code=503,
            detail="Не удалось проверить второй фактор, повторите позже",
        ) from None
    return dict(row) if row else None


def _lockout_minutes_totp(failures: int) -> int:
    """Блокировка растёт с числом неудач: 3, 6, 9, 12, 15 минут (потолок)."""
    return min(_TOTP_LOCK_MINUTES, max(1, failures * 3))


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
    email: str = Field(max_length=320)
    # bcrypt читает только первые 72 байта, но cost-12 на длинной строке
    # тратит CPU зря: без границы это бесплатный усилитель для DoS.
    password: str = Field(min_length=1, max_length=256)


def _make_token(user_id: str, email: str, role: str) -> str:
    """Собрать сессионный токен.

    В полезную нагрузку кладём случайный одноразовый nonce: без него токен
    полностью детерминирован (uid+почта+роль+секунда), и два входа одного
    админа в одну секунду дают ОДИН токен - то есть одну сессию. Тогда выход
    на одном устройстве выкидывает и второе, а отзыв сессий не работает.
    """
    payload = json.dumps(
        {
            "uid": user_id,
            "u": email,
            "r": role,
            "t": int(time.time()) + TOKEN_TTL,
            "n": secrets.token_urlsafe(9),
        },
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


def _set_session_cookie(response: Response, token: str) -> None:
    """Поставить сессионную cookie с HttpOnly.

    Токен в localStorage доступен любому JS на странице: одна XSS - и сессия
    угнана. Cookie ставит сервер и помечает HttpOnly, поэтому JS её не читает.
    SameSite=Lax закрывает CSRF для обычных переходов; Secure включаем только
    на HTTPS (на localhost по http cookie с Secure не сохранится).
    """
    secure = bool(getattr(config, "COOKIE_SECURE", False))
    response.set_cookie(
        key="token",
        value=token,
        max_age=TOKEN_TTL,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )


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
async def login(body: LoginRequest, request: Request, response: Response):
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

        # 2FA включается мастер-переключателем в админке (по умолчанию
        # выключена). Выключена - входим по паролю, как раньше. Включена:
        # у подключившего TOTP пароля мало (одноразовый challenge), а
        # неподключённому выдаём пропуск на настройку вместо сессии.
        if row["role"] == "admin" and await get_two_factor_enabled(pool):
            state = await _totp_state(pool, row["id"])
            if state and state.get("confirmed"):
                challenge = await issue_challenge(pool, str(row["id"]))
                logger.info("login_totp_required", email=row["email"], ip=ip)
                return {
                    "status": "totp_required",
                    "challenge_id": challenge,
                    "expires_in": _TOTP_CHALLENGE_TTL,
                }
            logger.info("login_totp_setup_required", email=row["email"], ip=ip)
            return {
                "status": "totp_setup_required",
                "enroll_id": await issue_enroll(pool, str(row["id"])),
                "expires_in": _ENROLL_TTL,
            }

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

        _set_session_cookie(response, token)
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
async def logout(request: Request, response: Response):
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

    response.delete_cookie("token", path="/")
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
    totp_enabled = False
    totp_locked_until = None
    try:
        state = await _totp_state(pool, user_data["uid"])
        if state:
            totp_enabled = bool(state.get("confirmed"))
            locked = state.get("locked_until")
            totp_locked_until = locked.isoformat() if locked is not None else None
    except Exception as exc:
        # Таблицы admin_totp может не быть до первой настройки 2FA.
        logger.warning("me_totp_state_unavailable", error=str(exc)[:200])
    return {
        "username": user_data["u"],
        "role": user_data["r"],
        "name": row["full_name"] if row else "",
        "totp_enabled": totp_enabled,
        # Панель показывает «заблокировано до», если код вводили неверно много раз.
        "totp_locked_until": totp_locked_until,
    }


async def _check_totp_or_recovery(
    pool, user_id, state: dict, code: str, invalid_status: int = 400
) -> None:
    """Проверить код 2FA: TOTP или recovery, с блокировкой после N ошибок.

    Вызывается из входа (verify) и из отключения (disable), поэтому живёт
    отдельно: иначе логика блокировки дублируется и расходится. Поднимает
    429 при блокировке и 400 при неверном коде.
    """
    retry_in = int(state.get("retry_in") or 0)
    if state.get("locked_until") is not None and retry_in > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Слишком много неверных кодов. Повторите через {_humanize_seconds(retry_in)}.",
            headers={"Retry-After": str(retry_in)},
        )

    secret = _decrypt_totp_secret(state["secret"])
    ok = bool(secret) and _verify_totp(secret, code)
    used_recovery = False

    if not ok:
        raw = state.get("recovery_codes") or []
        codes: list[str] = json.loads(raw) if isinstance(raw, str) else list(raw)
        target = _hash_recovery_code(code.strip())
        if target in codes:
            remaining = [c for c in codes if c != target]
            # Атомарно: гасим код только если он ещё в наборе. Без условия
            # `@>` два параллельных запроса с одним кодом оба прошли бы
            # read-check-write и погасили один recovery-код дважды.
            status = await pool.execute(
                """UPDATE admin_totp
                      SET recovery_codes = $3::jsonb
                    WHERE user_id = $1 AND recovery_codes @> $2::jsonb""",
                user_id, json.dumps([target]), json.dumps(remaining),
            )
            if status.split()[-1] == "1":
                ok = True
                used_recovery = True

    if not ok:
        failures = int(state.get("failed_attempts") or 0) + 1
        locked = failures >= _TOTP_MAX_FAILURES
        await pool.execute(
            """UPDATE admin_totp
                  SET failed_attempts = $2,
                      locked_until = CASE WHEN $3 THEN NOW() + ($4 || ' minutes')::interval
                                          ELSE locked_until END
                WHERE user_id = $1""",
            user_id, failures, locked, str(_lockout_minutes_totp(failures)),
        )
        if locked:
            logger.warning("totp_locked", uid=str(user_id), failures=failures)
            retry_after = _lockout_minutes_totp(failures) * 60
            raise HTTPException(
                status_code=429,
                detail="Слишком много неверных кодов",
                headers={"Retry-After": str(retry_after)},
            )
        raise HTTPException(status_code=invalid_status, detail="Неверный код")

    if not used_recovery:
        await pool.execute(
            "UPDATE admin_totp SET failed_attempts = 0, locked_until = NULL WHERE user_id = $1",
            user_id,
        )


# ---------------- 2FA endpoints ----------------

class TotpSetupResponse(BaseModel):
    otpauth_uri: str
    secret: str
    enroll_id: str | None = None


class TotpSetupRequest(BaseModel):
    enroll_id: str | None = Field(default=None, max_length=128)


class TotpCodeRequest(BaseModel):
    code: str = Field(min_length=4, max_length=32)
    enroll_id: str | None = Field(default=None, max_length=128)


class TotpChallengeRequest(BaseModel):
    challenge_id: str = Field(min_length=8, max_length=128)
    code: str = Field(min_length=4, max_length=32)


async def _require_admin(request: Request) -> dict:
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if user.get("r") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return user


async def _resolve_2fa_actor(request: Request, enroll_id: str | None) -> dict[str, Any]:
    """Кто настраивает 2FA: админ с сессией или админ с пропуском enroll_id.

    Второй путь нужен при включённом мастер-переключателе: сессии ещё нет,
    пока TOTP не настроен.
    """
    user = await get_current_user(request)
    if user is not None:
        if user.get("r") != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    if not enroll_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    user_id = await peek_enroll(pool, enroll_id)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    row = await pool.fetchrow(
        "SELECT id, email, role FROM users WHERE id = $1 AND is_active = true",
        user_id,
    )
    if row is None or row["role"] != "admin":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"uid": str(row["id"]), "u": row["email"], "r": row["role"]}


async def _ensure_totp_table(pool) -> None:
    """Локальная страховка: таблица (и её колонки) нужна и до применения миграции."""
    await pool.execute(
        """CREATE TABLE IF NOT EXISTS admin_totp (
               user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
               secret TEXT NOT NULL,
               confirmed BOOLEAN NOT NULL DEFAULT false,
               created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
               last_used_at TIMESTAMPTZ,
               last_totp_step BIGINT,
               recovery_codes JSONB,
               failed_attempts INTEGER NOT NULL DEFAULT 0,
               locked_until TIMESTAMPTZ
           )"""
    )
    # CREATE IF NOT EXISTS не добавляет колонку к уже существующей таблице,
    # поэтому добавляем её отдельно - иначе защита от повтора TOTP молча
    # выключится на БД, созданной до появления last_totp_step.
    await pool.execute(
        "ALTER TABLE admin_totp ADD COLUMN IF NOT EXISTS last_totp_step BIGINT"
    )


@router.get("/auth/2fa/policy")
@limiter.limit("60/minute")
async def totp_policy(request: Request):
    """Публичный статус 2FA: показывать ли шаг кода на экране входа.

    Секретов не отдаёт - только сам факт, что защита включена администратором.
    """
    pool = get_pool()
    enabled = await get_two_factor_enabled(pool) if pool is not None else False
    return {"enabled": enabled}


@router.get("/auth/2fa/status")
@limiter.limit("60/minute")
async def totp_status(request: Request):
    """Состояние 2FA для текущего админа (мастер-флаг + личное подключение)."""
    user = await _require_admin(request)
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    enabled = await get_two_factor_enabled(pool)
    state = await _totp_state(pool, user["uid"])
    confirmed = bool(state and state.get("confirmed"))
    return {
        # Каноничные имена.
        "enabled": enabled,
        "enrolled": confirmed,
        "available": True,
        # Совместимость: панель админ-консоли и её тесты читают эти имена.
        # Значения те же самые, дублирование только в именах полей.
        "globally_enabled": enabled,
        "configured": state is not None,
        "confirmed": confirmed,
    }


@router.post("/auth/2fa/setup", response_model=TotpSetupResponse)
@limiter.limit("10/minute")
async def totp_setup(request: Request, body: TotpSetupRequest | None = None):
    """Начать настройку TOTP. Секрет показывается один раз."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    if not await get_two_factor_enabled(pool):
        raise _two_factor_disabled()

    enroll_id = body.enroll_id if body else None
    user = await _resolve_2fa_actor(request, enroll_id)
    await _ensure_totp_table(pool)

    state = await _totp_state(pool, user["uid"])
    if state and state.get("confirmed"):
        raise HTTPException(
            status_code=409, detail="2FA уже настроена; сначала отключите её"
        )

    secret = _generate_totp_secret()
    await pool.execute(
        """INSERT INTO admin_totp (user_id, secret, confirmed, failed_attempts)
           VALUES ($1, $2, false, 0)
           ON CONFLICT (user_id) DO UPDATE
              SET secret = EXCLUDED.secret, confirmed = false,
                  recovery_codes = NULL, failed_attempts = 0, locked_until = NULL""",
        user["uid"], _encrypt_totp_secret(secret),
    )
    label = urllib.parse.quote(user["u"])
    issuer = urllib.parse.quote(_TOTP_ISSUER)
    await audit_action(request, "auth.2fa.setup", f"uid={user['uid']}")
    return TotpSetupResponse(
        otpauth_uri=f"otpauth://totp/{issuer}:{label}?secret={secret}&issuer={issuer}",
        secret=secret,
        enroll_id=enroll_id,
    )


@router.post("/auth/2fa/confirm")
@limiter.limit("10/minute")
async def totp_confirm(body: TotpCodeRequest, request: Request, response: Response):
    """Подтвердить код и получить recovery-коды (единственный раз)."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    if not await get_two_factor_enabled(pool):
        raise _two_factor_disabled()

    user = await _resolve_2fa_actor(request, body.enroll_id)
    state = await _totp_state(pool, user["uid"])
    if not state:
        raise HTTPException(status_code=400, detail="Сначала вызовите /auth/2fa/setup")

    secret = _decrypt_totp_secret(state["secret"])
    if secret is None or not _verify_totp(secret, body.code):
        raise HTTPException(status_code=400, detail="Неверный код")

    # Настройку завершает именно этот вызов: пропуск одноразовый.
    enroll_used = body.enroll_id is not None and (
        await consume_enroll(pool, body.enroll_id)
    ) is not None

    # 8 байт (64 бита), а не 4: 32-битные коды вскрываются перебором при
    # утечке БД вместе с SECRET_KEY (хэш выводится из него).
    codes = [secrets.token_hex(8) for _ in range(_RECOVERY_CODE_COUNT)]
    hashes = [_hash_recovery_code(c) for c in codes]
    await pool.execute(
        """UPDATE admin_totp
              SET confirmed = true, recovery_codes = $2::jsonb,
                  last_used_at = NOW(), failed_attempts = 0, locked_until = NULL
            WHERE user_id = $1""",
        user["uid"], json.dumps(hashes),
    )
    await audit_action(request, "auth.2fa.enabled", f"uid={user['uid']}")
    payload: dict[str, Any] = {"status": "enabled", "recovery_codes": codes}
    if enroll_used:
        # Админ настраивал TOTP по пропуску, сессии у него ещё нет - вход
        # завершаем сразу, чтобы фронтенд не гонял его по кругу.
        row = await pool.fetchrow(
            "SELECT id, email, role, full_name FROM users WHERE id = $1 AND is_active = true",
            user["uid"],
        )
        if row is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        payload.update(await _issue_login_response(pool, row, request, response))
    return payload


@router.post("/auth/2fa/verify")
@limiter.limit("10/minute")
async def totp_verify(body: TotpChallengeRequest, request: Request, response: Response):
    """Завершить вход: проверить TOTP по challenge_id и выдать токен."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    if not await get_two_factor_enabled(pool):
        raise _two_factor_disabled()

    user_id = await consume_challenge(pool, body.challenge_id)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Challenge истёк или неверен")

    state = await _totp_state(pool, user_id)
    if not state or not state.get("confirmed"):
        raise HTTPException(status_code=401, detail="2FA не настроена")

    await _check_totp_or_recovery(pool, user_id, state, body.code, invalid_status=401)

    row = await pool.fetchrow(
        "SELECT id, email, role, full_name FROM users WHERE id = $1 AND is_active = true",
        user_id,
    )
    if row is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Защита от повтора TOTP: принимаем код только если его шаг строго новее
    # последнего принятого. Один UPDATE делает это атомарно - без
    # read-check-write гонки, когда один код проходит дважды.
    step = int(time.time()) // _TOTP_STEP_SECONDS
    claimed = await pool.fetchval(
        """UPDATE admin_totp
              SET last_totp_step = $2, last_used_at = NOW(), failed_attempts = 0
            WHERE user_id = $1
              AND (last_totp_step IS NULL OR last_totp_step < $2)
        RETURNING user_id""",
        user_id, step,
    )
    if claimed is None:
        logger.warning("totp_replay_rejected", uid=user_id)
        raise HTTPException(status_code=401, detail="Код уже использован")

    await audit_action(request, "auth.2fa.login", f"uid={user_id}")
    logger.info("user_logged_in_2fa", email=row["email"], role=row["role"])
    return await _issue_login_response(pool, row, request, response)


@router.post("/auth/2fa/recovery")
@limiter.limit("10/minute")
async def totp_recovery(body: TotpChallengeRequest, request: Request, response: Response):
    """Вход по одноразовому recovery-коду (когда телефон недоступен)."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    if not await get_two_factor_enabled(pool):
        raise _two_factor_disabled()

    user_id = await consume_challenge(pool, body.challenge_id)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Challenge истёк или неверен")

    state = await _totp_state(pool, user_id)
    if not state or not state.get("confirmed"):
        raise HTTPException(status_code=401, detail="2FA не настроена")

    target = _hash_recovery_code(body.code.strip())
    # Одноразовый и атомарный расход: одна UPDATE удаляет код из jsonb-массива
    # и помечает использование. Если кода нет (или он уже потрачен), RETURNING
    # не вернёт строку - отклоняем 401. Так исчезает read-check-write гонка,
    # когда два запроса успевали израсходовать один код дважды.
    consumed = await pool.fetchval(
        """UPDATE admin_totp
              SET recovery_codes = recovery_codes - $2::text,
                  last_used_at = NOW()
            WHERE user_id = $1 AND recovery_codes ? $2::text
        RETURNING user_id""",
        user_id, target,
    )
    if consumed is None:
        raise HTTPException(status_code=401, detail="Неверный recovery-код")

    row = await pool.fetchrow(
        "SELECT id, email, role, full_name FROM users WHERE id = $1 AND is_active = true",
        user_id,
    )
    if row is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    await audit_action(request, "auth.2fa.recovery_used", f"uid={user_id}")
    logger.warning("user_logged_in_recovery_code", email=row["email"])
    return await _issue_login_response(pool, row, request, response)


@router.post("/auth/2fa/disable")
@limiter.limit("10/minute")
async def totp_disable(body: TotpCodeRequest, request: Request):
    """Отключить 2FA. Нужен действующий код или recovery-код."""
    user = await _require_admin(request)
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    if not await get_two_factor_enabled(pool):
        raise _two_factor_disabled()
    state = await _totp_state(pool, user["uid"])
    if not state or not state.get("confirmed"):
        raise HTTPException(status_code=400, detail="2FA не настроена")

    # Тот же разбор кода, что и при входе: неверные попытки копятся, после
    # пяти срабатывает блокировка. Раньше диалог отключения можно было
    # перебирать без ограничений (лимит 10/мин не считал неверные коды).
    await _check_totp_or_recovery(pool, user["uid"], state, body.code, invalid_status=400)

    await pool.execute("DELETE FROM admin_totp WHERE user_id = $1", user["uid"])
    await audit_action(request, "auth.2fa.disabled", f"uid={user['uid']}")
    return {"status": "disabled"}
