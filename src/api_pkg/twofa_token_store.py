"""Одноразовые 2FA-токены: challenge при входе и пропуск на настройку.

В БД лежит только SHA-256 сырого токена, а сам сырой токен отдаётся клиенту
ровно один раз при выдаче. Таблица totp_tokens создаётся миграцией
(totp_tokens_store), а ``_ensure_table`` - страховка на случай деплоя без
миграции. Модуль не импортирует FastAPI и не трогает БД на импорте: соединение
приходит только аргументом ``pool``, поэтому его можно импортировать без живой
базы.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import secrets

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

PURPOSE_CHALLENGE = "login_challenge"
PURPOSE_ENROLL = "enroll"
CHALLENGE_TTL_SECONDS = 300
ENROLL_TTL_SECONDS = 600

_ALLOWED_PURPOSES = frozenset({PURPOSE_CHALLENGE, PURPOSE_ENROLL})

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS totp_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    purpose    TEXT NOT NULL CHECK (purpose IN ('login_challenge', 'enroll')),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_totp_tokens_expires ON totp_tokens (expires_at)
"""

_table_ready = False
#: Замок на создание таблицы: без него два первых параллельных запроса
#: выполняют CREATE TABLE IF NOT EXISTS одновременно и один из них падает.
_ensure_lock: asyncio.Lock | None = None


def _hash(raw: str) -> str:
    """sha256 строки токена в hex: в БД не кладём сырой токен."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _ensure_table(pool: asyncpg.Pool) -> None:
    """Страховка на случай деплоя без миграции (totp_tokens_store.py)."""
    global _table_ready, _ensure_lock

    if _table_ready:
        return
    if _ensure_lock is None:
        _ensure_lock = asyncio.Lock()
    async with _ensure_lock:
        if _table_ready:
            return
        try:
            await pool.execute(_TABLE_DDL)
            await pool.execute(_INDEX_DDL)
        except Exception:
            # Проигравший гонку получил ошибку каталога: таблица уже есть -
            # проверяем фактическое наличие, иначе ошибку не глотаем.
            exists = await pool.fetchval(
                "SELECT to_regclass('public.totp_tokens') IS NOT NULL"
            )
            if not exists:
                raise
        _table_ready = True


async def issue_token(
    pool: asyncpg.Pool, user_id: str, purpose: str, ttl_seconds: int
) -> str:
    """Выдать новый одноразовый токен: в БД только sha256, клиенту - сырой."""
    if purpose not in _ALLOWED_PURPOSES:
        raise ValueError(f"unknown totp token purpose: {purpose!r}")
    await _ensure_table(pool)
    # Подметаем протухшие токены; сбой чистки не фатален.
    try:
        await pool.execute("DELETE FROM totp_tokens WHERE expires_at < now()")
    except Exception as exc:
        logger.warning("totp_token_cleanup_failed", error=str(exc))
    token = secrets.token_urlsafe(32)
    await pool.execute(
        """
        INSERT INTO totp_tokens (token_hash, user_id, purpose, expires_at)
        VALUES ($1, $2, $3, now() + $4)
        """,
        _hash(token),
        user_id,
        purpose,
        datetime.timedelta(seconds=ttl_seconds),
    )
    return token


async def consume_token(pool: asyncpg.Pool, raw: str, purpose: str) -> str | None:
    """Атомарно забрать токен: DELETE ... RETURNING делает его одноразовым.

    Ошибка БД не глотается: пусть всплывает к вызывающему коду (fail closed).
    """
    await _ensure_table(pool)
    user_id = await pool.fetchval(
        "DELETE FROM totp_tokens"
        " WHERE token_hash = $1 AND purpose = $2 AND expires_at > now()"
        " RETURNING user_id",
        _hash(raw),
        purpose,
    )
    if user_id is None:
        return None
    return str(user_id)


async def peek_token(pool: asyncpg.Pool, raw: str, purpose: str) -> str | None:
    """Посмотреть токен, не забирая его (не расходует одноразовость)."""
    await _ensure_table(pool)
    user_id = await pool.fetchval(
        "SELECT user_id FROM totp_tokens"
        " WHERE token_hash = $1 AND purpose = $2 AND expires_at > now()",
        _hash(raw),
        purpose,
    )
    if user_id is None:
        return None
    return str(user_id)


async def issue_challenge(pool: asyncpg.Pool, user_id: str) -> str:
    """Выдать challenge для входа (TTL 300 c)."""
    return await issue_token(pool, user_id, PURPOSE_CHALLENGE, CHALLENGE_TTL_SECONDS)


async def consume_challenge(pool: asyncpg.Pool, raw: str) -> str | None:
    """Забрать challenge входа, если он ещё жив."""
    return await consume_token(pool, raw, PURPOSE_CHALLENGE)


async def issue_enroll(pool: asyncpg.Pool, user_id: str) -> str:
    """Выдать пропуск на настройку 2FA (TTL 600 c)."""
    return await issue_token(pool, user_id, PURPOSE_ENROLL, ENROLL_TTL_SECONDS)


async def peek_enroll(pool: asyncpg.Pool, raw: str) -> str | None:
    """Посмотреть пропуск на настройку, не расходуя его."""
    return await peek_token(pool, raw, PURPOSE_ENROLL)


async def consume_enroll(pool: asyncpg.Pool, raw: str) -> str | None:
    """Забрать пропуск на настройку, если он ещё жив."""
    return await consume_token(pool, raw, PURPOSE_ENROLL)
