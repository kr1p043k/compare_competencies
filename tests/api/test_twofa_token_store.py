"""Модульные тесты twofa_token_store без живой БД.

Фейковый пул моделирует ровно те SQL-запросы, которые шлёт модуль (INSERT,
DELETE по протуханию, DELETE ... RETURNING для consume, SELECT для peek), чтобы
проверить семантику одноразовых токенов: single-use, протухание, peek без
расхода, mismatch по purpose, TTL и хранение только sha256. Сам фейковый пул
здесь не тестируется: он лишь воспроизводит контракт asyncpg, который нужен
модулю.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from src.api_pkg import twofa_token_store as store

USER_ID = "11111111-1111-1111-1111-111111111111"


class FakePool:
    """Маленький честный фейк asyncpg.Pool для таблицы totp_tokens."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.now: float = now

    def advance(self, seconds: float) -> None:
        """Сдвинуть внутренние часы вперёд (для имитации протухания)."""
        self.now += seconds

    async def execute(self, sql: str, *args: Any) -> str:
        s = sql.strip()
        if "CREATE TABLE" in s or "CREATE INDEX" in s:
            return "OK"
        if s.startswith("DELETE FROM totp_tokens WHERE expires_at < now()"):
            self._cleanup()
            return "DELETE 0"
        if s.startswith("INSERT INTO totp_tokens"):
            token_hash, user_id, purpose, ttl = args
            self.rows[token_hash] = {
                "user_id": user_id,
                "purpose": purpose,
                "issued_at": self.now,
                "expires_at": self.now + ttl.total_seconds(),
            }
            return "INSERT 0 1"
        raise AssertionError(f"unexpected execute: {s!r}")

    async def fetchval(self, sql: str, *args: Any) -> Any:
        s = sql.strip()
        if "to_regclass" in s:
            return True
        if s.startswith("DELETE FROM totp_tokens"):
            token_hash, purpose = args
            row = self.rows.get(token_hash)
            if row is None or row["purpose"] != purpose:
                return None
            if row["expires_at"] <= self.now:
                return None
            del self.rows[token_hash]
            return row["user_id"]
        if s.startswith("SELECT user_id FROM totp_tokens"):
            token_hash, purpose = args
            row = self.rows.get(token_hash)
            if row is None or row["purpose"] != purpose:
                return None
            if row["expires_at"] <= self.now:
                return None
            return row["user_id"]
        raise AssertionError(f"unexpected fetchval: {s!r}")

    def _cleanup(self) -> None:
        for key in [k for k, r in self.rows.items() if r["expires_at"] < self.now]:
            del self.rows[key]


@pytest.mark.asyncio
async def test_issue_then_consume_is_single_use():
    pool = FakePool()
    raw = await store.issue_challenge(pool, USER_ID)

    assert await store.consume_challenge(pool, raw) == USER_ID
    assert await store.consume_challenge(pool, raw) is None


@pytest.mark.asyncio
async def test_expired_token_is_not_consumable():
    pool = FakePool()
    raw = await store.issue_challenge(pool, USER_ID)

    pool.advance(store.CHALLENGE_TTL_SECONDS + 1)
    assert await store.consume_challenge(pool, raw) is None


@pytest.mark.asyncio
async def test_peek_does_not_consume():
    pool = FakePool()
    raw = await store.issue_enroll(pool, USER_ID)

    assert await store.peek_enroll(pool, raw) == USER_ID
    assert await store.consume_enroll(pool, raw) == USER_ID


@pytest.mark.asyncio
async def test_purpose_mismatch_returns_none():
    pool = FakePool()
    raw = await store.issue_enroll(pool, USER_ID)

    assert await store.consume_challenge(pool, raw) is None
    # Токен цел: его всё ещё можно забрать по правильному purpose.
    assert await store.consume_enroll(pool, raw) == USER_ID


@pytest.mark.asyncio
async def test_issue_applies_correct_ttl():
    pool = FakePool()
    ch_raw = await store.issue_challenge(pool, USER_ID)
    en_raw = await store.issue_enroll(pool, USER_ID)

    ch = pool.rows[store._hash(ch_raw)]
    en = pool.rows[store._hash(en_raw)]
    assert ch["expires_at"] - ch["issued_at"] == store.CHALLENGE_TTL_SECONDS
    assert en["expires_at"] - en["issued_at"] == store.ENROLL_TTL_SECONDS


@pytest.mark.asyncio
async def test_raw_token_is_not_stored():
    pool = FakePool()
    raw = await store.issue_enroll(pool, USER_ID)

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert digest == store._hash(raw)
    assert digest in pool.rows
    assert raw not in pool.rows


@pytest.mark.asyncio
async def test_unknown_purpose_raises_value_error():
    pool = FakePool()
    with pytest.raises(ValueError):
        await store.issue_token(pool, USER_ID, "bogus", 60)
