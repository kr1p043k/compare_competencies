"""Unit tests for admin hardening: step-up auth, TOTP lockout and retention.

Tests do not require a live PostgreSQL instance; any DB-facing helper is
replaced by a fake pool/session/monkeypatch.
"""

from __future__ import annotations

import argparse
import os
import socket
import urllib.parse
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import psycopg2
import pytest
from fastapi import HTTPException

import src.api_pkg.routers.auth as auth_mod
from src.api_pkg.request_logger import _is_missing_audit_table, _split_target
from src.api_pkg.routers.admin import _assert_caller_password

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)


def _db_reachable() -> bool:
    try:
        parts = urllib.parse.urlsplit(TEST_DB_URL)
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=2):
            return True
    except OSError:
        return False


# ------------------------------------------------------------------ _split_target


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("uid=00000000-0000-0000-0000-000000000001 role=teacher", ("user", "00000000-0000-0000-0000-000000000001")),
        ("email=a@b.c role=admin", ("user", "a@b.c")),
        ("id=abc", (None, "abc")),
        ("", (None, None)),
    ],
    ids=["uid_uuid", "email", "id", "empty"],
)
def test_split_target(target: str, expected: tuple[str | None, str | None]):
    assert _split_target(target) == expected


# ------------------------------------------------------------------ _is_missing_audit_table


class UndefinedTableError(Exception):
    """Stand-in for asyncpg UndefinedTableError."""


def test_is_missing_audit_table_pgcode_42p01():
    exc = Exception("relation does not exist")
    exc.pgcode = "42P01"
    assert _is_missing_audit_table(exc) is True


def test_is_missing_audit_table_with_orig_undefined_table():
    outer = Exception("outer")
    outer.orig = UndefinedTableError("missing")
    assert _is_missing_audit_table(outer) is True


def test_is_missing_audit_table_plain_valueerror():
    assert _is_missing_audit_table(ValueError("plain")) is False


# ------------------------------------------------------------------ _check_totp_or_recovery


class _FakePool:
    def __init__(self):
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))

    async def fetchrow(self, *args: Any, **kwargs: Any) -> None:
        return None


async def test_check_totp_locked_state_raises_429():
    pool = _FakePool()
    state = {
        "secret": "enc",
        "confirmed": True,
        "recovery_codes": None,
        "failed_attempts": 0,
        "locked_until": datetime.now(UTC),
        "retry_in": 300,
    }
    with pytest.raises(HTTPException) as exc_info:
        await auth_mod._check_totp_or_recovery(pool, uuid.uuid4(), state, "000000")
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers.get("Retry-After") == "300"
    assert pool.executed == []


async def test_check_totp_wrong_code_locks_at_fifth_failure(monkeypatch):
    monkeypatch.setattr(auth_mod, "_decrypt_totp_secret", lambda _stored: "secret")
    monkeypatch.setattr(auth_mod, "_verify_totp", lambda _secret, _code: False)
    pool = _FakePool()
    state = {
        "secret": "enc",
        "confirmed": True,
        "recovery_codes": [],
        "failed_attempts": 4,
        "locked_until": None,
        "retry_in": 0,
    }
    with pytest.raises(HTTPException) as exc_info:
        await auth_mod._check_totp_or_recovery(pool, uuid.uuid4(), state, "000000")
    assert exc_info.value.status_code == 429
    assert len(pool.executed) == 1
    query, args = pool.executed[0]
    assert "locked_until" in query
    assert args[1] == 5


async def test_check_totp_wrong_code_increments_failures_no_lock(monkeypatch):
    monkeypatch.setattr(auth_mod, "_decrypt_totp_secret", lambda _stored: "secret")
    monkeypatch.setattr(auth_mod, "_verify_totp", lambda _secret, _code: False)
    pool = _FakePool()
    state = {
        "secret": "enc",
        "confirmed": True,
        "recovery_codes": [],
        "failed_attempts": 0,
        "locked_until": None,
        "retry_in": 0,
    }
    with pytest.raises(HTTPException) as exc_info:
        await auth_mod._check_totp_or_recovery(pool, uuid.uuid4(), state, "000000")
    assert exc_info.value.status_code == 400
    assert len(pool.executed) == 1
    query, args = pool.executed[0]
    assert "locked_until" in query
    assert args[1] == 1
    assert args[2] is False


async def test_check_totp_valid_code_resets_counters(monkeypatch):
    monkeypatch.setattr(auth_mod, "_decrypt_totp_secret", lambda _stored: "secret")
    monkeypatch.setattr(auth_mod, "_verify_totp", lambda _secret, _code: True)
    pool = _FakePool()
    state = {
        "secret": "enc",
        "confirmed": True,
        "recovery_codes": [],
        "failed_attempts": 3,
        "locked_until": None,
        "retry_in": 0,
    }
    await auth_mod._check_totp_or_recovery(pool, uuid.uuid4(), state, "123456")
    assert len(pool.executed) == 1
    query, _args = pool.executed[0]
    assert "failed_attempts = 0" in query
    assert "locked_until = NULL" in query


async def test_check_totp_valid_recovery_code_consumes_it(monkeypatch):
    monkeypatch.setattr(auth_mod, "_decrypt_totp_secret", lambda _stored: "secret")
    monkeypatch.setattr(auth_mod, "_verify_totp", lambda _secret, _code: False)
    pool = _FakePool()
    code = "abcd1234"
    hashed = auth_mod._hash_recovery_code(code)
    state = {
        "secret": "enc",
        "confirmed": True,
        "recovery_codes": [hashed],
        "failed_attempts": 0,
        "locked_until": None,
        "retry_in": 0,
    }
    await auth_mod._check_totp_or_recovery(pool, uuid.uuid4(), state, code)
    assert len(pool.executed) == 1
    query, args = pool.executed[0]
    assert "recovery_codes" in query
    assert auth_mod.json.loads(args[1]) == []


# ------------------------------------------------------------------ _assert_caller_password


class _FakeSession:
    def __init__(self, result: bool):
        self._result = result
        self.queries: list[tuple[str, dict | None]] = []

    async def execute(self, query: Any, params: dict | None = None):
        self.queries.append((str(query), params))
        result = MagicMock()
        result.scalar_one.return_value = self._result
        return result


async def test_assert_caller_password_false_raises_403():
    session = _FakeSession(False)
    caller = {"uid": str(uuid.uuid4())}
    with pytest.raises(HTTPException) as exc_info:
        await _assert_caller_password(session, caller, "wrongpass")
    assert exc_info.value.status_code == 403
    assert "Неверный пароль администратора" in exc_info.value.detail


async def test_assert_caller_password_true_passes():
    session = _FakeSession(True)
    caller = {"uid": str(uuid.uuid4())}
    await _assert_caller_password(session, caller, "goodpass")
    assert len(session.queries) == 1
    sql, params = session.queries[0]
    assert "crypt(:pw, password_hash) = password_hash" in sql
    assert params is not None
    assert params["uid"] == caller["uid"]
    assert params["pw"] == "goodpass"


async def test_assert_caller_password_none_raises_403_without_query():
    session = _FakeSession(True)
    caller = {"uid": str(uuid.uuid4())}
    with pytest.raises(HTTPException) as exc_info:
        await _assert_caller_password(session, caller, None)
    assert exc_info.value.status_code == 403
    assert "подтвердите свой пароль" in exc_info.value.detail
    assert session.queries == []


async def test_assert_caller_password_no_uid_raises_401():
    session = _FakeSession(True)
    with pytest.raises(HTTPException) as exc_info:
        await _assert_caller_password(session, {"u": "x"}, "anypass")
    assert exc_info.value.status_code == 401
    assert session.queries == []


# ------------------------------------------------------------------ retention


def test_retention_dry_run_plan_includes_audit_logs(monkeypatch):
    class _FakeCursor:
        def __init__(self):
            self.sqls: list[str] = []

        def execute(self, sql: str, params=None):
            self.sqls.append(sql)

        def fetchone(self):
            return (5,)

        @property
        def rowcount(self):
            return 5

    class _FakeConnection:
        def __init__(self):
            self._cursor = _FakeCursor()

        def cursor(self):
            return self._cursor

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    fake = _FakeConnection()
    monkeypatch.setattr(psycopg2, "connect", lambda _dsn: fake)

    from src.cli import retention as ret_mod

    args = argparse.Namespace(
        vacancies_days=730,
        logs_days=90,
        run_days=365,
        audit_days=730,
        apply=False,
    )
    assert ret_mod.main(args) == 0
    sqls = fake._cursor.sqls
    assert any("audit_logs" in s and "COUNT(*)" in s for s in sqls)
    assert any("audit_logs" in s and "INTERVAL '730 days'" in s for s in sqls)
    assert not any("DELETE FROM audit_logs" in s for s in sqls)


def test_retention_undefined_table_returns_zero(monkeypatch):
    class _FakeCursor:
        def __init__(self):
            self.sqls: list[str] = []
            self._calls = 0

        def execute(self, sql: str, params=None):
            self.sqls.append(sql)
            self._calls += 1
            if self._calls == 1:
                raise psycopg2.errors.UndefinedTable("relation does not exist")

        def fetchone(self):
            return (0,)

    class _FakeConnection:
        def __init__(self):
            self._cursor = _FakeCursor()

        def cursor(self):
            return self._cursor

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    fake = _FakeConnection()
    monkeypatch.setattr(psycopg2, "connect", lambda _dsn: fake)

    from src.cli import retention as ret_mod

    args = argparse.Namespace(
        vacancies_days=730,
        logs_days=90,
        run_days=365,
        audit_days=730,
        apply=False,
    )
    assert ret_mod.main(args) == 0
