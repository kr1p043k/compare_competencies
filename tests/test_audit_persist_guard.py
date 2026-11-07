"""Барьеры durable-аудита: самовосстановление таблицы audit_logs.

Миграцию add_audit_logs на части баз не применяли (цепочку переписывали), поэтому
запись аудита должна сама создать таблицу, а не терять события навсегда.
"""

from __future__ import annotations

from types import SimpleNamespace

import src.api_pkg.request_logger as rl
import src.db as dbmod


class _UndefinedTableError(Exception):
    pgcode = "42P01"


class _FakePool:
    def __init__(self, *, ddl_ok: bool = True, regclass: bool = True):
        self.ddl: list[str] = []
        self.insert_attempts = 0
        self.missing = True
        self.ddl_ok = ddl_ok
        self.regclass = regclass

    async def execute(self, sql: str, *args):
        head = sql.lstrip().upper()
        if head.startswith("INSERT INTO AUDIT_LOGS"):
            self.insert_attempts += 1
            if self.missing:
                raise _UndefinedTableError("relation \"audit_logs\" does not exist")
            return
        self.ddl.append(sql)
        if not self.ddl_ok:
            raise RuntimeError("permission denied for schema public")
        if head.startswith("CREATE TABLE"):
            self.missing = False

    async def fetchval(self, _sql: str, *args):
        return self.regclass


def _fake_request() -> SimpleNamespace:
    return SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(request_id="rid-1"),
    )


def _patch_pool(monkeypatch, pool: _FakePool) -> None:
    monkeypatch.setattr(dbmod, "get_pool", lambda: pool)
    monkeypatch.setattr(rl, "_audit_table_missing", False)


async def test_audit_creates_missing_table_and_retries(monkeypatch):
    pool = _FakePool()
    _patch_pool(monkeypatch, pool)

    await rl._persist_audit_log(_fake_request(), "admin.test", "uid=abc", "a@b.c", "admin")

    assert pool.insert_attempts == 2, "INSERT должен повториться после создания таблицы"
    assert any("CREATE TABLE IF NOT EXISTS audit_logs" in d for d in pool.ddl)
    assert any("idx_audit_created" in d for d in pool.ddl)
    assert rl._audit_table_missing is False


async def test_audit_gives_up_when_ddl_denied(monkeypatch):
    pool = _FakePool(ddl_ok=False, regclass=False)
    _patch_pool(monkeypatch, pool)

    await rl._persist_audit_log(_fake_request(), "admin.test", "", "a@b.c", "admin")
    assert rl._audit_table_missing is True

    ddl_before = len(pool.ddl)
    attempts_before = pool.insert_attempts
    # Второй вызов не должен долбить БД: флаг уже поднят.
    await rl._persist_audit_log(_fake_request(), "admin.test", "", "a@b.c", "admin")
    assert len(pool.ddl) == ddl_before
    assert pool.insert_attempts == attempts_before


async def test_audit_does_not_mark_missing_on_other_errors(monkeypatch):
    """Прочие ошибки (не 'нет таблицы') не должны блокировать аудит навсегда."""

    class _Pool(_FakePool):
        async def execute(self, sql: str, *args):
            if sql.lstrip().upper().startswith("INSERT INTO AUDIT_LOGS"):
                self.insert_attempts += 1
                raise RuntimeError("connection reset by peer")
            return await super().execute(sql, *args)

    pool = _Pool()
    _patch_pool(monkeypatch, pool)

    await rl._persist_audit_log(_fake_request(), "admin.test", "", "a@b.c", "admin")
    assert rl._audit_table_missing is False


def test_split_target_extracts_user_and_id():
    assert rl._split_target("uid=11111111-1111-1111-1111-111111111111 role=admin") == (
        "user",
        "11111111-1111-1111-1111-111111111111",
    )
    assert rl._split_target("email=a@b.c role=admin") == ("user", "a@b.c")
    assert rl._split_target("id=abc") == (None, "abc")
    assert rl._split_target("") == (None, None)


def test_is_missing_audit_table_detects_nested_orig():
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = _UndefinedTableError("42P01")
    assert rl._is_missing_audit_table(wrapper) is True
    assert rl._is_missing_audit_table(ValueError("nope")) is False
