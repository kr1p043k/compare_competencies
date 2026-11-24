"""Tests for src/cli/retention.py.

Covers the four acceptance criteria:
- day-based arguments accept only 1..36500 and fail fast with a Russian error;
- the destructive arguments never reach the DB executor;
- dry-run is the default and performs no DELETE;
- retention_fk_on_delete migration exists and chains onto the previous head.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    from src.cli import retention

    parser = argparse.ArgumentParser(prog="cli")
    sub = parser.add_subparsers(dest="command", required=True)
    retention.build_parser(sub)
    return parser


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.rowcount = 0

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def fetchone(self):
        # COUNT(*) запросы: для dry-run возвращаем 0 кандидатов, чтобы
        # даже при apply=True не было реальных DELETE.
        return (0,)


class _FakeConnection:
    def __init__(self) -> None:
        self.autocommit = False
        self.cur = _FakeCursor()
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cur

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _patch_psycopg2(monkeypatch) -> _FakeConnection:
    conn = _FakeConnection()
    monkeypatch.setattr("psycopg2.connect", lambda *a, **k: conn)
    return conn


def _deletes(conn: _FakeConnection) -> list[str]:
    return [sql for sql in conn.cur.executed if sql.lstrip().upper().startswith("DELETE")]


# ---------------------------------------------------------------------------
# TASK 1: bounds validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["0", "-1", "-5", "36501", "99999", "abc", "", "1.5"])
def test_positive_days_rejects_invalid(raw: str) -> None:
    from src.cli.retention import positive_days

    with pytest.raises(argparse.ArgumentTypeError):
        positive_days(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", 1), ("36500", 36500), ("90", 90), ("730", 730)],
)
def test_positive_days_accepts_boundaries(raw: str, expected: int) -> None:
    from src.cli.retention import positive_days

    assert positive_days(raw) == expected


@pytest.mark.parametrize("flag", ["--vacancies-days", "--logs-days", "--run-days"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_day_arg_nonzero_exit_no_delete(
    flag: str, value: str, capsys, monkeypatch
) -> None:
    executor = MagicMock()
    monkeypatch.setattr("src.cli.retention.main", executor)

    with pytest.raises(SystemExit) as excinfo:
        _make_parser().parse_args(["retention", flag, value])

    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "от 1 до 36500" in err
    executor.assert_not_called()


# ---------------------------------------------------------------------------
# TASK 3 + dry-run behaviour
# ---------------------------------------------------------------------------

def test_valid_value_dry_run_no_delete(capsys, monkeypatch) -> None:
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(apply=False, vacancies_days=100, logs_days=90, run_days=365)
    rc = retention.main(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "к удалению" in out
    assert "dry-run" in out
    assert _deletes(conn) == []
    assert conn.rolled_back is True
    assert conn.committed is False


def test_default_invocation_is_dry_run(capsys, monkeypatch) -> None:
    args = _make_parser().parse_args(["retention"])
    assert args.apply is False
    assert args.vacancies_days == 730
    assert args.logs_days == 90
    assert args.run_days == 365

    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    assert retention.main(args) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert _deletes(conn) == []
    assert conn.rolled_back is True


def test_vacancies_filter_uses_coalesce(monkeypatch) -> None:
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(apply=False, vacancies_days=100, logs_days=90, run_days=365)
    retention.main(args)

    vac_sql = next(sql for sql in conn.cur.executed if "FROM vacancies" in sql)
    assert "COALESCE(published_at, created_at)" in vac_sql


# ---------------------------------------------------------------------------
# TASK 2: FK migration
# ---------------------------------------------------------------------------

def test_fk_migration_exists_and_chained() -> None:
    f = PROJECT_ROOT / "alembic" / "versions" / "retention_fk_on_delete.py"
    assert f.exists(), "retention_fk_on_delete migration file missing"

    content = f.read_text(encoding="utf-8")
    rev = re.search(r'revision: str = "([^"]+)"', content)
    down = re.search(r'down_revision: str \| None = "([^"]+)"', content)

    assert rev is not None, "revision not declared"
    assert down is not None, "down_revision not declared"
    assert rev.group(1) == "retention_fk_on_delete"
    assert down.group(1) == "add_admin_totp"
    assert "ON DELETE SET NULL" in content


def test_fk_migration_is_single_alembic_head() -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    script = ScriptDirectory.from_config(cfg)

    # Инвариант: ровно одна голова (иначе `alembic upgrade head` падает).
    # Конкретное имя ревизии здесь не проверяем: ветка живёт своей историей.
    assert len(script.get_heads()) == 1, script.get_heads()


# ---------------------------------------------------------------------------
# Сессии и терминальные прогоны
# ---------------------------------------------------------------------------

def _count_sql(conn: _FakeConnection, table: str) -> str:
    """COUNT-запрос по таблице: он выполняется всегда, в отличие от DELETE."""
    return next(sql for sql in conn.cur.executed if f"FROM {table}" in sql)


def test_sessions_cleanup_only_closed(monkeypatch) -> None:
    """Активные сессии (logged_out_at IS NULL) чистить нельзя - разлогиним живых."""
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(
        apply=False, vacancies_days=730, logs_days=90, run_days=365, session_days=30
    )
    retention.main(args)

    sql = _count_sql(conn, "sessions")
    assert "logged_out_at IS NOT NULL" in sql
    assert "30 days" in sql


def test_sessions_respect_custom_days(monkeypatch) -> None:
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(
        apply=False, vacancies_days=730, logs_days=90, run_days=365, session_days=7
    )
    retention.main(args)

    assert "7 days" in _count_sql(conn, "sessions")


def test_pipeline_runs_only_terminal_status(monkeypatch) -> None:
    """Идущий прогон (status='started') не должен попасть под чистку."""
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(
        apply=False, vacancies_days=730, logs_days=90, run_days=365, session_days=30
    )
    retention.main(args)

    sql = _count_sql(conn, "pipeline_runs")
    assert "status IN ('completed', 'failed', 'cancelled')" in sql
    assert "'started'" not in sql


def test_session_days_must_be_positive(monkeypatch) -> None:
    """--session-days=-1 удалил бы историю сессий целиком."""
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(
        apply=True, vacancies_days=730, logs_days=90, run_days=365, session_days=-1
    )
    assert retention.main(args) == 2
    assert conn.cur.executed == []


# ---------------------------------------------------------------------------
# TASK 7.8.2: сверка политики docs/RETENTION.md с фактическим планом CLI
# ---------------------------------------------------------------------------


def test_request_logs_window_uses_created_at_with_configured_days(monkeypatch) -> None:
    """Политика: request_logs чистится по created_at старше --logs-days."""
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(
        apply=False, vacancies_days=730, logs_days=45, run_days=365,
        session_days=30, audit_days=730,
    )
    retention.main(args)

    sql = _count_sql(conn, "request_logs")
    assert "created_at < NOW() - INTERVAL '45 days'" in sql


def test_audit_window_uses_created_at_with_default_days(monkeypatch) -> None:
    """Политика: audit_logs чистится по created_at, по умолчанию 730 дней."""
    conn = _patch_psycopg2(monkeypatch)
    from src.cli import retention

    args = SimpleNamespace(
        apply=False, vacancies_days=730, logs_days=90, run_days=365,
        session_days=30, audit_days=730,
    )
    retention.main(args)

    sql = _count_sql(conn, "audit_logs")
    assert "created_at < NOW() - INTERVAL '730 days'" in sql


def test_parser_defaults_match_policy_table() -> None:
    """Дефолты argparse = окна из таблицы политики RETENTION.md:
    vacancies 730, logs 90, sessions 30, runs 365, audit 730, dry-run."""
    args = _make_parser().parse_args(["retention"])
    assert args.vacancies_days == 730
    assert args.logs_days == 90
    assert args.session_days == 30
    assert args.run_days == 365
    assert args.audit_days == 730
    assert args.apply is False


# --- изоляция пропуска отсутствующей таблицы (SAVEPOINT на шаг) -------------


class UndefinedTableError(Exception):
    """Заменитель psycopg2.errors.UndefinedTable для фейкового драйвера."""


class _TxnCursor:
    """Курсор, эмулирующий psycopg2: COUNT по отсутствующей таблице падает по
    UndefinedTable; SAVEPOINT/RELEASE/DELETE просто записываются."""

    def __init__(self, rows_by_table: dict[str, int], missing: set[str]) -> None:
        self.rows_by_table = rows_by_table
        self.missing = missing
        self.executed: list[str] = []
        self.rowcount = 0
        self._last_table = ""

    def execute(self, sql: str) -> None:
        self.executed.append(sql)
        for table in self.rows_by_table:
            if f"FROM {table} " in sql:
                if table in self.missing:
                    raise UndefinedTableError(f'relation "{table}" does not exist')
                self._last_table = table
                self.rowcount = self.rows_by_table[table]
                return

    def fetchone(self):
        return (self.rows_by_table.get(self._last_table, 0),)


class _TxnConnection:
    def __init__(self, cursor: _TxnCursor) -> None:
        self.cur = cursor
        self.autocommit = False
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self.cur

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _patch_fake_psycopg2(monkeypatch, conn: _TxnConnection) -> None:
    import sys

    fake = SimpleNamespace(
        connect=lambda *a, **k: conn,
        errors=SimpleNamespace(UndefinedTable=UndefinedTableError),
    )
    monkeypatch.setitem(sys.modules, "psycopg2", fake)


def test_missing_table_step_does_not_discard_earlier_deletes(capsys, monkeypatch) -> None:
    """--apply + отсутствующая audit_logs: удаление vacancies должно
    сохраниться (откат только к savepoint шага), а не исчезнуть из-за
    общего conn.rollback()."""
    conn = _TxnConnection(
        _TxnCursor(
            {"vacancies": 5, "request_logs": 0, "sessions": 0,
             "pipeline_runs": 0, "audit_logs": 0},
            missing={"audit_logs"},
        )
    )
    _patch_fake_psycopg2(monkeypatch, conn)
    from src.cli import retention

    args = SimpleNamespace(
        apply=True, vacancies_days=730, logs_days=90, run_days=365,
        session_days=30, audit_days=730,
    )
    assert retention.main(args) == 0

    executed = conn.cur.executed
    assert "DELETE FROM vacancies" in " ".join(executed)
    assert "ROLLBACK TO SAVEPOINT audit_logs" in executed
    # Ни одного отката всей транзакции до финального commit:
    assert conn.rollbacks == 0
    assert conn.commits == 1
    out = capsys.readouterr().out
    assert "audit_logs" in out and "пропущено" in out
    assert "vacancies" in out and "удалено" in out


def test_every_policy_table_has_a_step_and_savepoint(monkeypatch) -> None:
    """Все 5 таблиц политики присутствуют в плане и каждый шаг изолирован
    собственным SAVEPOINT (пропуск одного не роняет остальные)."""
    conn = _TxnConnection(
        _TxnCursor(
            {"vacancies": 0, "request_logs": 0, "sessions": 0,
             "pipeline_runs": 0, "audit_logs": 0},
            missing=set(),
        )
    )
    _patch_fake_psycopg2(monkeypatch, conn)
    from src.cli import retention

    args = SimpleNamespace(
        apply=False, vacancies_days=730, logs_days=90, run_days=365,
        session_days=30, audit_days=730,
    )
    assert retention.main(args) == 0

    executed = conn.cur.executed
    for table in ("vacancies", "request_logs", "sessions", "pipeline_runs", "audit_logs"):
        assert f"SAVEPOINT {table}" in executed
        assert f"RELEASE SAVEPOINT {table}" in executed
        assert any(f"FROM {table} " in sql for sql in executed)
