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

    assert script.get_heads() == ["retention_fk_on_delete"]
    assert script.get_revision("retention_fk_on_delete").down_revision == "add_admin_totp"
