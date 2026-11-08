"""Шаг audit_logs в CLI retention: план, dry-run, отсутствие таблицы.

БД не нужна: psycopg2 подменяется фейком через sys.modules (первый импорт
идёт внутри main), соединение/курсор записывают выполненные запросы.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest

import src.cli.retention as retention


class _UndefinedTableError(Exception):
    pass


class _Errors:
    # psycopg2.errors.UndefinedTable (сравнивается по классу в retention.main).
    UndefinedTable = _UndefinedTableError


class _Cursor:
    def __init__(self, rows_by_table: dict[str, int], missing: set[str]):
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
                    raise _UndefinedTableError(f'relation "{table}" does not exist')
                self._last_table = table
                self.rowcount = self.rows_by_table[table]
                return

    def fetchone(self):
        return (self.rows_by_table.get(self._last_table, 0),)


class _Conn:
    def __init__(self, cursor: _Cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.autocommit = True

    def cursor(self) -> _Cursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _install_fake_psycopg2(monkeypatch, conn: _Conn) -> None:
    fake = SimpleNamespace(connect=lambda _dsn: conn, errors=_Errors)
    monkeypatch.setitem(sys.modules, "psycopg2", fake)


def _args(**overrides) -> SimpleNamespace:
    base = {
        "apply": False,
        "vacancies_days": 730,
        "logs_days": 90,
        "run_days": 365,
        "audit_days": 730,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dry_run_reports_audit_and_never_deletes(monkeypatch, capsys):
    cursor = _Cursor(
        {"vacancies": 3, "request_logs": 5, "pipeline_runs": 2, "audit_logs": 7}, missing=set()
    )
    conn = _Conn(cursor)
    _install_fake_psycopg2(monkeypatch, conn)

    assert retention.main(_args()) == 0

    executed = "\n".join(cursor.executed)
    assert "FROM audit_logs " in executed
    assert "730 days" in executed
    assert "DELETE" not in executed, "dry-run не должен удалять"
    assert conn.commits == 0
    assert conn.rollbacks == 1
    out = capsys.readouterr().out
    assert "audit_logs" in out and "к удалению: 7" in out


def test_apply_deletes_audit_rows(monkeypatch):
    cursor = _Cursor(
        {"vacancies": 0, "request_logs": 0, "pipeline_runs": 0, "audit_logs": 4}, missing=set()
    )
    conn = _Conn(cursor)
    _install_fake_psycopg2(monkeypatch, conn)

    assert retention.main(_args(apply=True)) == 0

    executed = "\n".join(cursor.executed)
    assert "DELETE FROM audit_logs " in executed
    assert conn.commits == 1


def test_missing_audit_table_is_skipped_not_fatal(monkeypatch, capsys):
    cursor = _Cursor(
        {"vacancies": 1, "request_logs": 2, "pipeline_runs": 0, "audit_logs": 0},
        missing={"audit_logs"},
    )
    conn = _Conn(cursor)
    _install_fake_psycopg2(monkeypatch, conn)

    assert retention.main(_args()) == 0

    out = capsys.readouterr().out
    assert "audit_logs" in out and "пропущено" in out
    # Остальные шаги всё равно посчитаны.
    assert "vacancies" in out and "request_logs" in out
    assert conn.rollbacks >= 1


def test_invalid_days_return_2_without_connecting(monkeypatch):
    def _must_not_connect(_dsn):  # pragma: no cover - не должен вызваться
        raise AssertionError("connect не должен вызываться при неверных аргументах")

    fake = SimpleNamespace(connect=_must_not_connect, errors=_Errors)
    monkeypatch.setitem(sys.modules, "psycopg2", fake)

    assert retention.main(_args(audit_days=0)) == 2


def test_default_audit_days_applied_when_attribute_missing(monkeypatch):
    cursor = _Cursor(
        {"vacancies": 0, "request_logs": 0, "pipeline_runs": 0, "audit_logs": 1}, missing=set()
    )
    conn = _Conn(cursor)
    _install_fake_psycopg2(monkeypatch, conn)

    args = _args()
    del args.audit_days  # старая версия CLI без --audit-days

    assert retention.main(args) == 0
    executed = "\n".join(cursor.executed)
    assert f"{retention.DEFAULT_AUDIT_DAYS} days" in executed


@pytest.mark.parametrize("value", [0, -1])
def test_positive_days_type_rejects_non_positive(value):
    with pytest.raises(Exception):
        retention.positive_days(str(value))


def _cli(argv: list[str]) -> argparse.Namespace:
    """Разбор аргументов тем же каркасом, что и src/cli/__main__.py."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    retention.build_parser(sub)
    return parser.parse_args(argv)


def test_cli_parser_wires_audit_days_flag():
    """Флаг --audit-days 730 доезжает до main, остальные дни - своими дефолтами."""
    args = _cli(["retention", "--audit-days", "730"])
    assert args.audit_days == 730
    assert args.vacancies_days == retention.DEFAULT_VACANCY_DAYS
    assert args.logs_days == retention.DEFAULT_LOG_DAYS
    assert args.run_days == retention.DEFAULT_RUN_DAYS
    assert args.apply is False


def test_cli_parser_defaults_when_flag_omitted():
    """Без --audit-days используется DEFAULT_AUDIT_DAYS, dry-run по умолчанию."""
    args = _cli(["retention"])
    assert args.audit_days == retention.DEFAULT_AUDIT_DAYS
    assert args.apply is False


def test_cli_parser_rejects_days_out_of_range(capsys):
    """"--audit-days 0" отклоняется ещё argparse-ом (code 2), не доходит до SQL."""
    with pytest.raises(SystemExit) as exc:
        _cli(["retention", "--audit-days", "0"])
    assert exc.value.code == 2
    assert "вне допустимого диапазона" in capsys.readouterr().err


def test_cli_dispatch_dry_run_shows_audit_step(monkeypatch, capsys):
    """Полный путь "python -m src.cli retention --audit-days 730" (диспетчер
    __main__): dry-run показывает шаг audit_logs и не делает DELETE.
    psycopg2 подменён, боевая БД не затрагивается.
    """
    cursor = _Cursor(
        {"vacancies": 0, "request_logs": 0, "pipeline_runs": 0, "audit_logs": 9}, missing=set()
    )
    conn = _Conn(cursor)
    _install_fake_psycopg2(monkeypatch, conn)

    import src.cli.__main__ as cli_main

    monkeypatch.setattr(sys, "argv", ["src.cli", "retention", "--audit-days", "730"])
    cli_main.main()

    executed = "\n".join(cursor.executed)
    assert "FROM audit_logs " in executed
    assert "730 days" in executed
    assert "DELETE" not in executed, "dry-run не должен удалять"
    assert conn.commits == 0 and conn.rollbacks == 1
    out = capsys.readouterr().out
    assert "audit_logs" in out and "к удалению: 9" in out
