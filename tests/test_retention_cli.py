"""Retention CLI: защита от опечаток и корректный dry-run.

Удаление данных необратимо, поэтому проверяем именно барьеры:
  * по умолчанию dry-run (rollback, ничего не удалено);
  * отрицательный/нулевой срок отклоняется с кодом 2;
  * SQL формируется из int-значений, а не из пользовательской строки.
"""
from __future__ import annotations

import argparse

import pytest

from src.cli import retention


def _args(**kw) -> argparse.Namespace:
    base = {
        "apply": False,
        "vacancies_days": 730,
        "logs_days": 90,
        "run_days": 365,
    }
    base.update(kw)
    return argparse.Namespace(**base)


class _FakeCursor:
    def __init__(self):
        self.executed: list[str] = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self.rowcount = 0

    def fetchone(self):
        return (0,)


class _FakeConn:
    def __init__(self):
        self.autocommit = True
        self.cur = _FakeCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


@pytest.fixture
def fake_db(monkeypatch):
    import psycopg2

    conn = _FakeConn()
    monkeypatch.setattr(psycopg2, "connect", lambda *_a, **_kw: conn)
    return conn


class TestNegativeGuard:
    @pytest.mark.parametrize("field", ["vacancies_days", "logs_days", "run_days"])
    def test_non_positive_days_rejected(self, fake_db, field):
        """--logs-days=-1 удалил бы почти всю таблицу: отклоняем."""
        code = retention.main(_args(**{field: -1}))
        assert code == 2
        assert fake_db.cur.executed == [], "до БД дело доходить не должно"

    def test_zero_days_rejected(self, fake_db):
        assert retention.main(_args(logs_days=0)) == 2


class TestDryRun:
    def test_default_is_dry_run(self, fake_db):
        retention.main(_args())
        assert fake_db.rolled_back is True
        assert fake_db.committed is False, "без --apply коммита быть не должно"

    def test_no_delete_statements_without_apply(self, fake_db):
        retention.main(_args())
        deletes = [q for q in fake_db.cur.executed if q.strip().upper().startswith("DELETE")]
        assert deletes == []


class TestApply:
    def test_apply_commits(self, fake_db):
        fake_db.cur.rowcount = 0
        retention.main(_args(apply=True))
        assert fake_db.committed is True

    def test_days_are_interpolated_as_numbers(self, fake_db):
        retention.main(_args(apply=True, logs_days=90))
        count_sql = [q for q in fake_db.cur.executed if "COUNT(*)" in q and "request_logs" in q]
        assert count_sql, "должен быть подсчёт кандидатов по request_logs"
        assert "90 days" in count_sql[0]
        assert "logs_days" not in count_sql[0], "имя переменной не должно попасть в SQL"


class TestDsn:
    def test_asyncpg_dsn_is_converted(self):
        assert retention._to_sync_dsn("postgresql+asyncpg://u:p@h/db") == "postgresql://u:p@h/db"

    def test_plain_dsn_untouched(self):
        assert retention._to_sync_dsn("postgresql://u:p@h/db") == "postgresql://u:p@h/db"
