"""Retention CLI: защита от опечаток и корректный dry-run.

Удаление данных необратимо, поэтому проверяем именно барьеры:
  * по умолчанию dry-run (rollback, ничего не удалено);
  * отрицательный/нулевой срок отклоняется с кодом 2;
  * SQL формируется из int-значений, а не из пользовательской строки.
"""
from __future__ import annotations

import argparse
import re

import pytest

from src.cli import retention


def _args(**kw) -> argparse.Namespace:
    base = {
        "apply": False,
        "vacancies_days": 730,
        "logs_days": 90,
        "session_days": 30,
        "run_days": 365,
    }
    base.update(kw)
    return argparse.Namespace(**base)


class _FakeCursor:
    """Записывает SQL и отдаёт счётчики кандидатов по таблицам.

    counts позволяет задать ненулевой результат COUNT(*) для конкретной таблицы,
    чтобы проверить, что dry-run не удаляет, а --apply удаляет на самом деле.
    """

    def __init__(self, counts: dict[str, int] | None = None):
        self.executed: list[str] = []
        self.rowcount = 0
        self.counts = counts if counts is not None else {}

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self.rowcount = 0

    def fetchone(self):
        sql = self.executed[-1] if self.executed else ""
        match = re.search(r"COUNT\(\*\)\s+FROM\s+(\w+)", sql, re.IGNORECASE)
        if match:
            return (self.counts.get(match.group(1), 0),)
        return (0,)


class _FakeConn:
    def __init__(self, counts: dict[str, int] | None = None):
        self.autocommit = True
        self.cur = _FakeCursor(counts)
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
def fake_db_factory(monkeypatch):
    import psycopg2

    def _make(counts=None):
        conn = _FakeConn(counts)
        monkeypatch.setattr(psycopg2, "connect", lambda *_a, **_kw: conn)
        return conn

    return _make


@pytest.fixture
def fake_db(fake_db_factory):
    return fake_db_factory()


class TestNegativeGuard:
    @pytest.mark.parametrize("field", ["vacancies_days", "logs_days", "session_days", "run_days"])
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


class TestSessions:
    def test_dry_run_does_not_delete_sessions(self, fake_db_factory):
        conn = fake_db_factory(counts={"sessions": 7})
        retention.main(_args())
        deletes = [q for q in conn.cur.executed if q.strip().upper().startswith("DELETE") and "sessions" in q]
        assert deletes == [], "dry-run не должен запускать DELETE для sessions"

    def test_apply_deletes_sessions(self, fake_db_factory):
        conn = fake_db_factory(counts={"sessions": 7})
        retention.main(_args(apply=True))
        deletes = [q for q in conn.cur.executed if q.strip().upper().startswith("DELETE") and "sessions" in q]
        assert deletes, "--apply должен удалять закрытые сессии"
        assert "logged_out_at IS NOT NULL" in deletes[0]

    def test_only_logged_out_sessions_are_counted(self, fake_db_factory):
        conn = fake_db_factory(counts={"sessions": 3})
        retention.main(_args())
        count_sql = [q for q in conn.cur.executed if "COUNT(*)" in q and "sessions" in q]
        assert count_sql
        # активные сессии (logged_out_at IS NULL) не попадают в кандидатов
        assert "logged_out_at IS NOT NULL" in count_sql[0]


class TestPipelineRunsStatus:
    def test_running_run_not_deleted_even_with_run_days_1(self, fake_db_factory):
        conn = fake_db_factory(counts={"pipeline_runs": 5})
        retention.main(_args(apply=True, run_days=1))
        deletes = [q for q in conn.cur.executed if q.strip().upper().startswith("DELETE") and "pipeline_runs" in q]
        assert deletes
        # незавершённый прогон (started) исключён из DELETE
        assert "'started'" not in deletes[0]
        assert "status IN ('completed', 'failed', 'cancelled')" in deletes[0]

    def test_running_run_not_in_candidate_count(self, fake_db_factory):
        conn = fake_db_factory(counts={"pipeline_runs": 5})
        retention.main(_args(run_days=1))
        count_sql = [q for q in conn.cur.executed if "COUNT(*)" in q and "pipeline_runs" in q]
        assert count_sql
        # незавершённый прогон не попадает в COUNT кандидатов
        assert "'started'" not in count_sql[0]
        assert "status IN ('completed', 'failed', 'cancelled')" in count_sql[0]
