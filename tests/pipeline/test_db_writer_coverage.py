"""Coverage for src/pipeline/db_writer.py - had no dedicated test file.

All DB I/O is faked: db_writer._pool() is monkeypatched with an in-memory
pool, JSON inputs live in tmp_path. No postgres connection is ever made.
"""
from __future__ import annotations

import json
import sys
import types
import uuid
from datetime import datetime

import pytest

from src.pipeline import db_writer
from src.pipeline.db_writer import (
    complete_pipeline_run,
    create_pipeline_run,
    save_coverage_from_json,
    save_trend_snapshot,
)


class _FakeConn:
    """In-memory asyncpg connection: records fetch/execute, returns canned rows."""

    def __init__(self, rows: list | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.fetched: list[tuple[str, tuple]] = []
        self.executed: list[tuple[str, tuple]] = []

    async def fetch(self, query: str, *args: object) -> list:
        self.fetched.append((query, args))
        return self.rows

    async def execute(self, query: str, *args: object) -> str:
        self.executed.append((query, args))
        return "OK"


class _AcquireCtx:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _FakePool:
    """Pool double: records execute() calls, counts how often _pool() is awaited."""

    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn
        self.executed: list[tuple[str, tuple]] = []
        self.pool_calls = 0

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self.conn)

    async def execute(self, query: str, *args: object) -> str:
        self.executed.append((query, args))
        return "OK"


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> _FakePool:
    conn = _FakeConn()
    pool = _FakePool(conn)

    async def _fake_pool() -> _FakePool:
        pool.pool_calls += 1
        return pool

    monkeypatch.setattr(db_writer, "_pool", _fake_pool)
    return pool


def _install_fake_vacancy_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ensure_extracted_skills imports VacancyParser lazily; stub the whole
    module so no real parser/dictionary loading happens in unit scope."""
    fake_mod = types.ModuleType("src.parsing.skills.vacancy_parser")

    class _FakeParser:
        pass

    fake_mod.VacancyParser = _FakeParser
    monkeypatch.setitem(sys.modules, "src.parsing.skills.vacancy_parser", fake_mod)


async def test_save_coverage_from_json_happy_path(tmp_path, fake_db):
    """Known discipline in an existing report is inserted exactly once with the
    computed (discipline_id, total, matched, ratio) tuple."""
    report = {
        "Информационные системы": {
            "skills": ["python", "django", "sql"],
            "market_skills": ["Python", "SQL"],
        }
    }
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    fake_db.conn.rows = [{"id": 7, "lowered": "информационные системы"}]

    saved = await save_coverage_from_json(path, run_id="run-42")

    assert saved == 1
    fetch_query, fetch_args = fake_db.conn.fetched[0]
    assert "FROM disciplines WHERE LOWER(name) = ANY" in fetch_query
    assert fetch_args == (["информационные системы"],)  # name is lowered, '_' -> ' '
    insert_query, insert_args = fake_db.conn.executed[0]
    assert "INSERT INTO coverage_analyses" in insert_query
    assert insert_args == (7, 3, 2, 0.6667)  # matched 2 of 3 skills


async def test_save_coverage_from_json_skips_unknown_discipline(tmp_path, fake_db):
    """Discipline missing from the DB table is skipped: returns 0, no INSERT."""
    report = {"Неизвестная": {"skills": ["python"], "market_skills": []}}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    fake_db.conn.rows = []  # disciplines lookup finds nothing

    saved = await save_coverage_from_json(path)

    assert saved == 0
    assert fake_db.conn.executed == []
    assert fake_db.pool_calls == 1  # DB was consulted, but nothing written


async def test_save_coverage_from_json_skip_inputs_never_touch_db(tmp_path, fake_db):
    """Boundary: None path, missing file and empty report all return 0 with zero
    pool acquisitions."""
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")

    assert await save_coverage_from_json(None) == 0
    assert await save_coverage_from_json(tmp_path / "missing.json") == 0
    assert await save_coverage_from_json(empty) == 0

    assert fake_db.pool_calls == 0
    assert fake_db.conn.fetched == []
    assert fake_db.conn.executed == []


async def test_create_pipeline_run_returns_uuid_and_inserts_row(fake_db):
    run_id = await create_pipeline_run("coverage", owner_id="user-1", direction_code="09.03.02")

    uuid.UUID(run_id)  # MUST be a valid UUID string; raises ValueError otherwise
    query, args = fake_db.executed[0]
    assert "INSERT INTO pipeline_runs" in query
    assert args[0] == run_id
    assert args[1] == "coverage"
    assert args[2:] == ("user-1", "09.03.02")


async def test_complete_pipeline_run_updates_status_error_stats(fake_db):
    await complete_pipeline_run("rid-1", status="failed", error="boom", stats={"vacancies": 10})

    query, args = fake_db.executed[0]
    assert "UPDATE pipeline_runs" in query
    assert args[0] == "failed"
    assert args[1] == "boom"
    assert json.loads(args[2]) == {"vacancies": 10}
    assert args[3] == "rid-1"


async def test_save_trend_snapshot_replaces_snapshot_and_invalidates_caches(fake_db, monkeypatch):
    invalidated: list[str] = []
    monkeypatch.setattr(db_writer, "_invalidate", lambda name: invalidated.append(name))

    await save_trend_snapshot(datetime(2026, 1, 1), {"python": 5}, source="hh_vacancies")

    (delete_query, delete_args), (insert_query, insert_args) = fake_db.executed
    assert "DELETE FROM trend_snapshots" in delete_query
    assert delete_args == (datetime(2026, 1, 1), "hh_vacancies")
    assert "INSERT INTO trend_snapshots" in insert_query
    assert json.loads(insert_args[2]) == {"python": 5}
    # Trend caches are stale after a new snapshot - invalidation is contractual.
    assert invalidated == ["trends", "forecast"]


def test_ensure_extracted_skills_canonicalizes_existing_lists(monkeypatch):
    """Pre-computed skills are only normalized (trim, lower, dedup) - no re-parse."""
    _install_fake_vacancy_parser(monkeypatch)
    vacs = [
        {"extracted_skills": [" SQL ", "sql", ""]},
        {"raw_data": {"extracted_skills": ["Python", "python"]}},
    ]

    filled = db_writer._ensure_extracted_skills(vacs)

    assert filled == 0
    assert vacs[0]["extracted_skills"] == ["sql"]
    assert vacs[1]["extracted_skills"] == ["python"]


def test_ensure_extracted_skills_skips_vacancies_without_text(monkeypatch):
    """No description/key_skills and no pre-computed skills -> nothing to parse."""
    _install_fake_vacancy_parser(monkeypatch)
    vacs = [{"id": "1"}]

    filled = db_writer._ensure_extracted_skills(vacs)

    assert filled == 0
    assert "extracted_skills" not in vacs[0]
