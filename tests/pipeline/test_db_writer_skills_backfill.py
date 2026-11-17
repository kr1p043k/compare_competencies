"""Task 6.4.1: backfill of the key_skills column from parsed skills.

HH basic feed carries no key_skills; skills are computed into
extracted_skills. save_vacancies_batch must fill the column from them
without overwriting real key_skills. No DB: _pool is monkeypatched with
a fake that captures executemany rows.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from src.pipeline import db_writer
from src.pipeline.db_writer import save_vacancies_batch


class _FakePoolExecutemany:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[tuple]]] = []

    async def executemany(self, query: str, rows: list[tuple]) -> None:
        self.executed.append((query, list(rows)))


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> _FakePoolExecutemany:
    pool = _FakePoolExecutemany()

    async def _fake_pool() -> _FakePoolExecutemany:
        return pool

    monkeypatch.setattr(db_writer, "_pool", _fake_pool)
    monkeypatch.setattr(db_writer, "_invalidate_caches", lambda *names: None)
    return pool


def _install_fake_vacancy_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub VacancyParser module: _ensure_extracted_skills imports it lazily."""
    fake_mod = types.ModuleType("src.parsing.skills.vacancy_parser")

    class _FakeParser:
        pass

    fake_mod.VacancyParser = _FakeParser
    monkeypatch.setitem(sys.modules, "src.parsing.skills.vacancy_parser", fake_mod)


def _row_for(pool: _FakePoolExecutemany, hh_id: int) -> tuple:
    query, rows = pool.executed[0]
    assert "INSERT INTO vacancies" in query
    (row,) = [r for r in rows if r[0] == hh_id]
    return row


async def test_basic_vacancy_key_skills_backfilled_from_parsed(
    fake_pool, monkeypatch
):
    """Basic HH vacancy: no key_skills, parsed skills present -> the
    key_skills column gets the parsed list (indices: 12 = key_skills json)."""
    _install_fake_vacancy_parser(monkeypatch)
    vacancies = [
        {
            "id": "111",
            "name": "Python-dev",
            "extracted_skills": ["python", "sql"],  # как после парсинга описания
        }
    ]

    saved = await save_vacancies_batch(vacancies)

    assert saved == 1
    row = _row_for(fake_pool, 111)
    assert json.loads(row[12]) == ["python", "sql"]
    # parsed_skills не дублирует логику: колонка навыка та же, что и раньше
    assert json.loads(row[13]) == ["python", "sql"]


async def test_real_key_skills_not_overwritten(fake_pool, monkeypatch):
    """HH vacancy WITH key_skills keeps its own values untouched."""
    _install_fake_vacancy_parser(monkeypatch)
    vacancies = [
        {
            "id": "222",
            "name": "Analyst",
            "key_skills": [{"name": "Excel"}, {"name": ""}, {"name": "SQL"}],
            "extracted_skills": ["excel", "sql", "python"],
        }
    ]

    await save_vacancies_batch(vacancies)

    row = _row_for(fake_pool, 222)
    assert json.loads(row[12]) == ["Excel", "SQL"]  # originals, not parsed


async def test_no_key_skills_and_no_parsed_stays_empty(fake_pool, monkeypatch):
    """Basic vacancy without description-derived skills: column stays []
    (nothing invented, no crash)."""
    _install_fake_vacancy_parser(monkeypatch)
    vacancies = [{"id": "333", "name": "Bare basic"}]

    await save_vacancies_batch(vacancies)

    row = _row_for(fake_pool, 333)
    assert json.loads(row[12]) == []
    assert row[13] is None
