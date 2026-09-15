"""Integration: save_vacancies_batch persists extracted_skills (fake hh_id, cleaned up)."""
import asyncpg
import pytest

DSN = "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies"
FAKE_HH = -777777777


async def test_save_batch_persists_extracted_skills():
    from src.pipeline.db_writer import save_vacancies_batch
    v = {"id": str(FAKE_HH), "name": "TEST SWEEP PROBE", "experience": {},
         "salary": {}, "employer": {}, "area": {}, "snippet": {},
         "key_skills": [], "description": "test", "published_at": None,
         "alternate_url": None, "extracted_skills": ["python", "sql"]}
    n = await save_vacancies_batch([v])
    assert n == 1
    con = await asyncpg.connect(DSN)
    try:
        row = await con.fetchrow(
            "SELECT parsed_skills FROM vacancies WHERE hh_id = $1", FAKE_HH)
        assert row is not None
        val = row["parsed_skills"]
        if isinstance(val, str):
            import json as _json
            val = _json.loads(val)
        assert sorted(val) == ["python", "sql"]
    finally:
        await con.execute("DELETE FROM vacancies WHERE hh_id = $1", FAKE_HH)
        left = await con.fetchval(
            "SELECT COUNT(*) FROM vacancies WHERE hh_id = $1", FAKE_HH)
        assert left == 0
        await con.close()
