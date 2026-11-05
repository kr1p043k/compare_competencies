"""N+1 regression proof: batch queries in the ZUN import + teacher students list.

For `_import_direction` the fake pool records every SQL round-trip.  The number
of DB calls must stay constant (independent of the number of disciplines/
competencies in the KRM file), and the response payload must keep the pre-fix
shape.  `teacher_students` must issue exactly one DB query no matter how many
students there are, and keep the response order.
"""

import json
from pathlib import Path

from src import config
from src.api_pkg.routers import teacher as teacher_mod
from src.api_pkg.routers import zun as zun_mod
from src.api_pkg.routers.profiles import _self_profile_name

DIR_CODE = "09.03.02"


def _uid(n: int) -> str:
    return f"{n:08d}-0000-0000-0000-000000000000"


def _build_krm(n: int) -> dict:
    """N disciplines, one competency each, with knowledge/abilities/skills."""
    disciplines = {}
    for i in range(1, n + 1):
        disciplines[f"Disc{i}"] = {
            "ksa": {
                f"OPK-{i}": {
                    "knowledge": [f"K{i}-1", f"K{i}-2"],
                    "abilities": [f"A{i}-1"],
                    "skills": [f"S{i}-1"],
                }
            }
        }
    return {DIR_CODE: {"disciplines": disciplines}}


class _Tx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class CountingPool:
    """Fake asyncpg-like pool recording every SQL round-trip."""

    def __init__(self, dirs, disc_rows, comp_rows, existing_rows=()):
        self._dirs = list(dirs)
        self._disc = list(disc_rows)
        self._comp = list(comp_rows)
        self._existing = list(existing_rows)
        self.sql_calls: list[str] = []
        self.inserted_rows: list[tuple] = []

    async def fetch(self, sql, *args):
        self.sql_calls.append("fetch")
        if "FROM directions" in sql:
            return self._dirs
        if "FROM disciplines" in sql:
            return self._disc
        if "FROM competencies" in sql:
            return self._comp
        if "FROM ksa_entries" in sql:
            return self._existing
        return []

    async def fetchval(self, sql, *args):
        self.sql_calls.append("fetchval")
        return None

    async def fetchrow(self, sql, *args):
        self.sql_calls.append("fetchrow")
        return None

    async def execute(self, sql, *args):
        self.sql_calls.append("execute")

    async def executemany(self, sql, args):
        self.sql_calls.append("executemany")
        self.inserted_rows.extend(list(args))

    def acquire(self):
        return _Conn(self)


class _Conn:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def transaction(self):
        return _Tx()

    async def executemany(self, sql, args):
        self.pool.sql_calls.append("executemany")
        self.pool.inserted_rows.extend(list(args))


def _make_import_pool(n: int, existing_rows=()):
    dir_id = _uid(900)
    dirs = [{"id": dir_id, "name": "Направление"}]
    disc_rows = [{"id": _uid(1000 + i), "name": f"Disc{i}"} for i in range(1, n + 1)]
    comp_rows = [
        {"id": _uid(2000 + i), "discipline_id": _uid(1000 + i), "code": f"OPK-{i}"}
        for i in range(1, n + 1)
    ]
    return CountingPool(dirs, disc_rows, comp_rows, existing_rows)


def _write_krm(tmp_path: Path, n: int) -> None:
    (tmp_path / f"krm_disciplines_{DIR_CODE}.json").write_text(
        json.dumps(_build_krm(n), ensure_ascii=False), encoding="utf-8"
    )


async def test_import_db_calls_constant_across_n(monkeypatch, tmp_path):
    """DB call count must not scale with the number of disciplines."""
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path)
    counts = {}
    for n in (3, 10):
        _write_krm(tmp_path, n)
        pool = _make_import_pool(n)
        await zun_mod._import_direction(pool, DIR_CODE, dry_run=False)
        counts[n] = len(pool.sql_calls)
    assert counts[3] == counts[10]
    # 4 batched SELECTs + 1 executemany, независимо от N.
    assert counts[3] <= 5
    assert counts[3] == 5


async def test_import_payload_matches_pre_fix_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path)
    _write_krm(tmp_path, 3)
    pool = _make_import_pool(3)
    result = await zun_mod._import_direction(pool, DIR_CODE, dry_run=False)

    assert result == {
        "dir_code": DIR_CODE,
        "inserted": 12,
        "skipped_dups": 0,
        "skipped_gap": 0,
        "disciplines_processed": 3,
    }

    expected_inserts = []
    for i in (1, 2, 3):
        cid = _uid(2000 + i)
        expected_inserts.append((cid, "knowledge", f"K{i}-1", 1))
        expected_inserts.append((cid, "knowledge", f"K{i}-2", 2))
        expected_inserts.append((cid, "abilities", f"A{i}-1", 3))
        expected_inserts.append((cid, "skills", f"S{i}-1", 4))
    assert pool.inserted_rows == expected_inserts


async def test_import_dedup_and_gap_semantics_preserved(monkeypatch, tmp_path):
    """Duplicate texts skip (once inserted) and filled categories skip entirely."""
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path)
    krm = {
        DIR_CODE: {
            "disciplines": {
                "Disc1": {
                    "ksa": {
                        "OPK-1": {
                            "knowledge": ["K1-1", "K1-2", "K1-2"],
                            "abilities": ["A1-1"],
                            "skills": ["S1-1"],
                        }
                    }
                }
            }
        }
    }
    (tmp_path / f"krm_disciplines_{DIR_CODE}.json").write_text(
        json.dumps(krm, ensure_ascii=False), encoding="utf-8"
    )
    # abilities уже заполнена -> gap; sort_order влияет на нумерацию новых строк.
    existing = [
        {"competency_id": _uid(2001), "ksa_type": "abilities",
         "original_text": "A1-1", "sort_order": 5},
    ]
    pool = _make_import_pool(1, existing)
    result = await zun_mod._import_direction(pool, DIR_CODE, dry_run=False)

    assert result == {
        "dir_code": DIR_CODE,
        "inserted": 3,
        "skipped_dups": 1,
        "skipped_gap": 1,
        "disciplines_processed": 1,
    }
    assert pool.inserted_rows == [
        (_uid(2001), "knowledge", "K1-1", 6),
        (_uid(2001), "knowledge", "K1-2", 7),
        (_uid(2001), "skills", "S1-1", 8),
    ]


async def test_import_dry_run_does_not_insert_or_dedupe(monkeypatch, tmp_path):
    """В dry_run дубли в файле не схлопываются (ничего не пишется в БД)."""
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path)
    krm = {
        DIR_CODE: {
            "disciplines": {
                "Disc1": {"ksa": {"OPK-1": {"knowledge": ["X", "X"],
                                            "abilities": [], "skills": []}}}
            }
        }
    }
    (tmp_path / f"krm_disciplines_{DIR_CODE}.json").write_text(
        json.dumps(krm, ensure_ascii=False), encoding="utf-8"
    )
    pool = _make_import_pool(1)
    result = await zun_mod._import_direction(pool, DIR_CODE, dry_run=True)

    assert result == {
        "dir_code": DIR_CODE,
        "inserted": 2,
        "skipped_dups": 0,
        "skipped_gap": 0,
        "disciplines_processed": 1,
    }
    assert pool.inserted_rows == []
    # dry_run: только 4 SELECT, без executemany.
    assert "executemany" not in pool.sql_calls
    assert len(pool.sql_calls) == 4


async def test_teacher_students_single_db_call_and_order(monkeypatch):
    emails = [f"s{i:02d}@example.org" for i in range(1, 11)]
    rows = [{"email": e, "full_name": f"Student {i}"} for i, e in enumerate(emails, 1)]

    students_dir = Path(config.DATA_DIR) / "students"
    students_dir.mkdir(parents=True, exist_ok=True)

    def _write_profile(email: str, data: dict) -> None:
        fpath = _self_profile_name(email)[1]
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        Path(fpath).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # s01 имеет заполненный профиль, s02 - пустой (defaults), остальные без файла.
    _write_profile(emails[0], {"target_level": "senior",
                                "skills": ["Python", "SQL"],
                                "competencies": ["C1"]})
    _write_profile(emails[1], {})

    class ListPool:
        def __init__(self, student_rows):
            self._rows = list(student_rows)
            self.fetch_calls = 0

        async def fetch(self, sql, *args):
            self.fetch_calls += 1
            return self._rows

    pool = ListPool(rows)
    monkeypatch.setattr(teacher_mod, "get_pool", lambda: pool)
    monkeypatch.setattr(teacher_mod.limiter, "enabled", False)

    result = await teacher_mod.teacher_students(None)

    assert pool.fetch_calls == 1
    assert result["total"] == len(emails)
    students = result["students"]
    assert [s["email"] for s in students] == emails  # порядок исходного списка

    expected = []
    for i, email in enumerate(emails, 1):
        item = {
            "email": email,
            "full_name": f"Student {i}",
            "target_level": None,
            "skills_count": 0,
            "competencies_count": 0,
            "has_profile": False,
        }
        expected.append(item)
    expected[0].update({"target_level": "senior", "skills_count": 2,
                        "competencies_count": 1, "has_profile": True})
    expected[1].update({"target_level": "middle", "has_profile": True})
    assert students == expected
