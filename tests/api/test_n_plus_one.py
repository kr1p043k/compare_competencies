"""Доказательство, что импорт ZUN делает CONSTANT число обращений к БД.

Было: на каждый элемент - EXISTS по типу, EXISTS по тексту, MAX(sort_order)+1
и INSERT, то есть число запросов росло вместе с размером файла (N+1).
Стало: 4 чтения + один executemany, не зависящие от N.

Тест вызывает `_import_direction` напрямую (юнит-тест батчинга), поэтому не
зависит от auth-гейтов эндпоинта.
"""

from __future__ import annotations

import json

import src.api_pkg.routers.zun as zun_mod

DIR_ID = "11111111-1111-1111-1111-111111111111"
DISC_ID = "22222222-2222-2222-2222-222222222222"
COMP_ID = "33333333-3333-3333-3333-333333333333"


class CountingPool:
    """Пул, считающий каждый вызов и отдающий готовые строки."""

    def __init__(self, existing=None):
        self.calls: list[str] = []
        self._existing = existing or []

    async def fetch(self, sql, *a):
        self.calls.append("fetch")
        flat = " ".join(sql.split())
        if "FROM directions" in flat:
            return [{"id": DIR_ID, "name": "Направление"}]
        if "FROM disciplines" in flat:
            return [{"id": DISC_ID, "name": "Дисциплина"}]
        if "FROM competencies" in flat:
            return [{"id": COMP_ID, "discipline_id": DISC_ID, "code": "K1"}]
        if "FROM ksa_entries" in flat:
            return list(self._existing)
        return []

    async def executemany(self, sql, rows):
        self.calls.append("executemany")

    def acquire(self):
        pool = self

        class _Conn:
            async def executemany(self, sql, rows):
                pool.calls.append("executemany")

            def transaction(self):
                return _NullCtx()

        return _ConnCtx(_Conn())


class _NullCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _ConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


def _krm_file(tmp_path, n_texts: int):
    texts = [f"навык номер {i}" for i in range(n_texts)]
    payload = {
        "09.03.02": {
            "disciplines": {
                "Дисциплина": {
                    "ksa": {"K1": {"knowledge": texts, "abilities": [], "skills": []}},
                }
            }
        }
    }
    path = tmp_path / "krm_disciplines_09.03.02.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


async def _run(tmp_path, monkeypatch, pool, n_texts, dry_run):
    _krm_file(tmp_path, n_texts)
    monkeypatch.setattr(zun_mod.config, "REFERENCE_DIR", tmp_path, raising=False)
    return await zun_mod._import_direction(pool, "09.03.02", dry_run)


async def test_db_call_count_is_constant_across_n(tmp_path, monkeypatch):
    pool3 = CountingPool()
    await _run(tmp_path, monkeypatch, pool3, 3, dry_run=False)

    pool10 = CountingPool()
    await _run(tmp_path, monkeypatch, pool10, 10, dry_run=False)

    n3, n10 = len(pool3.calls), len(pool10.calls)
    assert n3 == n10, f"число запросов зависит от N: N=3 -> {n3}, N=10 -> {n10}"
    assert n3 <= 6, f"слишком много обращений к БД: {pool3.calls}"
    assert pool3.calls.count("executemany") == 1, pool3.calls
    assert pool3.calls.count("fetch") == 4, pool3.calls


async def test_import_payload_shape(tmp_path, monkeypatch):
    pool = CountingPool()
    body = await _run(tmp_path, monkeypatch, pool, 5, dry_run=False)
    assert body == {
        "dir_code": "09.03.02",
        "inserted": 5,
        "skipped_dups": 0,
        "skipped_gap": 0,
        "disciplines_processed": 1,
    }


async def test_dry_run_does_not_write(tmp_path, monkeypatch):
    pool = CountingPool()
    body = await _run(tmp_path, monkeypatch, pool, 4, dry_run=True)
    assert body["inserted"] == 4
    assert "executemany" not in pool.calls, "dry-run не должен ничего писать"
    assert pool.calls.count("fetch") == 4


async def test_gap_fill_skips_block_when_type_exists(tmp_path, monkeypatch):
    """Если у компетенции уже есть записи типа - весь блок пропускается."""
    existing = [{
        "competency_id": COMP_ID,
        "ksa_type": "knowledge",
        "original_text": "старое",
        "sort_order": 1,
    }]
    pool = CountingPool(existing=existing)
    body = await _run(tmp_path, monkeypatch, pool, 5, dry_run=False)

    assert body["skipped_gap"] == 5
    assert body["inserted"] == 0
    assert "executemany" not in pool.calls


async def test_duplicate_text_is_skipped(tmp_path, monkeypatch):
    """Текст, уже присутствующий в БД, считается дублем и не вставляется."""
    existing = [{
        "competency_id": COMP_ID,
        "ksa_type": "knowledge",
        "original_text": "навык номер 0",
        "sort_order": 1,
    }]
    pool = CountingPool(existing=existing)
    # gap-fill не срабатывает только если тип не считается «уже наполненным».
    # Здесь тип наполнен, поэтому проверяем именно работу дедупликации текстов
    # через режим dry_run, где gap-fill не применяется.
    body = await _run(tmp_path, monkeypatch, pool, 3, dry_run=True)
    assert body["skipped_dups"] == 1
    assert body["inserted"] == 2
