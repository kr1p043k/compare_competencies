"""GET /admin/llm/cost: сводка стоимости LLM с fallback (нет БД / ошибка схемы)."""
from __future__ import annotations

from datetime import date

import src.api_pkg.routers.admin as admin_mod


class _FakePool:
    def __init__(self, by_model, by_day, exc=None):
        self._by_model = by_model
        self._by_day = by_day
        self._exc = exc

    async def fetch(self, sql, *args):
        if self._exc is not None:
            raise self._exc
        if "group by model" in sql.lower():
            return self._by_model
        return self._by_day


async def test_summary_ok():
    pool = _FakePool(
        by_model=[
            {"model": "yandexgpt-lite", "requests": 3, "cost_usd": 0.0111},
            {"model": "gpt-oss:120b", "requests": 5, "cost_usd": 0},
        ],
        by_day=[{"day": date(2026, 1, 1), "requests": 8, "cost_usd": 0.0111}],
    )
    out = await admin_mod._llm_cost_summary(30, pool=pool)
    assert out["reason"] is None
    assert out["total_usd"] == 0.0111
    assert out["requests"] == 8
    assert out["by_model"][0] == {"model": "yandexgpt-lite", "requests": 3, "cost_usd": 0.0111}
    assert out["by_day"][0] == {"day": "2026-01-01", "requests": 8, "cost_usd": 0.0111}


async def test_summary_without_pool(monkeypatch):
    monkeypatch.setattr("src.db.get_pool", lambda: None)
    out = await admin_mod._llm_cost_summary(30)
    assert out["reason"]
    assert out["total_usd"] == 0.0
    assert out["by_model"] == []


async def test_summary_schema_error_falls_back():
    pool = _FakePool([], [], exc=RuntimeError('column "cost_usd" does not exist'))
    out = await admin_mod._llm_cost_summary(30, pool=pool)
    assert out["reason"]
    assert "cost_usd" in out["reason"]
    assert out["by_model"] == []
    assert out["total_usd"] == 0.0
