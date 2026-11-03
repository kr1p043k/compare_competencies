"""Инвалидация кэшей: кэш не должен отдавать устаревшие цифры после записи.

Раньше TTL был единственным механизмом: после пересчёта пайплайна
/forecast/all до 5 минут и /competency-trends до 10 минут отдавали старые
данные. Здесь проверяем, что шина сбрасывает оба кэша по событию.
"""
import time

import pytest

# Импортируем роутеры на этапе сбора тестов: они регистрируют свои кэши в шине
# при импорте. Иначе первый снимок состояния в фикстуре не увидит их подписки.
import src.api_pkg.routers.forecast as _forecast_mod  # noqa: F401
import src.api_pkg.routers.trends as _trends_mod  # noqa: F401
from src.cache_bus import invalidate, invalidate_all, register, registered


@pytest.fixture(autouse=True)
def _clean_bus():
    """Изолировать тесты, не ломая реальные подписки.

    Шина глобальная, и в ней уже есть боевые подписчики (forecast, trends).
    Поэтому запоминаем исходное состояние и после теста откатываем к нему,
    а не чистим всё подряд - иначе тесты инвалидации по имени не находили бы
    своих подписчиков.
    """
    import src.cache_bus as bus

    saved = {k: list(v) for k, v in bus._INVALIDATORS.items()}
    yield
    bus._INVALIDATORS.clear()
    bus._INVALIDATORS.update({k: list(v) for k, v in saved.items()})


class TestCacheBus:
    def test_register_and_invalidate(self):
        calls = []
        register("demo", lambda: calls.append(1))
        assert invalidate("demo") == 1
        assert calls == [1]

    def test_register_is_idempotent(self):
        calls = []

        def cb():
            calls.append(1)

        register("demo", cb)
        register("demo", cb)
        invalidate("demo")
        assert calls == [1], "один и тот же callback не должен зваться дважды"

    def test_invalidate_unknown_cache_is_noop(self):
        assert invalidate("nope") == 0

    def test_failing_callback_does_not_block_others(self):
        calls = []

        def boom():
            raise RuntimeError("сломался")

        register("demo", boom)
        register("demo", lambda: calls.append("ok"))
        ok = invalidate("demo")
        assert ok == 1
        assert calls == ["ok"], "остальные подписчики обязаны отработать"

    def test_invalidate_all_covers_every_cache(self):
        calls = []
        register("a", lambda: calls.append("a"))
        register("b", lambda: calls.append("b"))
        invalidate_all()
        assert sorted(calls) == ["a", "b"]

    def test_registered_reports_counts(self):
        register("x", lambda: None)
        register("x", lambda: None)
        assert registered()["x"] == 2


class TestForecastCacheInvalidation:
    def test_clear_drops_all_three_caches(self):
        from src.api_pkg.routers import forecast as fc

        fc._TOP_CACHE[("k",)] = (time.monotonic(), (1,))
        fc._ALL_CACHE[("k",)] = (time.monotonic(), (1,))
        fc._META_CACHE[("k",)] = (time.monotonic(), {"a": 1})

        fc._clear_forecast_caches()

        assert fc._TOP_CACHE == {}
        assert fc._ALL_CACHE == {}
        assert fc._META_CACHE == {}

    def test_invalidate_by_name_clears_forecast(self):
        from src.api_pkg.routers import forecast as fc

        fc._ALL_CACHE[("k",)] = (time.monotonic(), (1,))
        invalidate("forecast")
        assert fc._ALL_CACHE == {}


class TestTrendCacheInvalidation:
    def test_clear_drops_computed_trends(self):
        from src.api_pkg.routers import trends as tr

        tr._CT_CACHE[("rising", 200)] = (time.monotonic(), {"total": 1})
        tr._clear_trend_caches()
        assert tr._CT_CACHE == {}

    def test_invalidate_by_name_clears_trends(self):
        from src.api_pkg.routers import trends as tr

        tr._CT_CACHE[("rising", 200)] = (time.monotonic(), {"total": 1})
        invalidate("trends")
        assert tr._CT_CACHE == {}


class TestWriterInvalidates:
    """Запись данных обязана сама сбрасывать кэши."""

    def test_vacancies_batch_calls_invalidate(self, monkeypatch):
        from src.pipeline import db_writer

        called: list[tuple] = []
        monkeypatch.setattr(db_writer, "_invalidate_caches", lambda *n: called.append(n))

        class Pool:
            async def executemany(self, *a, **kw):
                return None

        async def _pool():
            return Pool()

        monkeypatch.setattr(db_writer, "_pool", _pool)

        import asyncio

        asyncio.run(
            db_writer.save_vacancies_batch(
                [{"id": 1, "name": "Dev", "key_skills": [{"name": "python"}]}]
            )
        )

        assert called, "save_vacancies_batch должен инвалидировать кэши"
        assert "forecast" in called[0]
