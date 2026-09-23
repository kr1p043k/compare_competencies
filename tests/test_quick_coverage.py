"""Быстрые pure-unit тесты для поднятия покрытия: Result, RetryPolicy,
CurriculumOptimizer, ru_morph, CacheManager. Без сети, БД и тяжелых моделей.
"""

import pytest

from src import Err, Ok


class TestResultHelpers:
    def test_ok_chain(self):
        r = Ok(2).map(lambda x: x * 3).and_then(lambda x: Ok(x + 1))
        assert r.ok() == 7
        assert r.is_ok() and not r.is_err()

    def test_err_shortcircuit(self):
        from src.errors import DomainError

        e = Err(DomainError(message="x"))
        assert e.map(lambda x: 1) is e
        assert e.and_then(lambda x: Ok(1)) is e
        assert e.is_err()
        assert "x" in str(e.err())

    def test_or_else_and_unwrap(self):
        from src.errors import DomainError

        assert Ok(1).or_else(lambda e: Ok(2)).ok() == 1
        assert Err(DomainError(message="x")).or_else(lambda e: Ok(2)).ok() == 2
        assert Ok(1).unwrap_or(9) == 1
        assert Err(DomainError(message="x")).unwrap_or(9) == 9
        assert Ok(1).expect("boom") == 1
        with pytest.raises(Exception):
            Err(DomainError(message="boom")).expect("boom")
        assert Ok(1).map_err(lambda e: e) is not None


class TestRetryPolicy:
    def test_ok_first_try(self):
        from src.retry import RetryPolicy

        calls = []
        r = RetryPolicy(max_retries=2).execute(lambda: (calls.append(1), Ok("v"))[1])
        assert r.ok() == "v"
        assert len(calls) == 1

    def test_err_then_ok(self):
        from src.retry import RetryPolicy
        from src.errors import DomainError

        calls = []

        def flaky():
            calls.append(1)
            return Ok("v") if len(calls) == 2 else Err(DomainError(message="tmp"))

        assert RetryPolicy(max_retries=2, jitter=False).execute(flaky).ok() == "v"
        assert len(calls) == 2

    def test_exhausted(self):
        from src.retry import RetryPolicy
        from src.errors import DomainError

        r = RetryPolicy(max_retries=1, jitter=False).execute(
            lambda: Err(DomainError(message="always"))
        )
        assert r.is_err()

    def test_non_retryable(self):
        from src.retry import RetryPolicy

        r = RetryPolicy(max_retries=3, retryable_exceptions=(ValueError,)).execute(
            lambda: (_ for _ in ()).throw(KeyError("nope"))
        )
        assert r.is_err()
        assert "non-retryable" in str(r.err())

    def test_retryable_exception_then_ok(self):
        from src.retry import RetryPolicy

        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("tmp")
            return Ok("v")

        assert RetryPolicy(max_retries=2, jitter=False).execute(flaky).ok() == "v"

    async def test_async_ok(self):
        from src.retry import RetryPolicy

        async def fn():
            return Ok("v")

        assert (await RetryPolicy(max_retries=1).execute_async(fn)).ok() == "v"

    async def test_async_exhausted(self):
        from src.retry import RetryPolicy
        from src.errors import DomainError

        async def fn():
            return Err(DomainError(message="nope"))

        assert (await RetryPolicy(max_retries=1, jitter=False).execute_async(fn)).is_err()


class TestCurriculumOptimizer:
    def _summary(self, **kw):
        from src.models.teacher_analysis import DirectionSummary

        base = dict(
            direction_code="09.03.02",
            direction_name="d",
            profile="base",
        )
        base.update(kw)
        return DirectionSummary(**base)

    def test_none_and_empty_err(self):
        from src.predictors.curriculum_optimizer import CurriculumOptimizer

        assert CurriculumOptimizer().optimize(None).is_err()
        assert CurriculumOptimizer().optimize(self._summary()).is_err()

    def test_low_coverage_and_emerging(self):
        from src.predictors.curriculum_optimizer import CurriculumOptimizer

        s = self._summary(
            disciplines=[
                {"name": "Философия длинное название", "coverage_level": "low", "gaps": 5},
                {"name": "Матан", "coverage_level": "high", "gaps": 150},
            ],
            top_emerging=[{"skill": "rust"}, {"skill": "go"}],
        )
        recs = CurriculumOptimizer().optimize(s).ok()
        types = {r.type for r in recs}
        assert "major_revision" in types
        assert "add_new_content" in types
        assert "update_content" in types


class TestRuMorph:
    def test_lemma_russian(self):
        from src.text.ru_morph import available, lemma, lemma_key

        assert available() is True
        assert lemma("компетенций") == "компетенция"
        assert lemma_key("Анализ данных!") != ""


class TestCacheManager:
    def test_save_load_exists_invalidate(self, tmp_path):
        from src.cache_manager import CacheManager

        cm = CacheManager(tmp_path)
        assert cm.exists("k") is False
        cm.save("k", {"a": 1})
        assert cm.exists("k") is True
        assert cm.load("k").ok() == {"a": 1}
        cm.invalidate("k")
        assert cm.exists("k") is False

    def test_load_missing_none(self, tmp_path):
        from src.cache_manager import CacheManager

        assert CacheManager(tmp_path).load("nope").is_err()
