import time
from unittest.mock import MagicMock, patch

import pytest

from src import Err, Ok
from src.errors import DomainError
from src.retry import RetryPolicy


class TestRetryPolicyInit:
    def test_default_params(self):
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay == 1.0
        assert p.max_delay == 60.0
        assert p.backoff_factor == 2.0
        assert p.jitter is True

    def test_custom_params(self):
        p = RetryPolicy(max_retries=5, base_delay=0.5, max_delay=30.0, backoff_factor=1.5, jitter=False)
        assert p.max_retries == 5
        assert p.jitter is False

    def test_description(self):
        p = RetryPolicy(max_retries=2)
        assert "RetryPolicy" in p.description
        assert "retries=2" in p.description


class TestBackoff:
    def test_exponential(self):
        p = RetryPolicy(base_delay=1.0, backoff_factor=2.0, jitter=False)
        assert p._backoff(0) == 1.0
        assert p._backoff(1) == 2.0
        assert p._backoff(2) == 4.0

    def test_max_delay_cap(self):
        p = RetryPolicy(base_delay=10, backoff_factor=10, max_delay=50, jitter=False)
        assert p._backoff(0) == 10.0
        assert p._backoff(1) == 50.0
        assert p._backoff(5) == 50.0

    def test_jitter_range(self):
        p = RetryPolicy(base_delay=10, jitter=True)
        for attempt in range(5):
            d = p._backoff(attempt)
            assert 0.5 * 10 <= d <= 1.0 * 10 * (2 ** attempt)


class TestExecuteOk:
    def test_ok_first_attempt(self):
        fn = MagicMock(return_value=Ok("success"))
        p = RetryPolicy(max_retries=3)
        result = p.execute(fn)
        assert result == Ok("success")
        fn.assert_called_once()

    def test_ok_after_retries(self):
        calls = [Err(DomainError("fail")), Err(DomainError("fail")), Ok("ok")]
        fn = MagicMock(side_effect=calls)
        p = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
        with patch.object(time, "sleep") as mock_sleep:
            result = p.execute(fn)
            assert result == Ok("ok")
            assert fn.call_count == 3
            assert mock_sleep.call_count == 2


class TestExecuteErr:
    def test_all_fail(self):
        fn = MagicMock(return_value=Err(DomainError("fail")))
        p = RetryPolicy(max_retries=2, base_delay=0.01, jitter=False)
        with patch.object(time, "sleep"):
            result = p.execute(fn)
            assert result.is_err()
            assert "fail" in result.err().message

    def test_exception_retryable(self):
        fn = MagicMock(side_effect=ValueError("bad"))
        p = RetryPolicy(max_retries=1, base_delay=0.01, jitter=False)
        with patch.object(time, "sleep"):
            result = p.execute(fn)
            assert result.is_err()

    def test_exception_not_retryable(self):
        fn = MagicMock(side_effect=TypeError("bad type"))
        p = RetryPolicy(max_retries=1, retryable_exceptions=(ValueError,), base_delay=0.01, jitter=False)
        with patch.object(time, "sleep"):
            result = p.execute(fn)
            assert result.is_err()

    def test_exception_last_attempt(self):
        fn = MagicMock(side_effect=ValueError("fail"))
        p = RetryPolicy(max_retries=0, base_delay=0.01, jitter=False)
        result = p.execute(fn)
        assert result.is_err()
        assert "failed after 0 retries" in result.err().message

    def test_empty_retry_policy(self):
        fn = MagicMock(return_value=Err(DomainError("fail")))
        p = RetryPolicy(max_retries=0, base_delay=0.01, jitter=False)
        result = p.execute(fn)
        assert result.is_err()


class TestExecuteArgs:
    def test_passes_args(self):
        fn = MagicMock(return_value=Ok("ok"))
        p = RetryPolicy(max_retries=0)
        p.execute(fn, 1, key="val")
        fn.assert_called_once_with(1, key="val")
