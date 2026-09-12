"""Unit: breaker closed -> open -> half-open -> closed."""
import time

from src.circuit_breaker import CircuitBreaker


def test_closed_allows_and_resets():
    b = CircuitBreaker(fail_threshold=3, cooldown_s=60)
    assert b.state == "closed" and b.allow()
    b.record_failure()
    b.record_success()
    assert b.consecutive_failures == 0 and b.state == "closed"


def test_opens_after_threshold():
    b = CircuitBreaker(fail_threshold=2, cooldown_s=3600)
    b.record_failure()
    assert b.allow()
    b.record_failure()
    assert b.state == "open" and not b.allow()


def test_half_open_trial(monkeypatch):
    b = CircuitBreaker(fail_threshold=1, cooldown_s=10)
    b.record_failure()
    assert b.state == "open"
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 11)
    assert b.state == "half-open" and b.allow()
    b.record_success()
    assert b.state == "closed" and b.allow()


def test_half_open_failure_reopens(monkeypatch):
    b = CircuitBreaker(fail_threshold=5, cooldown_s=10)
    for _ in range(5):
        b.record_failure()
    assert b.state == "open"
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 11)
    assert b.allow()
    b.record_failure()
    assert b.state == "open" and not b.allow()
