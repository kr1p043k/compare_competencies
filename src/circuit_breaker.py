"""Tiny circuit breaker for outbound integrations (hh.ru collector).

States: closed -> open (after N consecutive failures) -> half-open (one trial
after cooldown) -> closed/open. Serving path is never touched: the market cache
is the fallback, so an open breaker only skips collection attempts.
"""
from __future__ import annotations

import time


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 3, cooldown_s: float = 6 * 3600):
        if fail_threshold < 1:
            raise ValueError("fail_threshold must be >= 1")
        self._threshold = fail_threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self._cooldown:
            return "half-open"
        return "open"

    def allow(self) -> bool:
        """True if a call attempt may proceed (closed or half-open trial)."""
        return self.state in ("closed", "half-open")

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self.state == "half-open" or self._failures >= self._threshold:
            self._opened_at = time.monotonic()

    @property
    def consecutive_failures(self) -> int:
        return self._failures
