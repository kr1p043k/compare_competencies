"""Centralized retry policy with exponential backoff."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

import structlog

from src import Err, Ok, Result
from src.errors import DomainError

T = TypeVar("T")
logger = structlog.get_logger("retry")


class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    def execute(self, fn: Callable[..., Result[T, DomainError]], *args: Any, **kwargs: Any) -> Result[T, DomainError]:
        last_error: DomainError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                match result:
                    case Ok(_):
                        return result
                    case Err(e):
                        last_error = e
                        if attempt < self.max_retries:
                            delay = self._backoff(attempt)
                            logger.warning(
                                "retry_attempt",
                                attempt=attempt + 1,
                                max_retries=self.max_retries,
                                delay=round(delay, 2),
                                error=str(e),
                            )
                            time.sleep(delay)
                            continue
                        return Err(e)
            except Exception as e:
                if not isinstance(e, self.retryable_exceptions):
                    return Err(DomainError(message=f"non-retryable: {e}"))
                last_error = DomainError(message=str(e))
                if attempt < self.max_retries:
                    delay = self._backoff(attempt)
                    logger.warning("retry_exception", attempt=attempt + 1, delay=round(delay, 2), error=str(e))
                    time.sleep(delay)
                    continue
                return Err(DomainError(message=f"failed after {self.max_retries} retries: {e}"))
        return Err(last_error or DomainError(message="retry policy exhausted"))

    def _backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5
        return delay

    @property
    def description(self) -> str:
        return f"RetryPolicy(retries={self.max_retries}, backoff={self.backoff_factor}, max_delay={self.max_delay}s)"
