import asyncio
import functools
import time

import structlog

logger = structlog.get_logger("decorators")


def timed(log_key: str = ""):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                key = log_key or fn.__name__
                logger.info("timed", function=key, elapsed_sec=round(elapsed, 4))

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                key = log_key or fn.__name__
                logger.info("timed", function=key, elapsed_sec=round(elapsed, 4))

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper
    return decorator


def timed_async(log_key: str = ""):
    return timed(log_key)


class timed_block:
    """Context manager for timing inline blocks.

    Usage:
        with timed_block("Stage 1"):
            do_something()
    """

    def __init__(self, log_key: str):
        self.log_key = log_key
        self._start: float = 0.0
        self._logger = structlog.get_logger("decorators")

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self._logger.info("timed", function=self.log_key, elapsed_sec=round(elapsed, 4))
