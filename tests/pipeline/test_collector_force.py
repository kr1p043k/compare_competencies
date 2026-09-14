"""Unit: collector force params keep default behavior."""
import asyncio

import pytest


def test_try_collect_signature():
    import inspect
    from src.pipeline.background_collector import _try_collect
    sig = inspect.signature(_try_collect)
    assert sig.parameters["force_period_days"].default is None
    assert sig.parameters["force"].default is False
