"""Smoke tests for cross-test contamination that makes the suite order-dependent."""

import random
import re
import sys
import types
import unittest.mock
from pathlib import Path

import numpy as np

SYS_MODULE_STABLE_NAMES = {"shap", "cv2", "sentence_transformers"}


def test_no_magicmock_sys_modules_stubs():
    """Optional package stubs must be real module objects, not MagicMock instances."""
    for name in SYS_MODULE_STABLE_NAMES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        assert isinstance(mod, types.ModuleType), f"{name} leaked as {type(mod)}"
        assert not isinstance(mod, unittest.mock.MagicMock), f"{name} is a MagicMock"

    for name, mod in sys.modules.items():
        if name.startswith("test") or name.startswith("tests."):
            continue
        assert not isinstance(mod, unittest.mock.MagicMock), (
            f"sys.modules[{name!r}] is a MagicMock (order-dependent leak)"
        )


def test_random_reseeding_is_deterministic():
    """The conftest re-seeding fixture makes global PRNGs reproducible."""
    seed = 12345

    random.seed(seed)
    np.random.seed(seed)
    r1 = random.random()
    n1 = np.random.rand()

    random.seed(seed)
    np.random.seed(seed)
    r2 = random.random()
    n2 = np.random.rand()

    assert r1 == r2
    assert np.array_equal(n1, n2)


def test_helpers_does_not_use_bare_magicmock_as_path():
    """tests/pipeline/test_helpers.py must use real paths/loggers, not bare MagicMock()."""
    path = Path(__file__).with_name("pipeline") / "test_helpers.py"
    content = path.read_text(encoding="utf-8")

    assert "__truediv__.return_value = MagicMock()" not in content, (
        "bare MagicMock used as a filesystem path"
    )
    assert not re.search(r"save_detailed_vacancies\([^)]*MagicMock\(", content), (
        "save_detailed_vacancies still receives a MagicMock log/path"
    )
