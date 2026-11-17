# tests/scripts/test_make_ensemble_seed.py
"""TASK 6.6.3: make_ensemble training must be deterministic under --seed.

Imports only scripts.experiments.make_ensemble (sklearn/numpy/joblib at module
level; engine/DB imports are deferred inside main), so no torch/xgboost/asyncpg.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiments.make_ensemble import parse_args, per_role_split  # noqa: E402


def _synthetic_rows() -> list[dict]:
    rows: list[dict] = []
    for role_i in range(8):
        role = f"role_{role_i}"
        for s in range(12):
            rows.append({"role": role, "skill": f"{role}_skill_{s}", "grade": s % 3})
    return rows


class TestSeededSplit:
    def test_same_seed_identical_arrays(self):
        rows = _synthetic_rows()
        tr1, te1 = per_role_split(rows, seed=42)
        tr2, te2 = per_role_split(rows, seed=42)
        assert tr1 == tr2
        assert te1 == te2

    def test_repeated_calls_deterministic(self):
        rows = _synthetic_rows()
        ref = per_role_split(rows, seed=7)
        for _ in range(3):
            assert per_role_split(rows, seed=7) == ref
        tr, te = ref
        assert set(tr).isdisjoint(te)
        assert sorted(tr + te) == list(range(len(rows)))

    def test_different_seed_changes_split(self):
        rows = _synthetic_rows()
        tr_a, te_a = per_role_split(rows, seed=1)
        tr_b, te_b = per_role_split(rows, seed=2)
        assert te_a != te_b  # seed actually threaded into random_state


class TestCliSeed:
    def test_default_seed_is_42(self):
        assert parse_args([]).seed == 42

    def test_seed_flag_parsed(self):
        assert parse_args(["--seed", "123"]).seed == 123

    def test_seed_flag_non_integer_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--seed", "abc"])
