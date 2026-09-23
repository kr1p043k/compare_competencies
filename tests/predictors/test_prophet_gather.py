"""Регрессия пустой строки и регистровых дублей в Prophet-историях."""
from collections import Counter
from datetime import date

import pytest

from src.predictors.prophet_forecast import (
    ProphetForecastEngine,
    Snapshot,
    _has_mixed_token,
    anchor_monthly_counts,
    drop_broken_snapshots,
    merge_supplement_rows,
)
from src.predictors.skill_forecast import compute_observed_drops


def _snaps():
    return [
        Snapshot(date(2026, 5, 1), {"python": 50, "": 10, "1c": 5}),
        Snapshot(date(2026, 6, 1), {"python": 60, "   ": 3, "1C": 7}),
    ]


def test_gather_drops_blanks_and_merges_case():
    h = ProphetForecastEngine()._gather_history(_snaps())
    assert "" not in h
    assert not any(not k.strip() for k in h)
    assert "1c" in h or "1C" in h
    merged = h.get("1c", h.get("1C"))
    assert len(merged) == 2  # по точке на месяц, без дублей
    assert h["python"] == [(date(2026, 5, 1), 50), (date(2026, 6, 1), 60)]


def test_gather_sums_same_month_dupes():
    h = ProphetForecastEngine()._gather_history([
        Snapshot(date(2026, 5, 1), {"1c": 5, "1C": 7}),
    ])
    key = "1c" if "1c" in h else "1C"
    assert h[key] == [(date(2026, 5, 1), 12)]


def test_fit_has_no_blank_skills():
    eng = ProphetForecastEngine()
    res = eng.fit(_snaps(), fallback_freqs={"python": 60})
    assert res.is_ok()
    assert "" not in eng._skill_history
    assert "" not in eng._models
    assert "" not in eng._last_actual_freq


class TestAnchorMonthlyCounts:
    APR = date(2026, 4, 1)
    AUG = date(2026, 8, 1)

    def test_volume_growth_flattened(self):
        # один и тот же share (10%) при росте коллекции 4x: было бы +300% "роста"
        monthly = {self.APR: Counter({"a": 100}), self.AUG: Counter({"a": 400})}
        totals = {self.APR: 1000, self.AUG: 4000}
        out = anchor_monthly_counts(monthly, totals)
        assert out[self.APR] == {"a": 400.0}
        assert out[self.AUG] == {"a": 400.0}

    def test_real_share_shift_preserved(self):
        monthly = {self.APR: Counter({"a": 100}), self.AUG: Counter({"a": 800})}
        totals = {self.APR: 1000, self.AUG: 4000}
        out = anchor_monthly_counts(monthly, totals)
        assert out[self.APR] == {"a": 400.0}
        assert out[self.AUG] == {"a": 800.0}

    def test_last_month_untouched_and_zero_skipped(self):
        monthly = {self.APR: Counter({"a": 5})}
        assert anchor_monthly_counts(monthly, {}) == {}
        assert anchor_monthly_counts(monthly, {self.APR: 0}) == {}
        out = anchor_monthly_counts(
            {self.APR: Counter({"a": 5}), self.AUG: Counter({"a": 20})},
            {self.APR: 100, self.AUG: 200},
        )
        assert out[self.AUG] == {"a": 20}


class TestMixedToken:
    def test_junk_detected(self):
        assert _has_mixed_token("cиcтeмнoe мышлeниe") is True
        assert _has_mixed_token("paбoтa в кoмaндe") is True

    def test_legit_kept(self):
        assert _has_mixed_token("a/b тестирование") is False
        assert _has_mixed_token("c/c++") is False
        assert _has_mixed_token("1c предприятие") is False
        assert _has_mixed_token("sql") is False


class TestMergeSupplement:
    MAY = date(2026, 5, 1)

    def test_homoglyph_merged_canonical_wins(self):
        rows = [
            (self.MAY, "cиcтeмнoe мышлeниe", 137),
            (self.MAY, "системное мышление", 245),
        ]
        out = merge_supplement_rows(rows, set())
        assert out[self.MAY] == {"системное мышление": 382}

    def test_case_merged(self):
        rows = [(self.MAY, "1c", 100), (self.MAY, "1C", 7)]
        out = merge_supplement_rows(rows, set())
        assert sum(out[self.MAY].values()) == 107
        assert len(out[self.MAY]) == 1

    def test_blanks_and_file_skills_skipped(self):
        rows = [(self.MAY, "", 5), (self.MAY, "   ", 3), (self.MAY, "python", 10)]
        out = merge_supplement_rows(rows, {"python"})
        assert out == {}

    def test_residual_junk_dropped(self):
        rows = [(self.MAY, "paбoтaть b komanдe", 45)]
        assert merge_supplement_rows(rows, set()) == {}

    def test_legit_mixed_kept(self):
        rows = [(self.MAY, "a/b тестирование", 90), (self.MAY, "c/c++", 160)]
        out = merge_supplement_rows(rows, set())
        assert out[self.MAY] == {"a/b тестирование": 90, "c/c++": 160}


class TestSupplementBlacklist:
    MAY = date(2026, 5, 1)
    BLOCKED = ["mikrotik", "usergate", "vlan", "=nat", "=ips", "oc windows"]

    def test_blocked_dropped(self):
        rows = [
            (self.MAY, "mikrotik", 130),
            (self.MAY, "MikroTik RouterOS", 5),
            (self.MAY, "nat", 174),
            (self.MAY, "ips", 144),
            (self.MAY, "usergate", 80),
            (self.MAY, "vlan", 285),
            (self.MAY, "oc windows", 158),
            (self.MAY, "c++", 425),
        ]
        out = merge_supplement_rows(rows, set(), blocked=self.BLOCKED)
        assert out[self.MAY] == {"c++": 425}

    def test_blocked_exact_prefix_spares_legit(self):
        rows = [(self.MAY, "react native", 22), (self.MAY, "ipsec", 76)]
        out = merge_supplement_rows(rows, set(), blocked=self.BLOCKED)
        assert out[self.MAY] == {"react native": 22, "ipsec": 76}

    def test_no_blocked_param_keeps_old_behavior(self):
        rows = [(self.MAY, "mikrotik", 130)]
        assert merge_supplement_rows(rows, set()) == {self.MAY: {"mikrotik": 130}}


class TestDropBrokenSnapshots:
    def _snap(self, month, n):
        return (date(2026, month, 1), {f"s{i}": float(i) for i in range(n)})

    def test_drops_partial_collections(self):
        snaps = [self._snap(5, 882), self._snap(6, 1153), self._snap(7, 146),
                 self._snap(8, 6), self._snap(9, 495)]
        kept = drop_broken_snapshots(snaps)
        assert [d.month for d, _ in kept] == [5, 6, 9]

    def test_needs_three_snapshots(self):
        snaps = [self._snap(5, 10), self._snap(6, 1)]
        assert drop_broken_snapshots(snaps) == snaps

    def test_all_broken_returns_input(self):
        snaps = [self._snap(5, 0), self._snap(6, 0), self._snap(7, 0)]
        assert drop_broken_snapshots(snaps) == snaps


class TestObservedDrops:
    def _obs(self):
        return {
            "a": [(date(2026, m, 1), float(v)) for m, v in
                  [(4, 100), (5, 90), (6, 80), (7, 70), (8, 60)]],
            "b": [(date(2026, m, 1), float(v)) for m, v in [(8, 50), (9, 60)]],
            "c": [(date(2026, 8, 1), 0.0), (date(2026, 9, 1), 10.0)],
        }

    def test_window_math(self):
        rows = compute_observed_drops(self._obs(), 3)
        by_skill = {r["skill"]: r for r in rows}
        # окно = последние 3 точки: 80 -> 60
        assert by_skill["a"]["observed_change_pct"] == -25.0
        assert by_skill["a"]["points"] == 5

    def test_sufficiency_gates(self):
        assert {r["skill"] for r in compute_observed_drops(self._obs(), 1)} == {"a", "b"}
        assert {r["skill"] for r in compute_observed_drops(self._obs(), 3)} == {"a"}
        assert compute_observed_drops(self._obs(), 6) == []
        assert compute_observed_drops(self._obs(), 12) == []

    def test_zero_first_skipped(self):
        rows = compute_observed_drops(self._obs(), 1)
        assert "c" not in {r["skill"] for r in rows}

    def test_min_freq_and_sort(self):
        obs = {
            "x": [(date(2026, 8, 1), 100.0), (date(2026, 9, 1), 50.0)],
            "y": [(date(2026, 8, 1), 1000.0), (date(2026, 9, 1), 900.0)],
        }
        rows = compute_observed_drops(obs, 1, min_freq=60)
        assert [r["skill"] for r in rows] == ["y"]
        rows = compute_observed_drops(obs, 1)
        assert [r["skill"] for r in rows] == ["x", "y"]  # -50% раньше -10%

    def test_invalid_months(self):
        with pytest.raises(ValueError):
            compute_observed_drops(self._obs(), 5)

    def test_engine_keeps_observed(self):
        from src.predictors.skill_forecast import SkillForecastEngine
        eng = SkillForecastEngine()
        eng.fit({"python": 80})
        assert isinstance(eng._observed, dict)
