"""Demand coverage must survive degenerate denominators.

market_top_coverage compares a discipline against the market skills relevant to
it. Two failure modes showed up on real data: an empty relevant set (metric
silently 0%) and a tiny set (2 of 3 reads as 67%). These tests pin the guards.

The relevance filter only runs when the discipline has a real embedding profile
(`has_full_profile`); otherwise relevance is noise and the denominator would be
garbage, so the global top-N is used.
"""

from __future__ import annotations

import pytest

from src.analyzers.coverage_analyzer import (
    DISCIPLINE_RELEVANCE_THRESHOLD,
    MARKET_TOP_N,
    CoverageAnalyzer,
)
from src.analyzers.skill_matcher import SkillMatcher


class StubScorer:
    """Relevance stub: scores by an explicit table, 0.0 otherwise."""

    def __init__(self, scores: dict[str, float], has_profile: bool = True):
        self.scores = scores
        self._has_profile = has_profile

    def has_full_profile(self, discipline_name: str) -> bool:
        return self._has_profile

    def compute_relevance_batch(self, skills, discipline_name=None):
        return {s: self.scores.get(s, 0.0) for s in skills}


def build(
    market: dict[str, int],
    scores: dict[str, float],
    weights=None,
    has_profile: bool = True,
) -> CoverageAnalyzer:
    matcher = SkillMatcher(market, market_weights=weights)
    matcher.set_market(market, weights=weights)
    return CoverageAnalyzer(
        matcher, discipline_scorer=StubScorer(scores, has_profile=has_profile)
    )


MARKET = {f"skill{i:03d}": 1000 - i for i in range(MARKET_TOP_N)}
RPD = {"PK-1": ["skill000", "skill001"]}


def test_relevant_subset_is_the_denominator():
    scores = {f"skill{i:03d}": 0.9 for i in range(10)}
    cov = build(MARKET, scores).analyze_discipline("d", "disc", RPD).ok()
    assert cov.market_top_coverage == pytest.approx(0.2)  # 2 of 10 relevant


def test_empty_relevance_falls_back_to_global_top():
    # No skill is relevant -> the filter is skipped, not applied to an empty set.
    cov = build(MARKET, {}).analyze_discipline("d", "disc", RPD).ok()
    assert cov.market_top_coverage == pytest.approx(2 / MARKET_TOP_N, rel=1e-3)


def test_threshold_is_respected():
    # Exactly at the cutoff counts as relevant; just below does not.
    names = [f"skill{i:03d}" for i in range(10)]
    at = {s: DISCIPLINE_RELEVANCE_THRESHOLD for s in names}
    below = {s: DISCIPLINE_RELEVANCE_THRESHOLD - 0.01 for s in names}
    assert build(MARKET, at).analyze_discipline("d", "disc", RPD).ok().market_top_coverage == pytest.approx(0.2)
    assert build(MARKET, below).analyze_discipline("d", "disc", RPD).ok().market_top_coverage == pytest.approx(
        2 / MARKET_TOP_N, rel=1e-3
    )


def test_zero_weights_fall_back_to_unweighted_ratio():
    scores = {s: 0.9 for s in ("skill000", "skill001")}
    weights = {s: 0.0 for s in MARKET}
    cov = build(MARKET, scores, weights).analyze_discipline("d", "disc", RPD).ok()
    assert cov.market_weight_coverage == cov.market_top_coverage


def test_weighted_uses_the_same_denominator_as_unweighted():
    scores = {f"skill{i:03d}": 0.9 for i in range(10)}
    weights = {"skill000": 0.5, "skill001": 0.2}
    for i in range(2, 10):
        weights[f"skill{i:03d}"] = 0.3 / 8
    cov = build(MARKET, scores, weights).analyze_discipline("d", "disc", RPD).ok()
    assert cov.market_top_coverage == pytest.approx(0.2)
    assert cov.market_weight_coverage == pytest.approx(0.7)  # 0.7 of 1.0


def test_coverage_is_never_negative_or_above_one():
    scores = {s: 0.9 for s in MARKET}
    cov = build(MARKET, scores).analyze_discipline("d", "disc", RPD).ok()
    assert 0.0 <= cov.market_top_coverage <= 1.0
    assert 0.0 <= cov.market_weight_coverage <= 1.0
