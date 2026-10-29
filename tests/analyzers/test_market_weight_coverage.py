"""Demand-weighted market coverage.

market_top_coverage counts every skill equally. market_weight_coverage weights
each top-N market skill by its share of vacancies, so covering a skill that
appears in most postings counts more than covering a niche one. When no
weights are supplied the two must agree exactly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.skill_matcher import SkillMatcher

MARKET = {"python": 1000, "sql": 500, "excel": 200, "cobol": 50}
RPD = {"PK-1": ["python", "excel"]}


def analyzer(weights: dict[str, float] | None) -> CoverageAnalyzer:
    matcher = SkillMatcher(MARKET, market_weights=weights)
    matcher.set_market(MARKET, weights=weights)
    return CoverageAnalyzer(matcher)


def test_without_weights_weighted_equals_top_coverage():
    cov = analyzer(None).analyze_discipline("d1", "disc", RPD).ok()
    assert cov.market_weight_coverage == cov.market_top_coverage
    assert cov.market_top_coverage == pytest.approx(0.5)  # 2 of 4 covered


def test_weights_shift_coverage_toward_demand():
    # python 0.6, excel 0.1 covered; sql 0.25 and cobol 0.05 are gaps.
    weights = {"python": 0.6, "sql": 0.25, "excel": 0.1, "cobol": 0.05}
    cov = analyzer(weights).analyze_discipline("d1", "disc", RPD).ok()
    assert cov.market_top_coverage == pytest.approx(0.5)
    assert cov.market_weight_coverage == pytest.approx(0.7)


def test_all_weights_zero_falls_back_to_zero_not_crash():
    weights = {name: 0.0 for name in MARKET}
    cov = analyzer(weights).analyze_discipline("d1", "disc", RPD).ok()
    assert cov.market_weight_coverage == 0.0


def test_unknown_skills_have_zero_weight():
    weights = {"python": 0.9, "sql": 0.1}  # excel/cobol absent
    cov = analyzer(weights).analyze_discipline("d1", "disc", RPD).ok()
    # Denominator is the weight of all top-N market skills (python + sql = 1.0);
    # only python is covered, so the ratio is 0.9 rather than a division error.
    assert cov.market_weight_coverage == pytest.approx(0.9)


def test_set_market_accepts_weights_and_keeps_backward_compatibility():
    matcher = SkillMatcher(MARKET)
    assert matcher.set_market(MARKET).is_ok()
    assert matcher.market_weights == {}
    assert matcher.set_market(MARKET, weights={"python": 0.5}).is_ok()
    assert matcher.market_weights == {"python": 0.5}


def test_coverage_model_exposes_the_new_field():
    cov = analyzer(None).analyze_discipline("d1", "disc", RPD).ok()
    assert hasattr(cov, "market_weight_coverage")
    assert 0.0 <= cov.market_weight_coverage <= 1.0
