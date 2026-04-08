import pytest

def test_gap_analyzer_init_and_dynamic_thresholds(gap_analyzer):
    assert gap_analyzer.total_weight > 0
    assert gap_analyzer.HIGH_IMPORTANCE > gap_analyzer.MEDIUM_IMPORTANCE


def test_analyze_gap_returns_categories(gap_analyzer):
    result = gap_analyzer.analyze_gap(["Python", "SQL"])
    assert "high_priority" in result
    assert "medium_priority" in result
    assert "low_priority" in result
    assert result["total_gaps"] > 0


def test_coverage_calculates_correctly(gap_analyzer):
    coverage, details = gap_analyzer.coverage(["Python", "SQL", "Git"])
    assert 0 <= coverage <= 100
    assert details["covered_weight"] > 0
    assert details["coverage_percent"] == coverage


def test_top_market_skills(gap_analyzer):
    top = gap_analyzer.top_market_skills(top_n=5)
    assert len(top) == 5
    assert top[0]["rank"] == 1
    assert "skill" in top[0]


def test_get_recommendations(gap_analyzer):
    gaps = gap_analyzer.analyze_gap(["Python"])
    recs = gap_analyzer.get_recommendations(["Python"], gaps)
    assert len(recs) > 0
    assert any("Приоритет" in r for r in recs)