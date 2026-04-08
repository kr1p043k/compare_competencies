# tests/analyzers/test_gap_analyzer.py
import pytest
from src.analyzers.gap_analyzer import GapAnalyzer

class TestGapAnalyzerExtended:
    def test_init_with_empty_weights(self):
        analyzer = GapAnalyzer({})
        assert analyzer.total_weight == 0
        assert analyzer.HIGH_IMPORTANCE == 0.70
        assert analyzer.MEDIUM_IMPORTANCE == 0.30

    def test_dynamic_thresholds_with_varied_weights(self):
        weights = {"a": 1, "b": 2, "c": 5, "d": 10}
        analyzer = GapAnalyzer(weights)
        # Проверяем, что пороги вычислены как перцентили
        assert 0 < analyzer.HIGH_IMPORTANCE < 1
        assert 0 < analyzer.MEDIUM_IMPORTANCE < 1
        assert analyzer.HIGH_IMPORTANCE > analyzer.MEDIUM_IMPORTANCE

    def test_analyze_gap_empty_student(self):
        weights = {"python": 10, "sql": 5}
        analyzer = GapAnalyzer(weights)
        result = analyzer.analyze_gap([])
        assert result["total_gaps"] == 2
        assert len(result["high_priority"]) + len(result["medium_priority"]) + len(result["low_priority"]) == 2

    def test_analyze_gap_all_covered(self):
        weights = {"python": 10, "sql": 5}
        analyzer = GapAnalyzer(weights)
        result = analyzer.analyze_gap(["python", "sql"])
        assert result["total_gaps"] == 0

    def test_top_market_skills(self):
        weights = {"python": 10, "sql": 8, "docker": 5}
        analyzer = GapAnalyzer(weights)
        top = analyzer.top_market_skills(2)
        assert len(top) == 2
        assert top[0]["skill"] == "python"
        assert top[1]["skill"] == "sql"
        assert "rank" in top[0]
        assert "priority" in top[0]

    def test_coverage_perfect(self):
        weights = {"python": 10, "sql": 5}
        analyzer = GapAnalyzer(weights)
        coverage, details = analyzer.coverage(["python", "sql"])
        assert coverage == 100.0
        assert details["covered_skills_count"] == 2

    def test_get_recommendations_various_coverages(self):
        weights = {"a": 10, "b": 8, "c": 6}
        analyzer = GapAnalyzer(weights)
        gaps = analyzer.analyze_gap([])
        # Низкое покрытие
        recs = analyzer.get_recommendations([], gaps)
        assert "КРИТИЧНО" in recs[0]
        # Среднее покрытие
        recs = analyzer.get_recommendations(["a"], gaps)
        assert "Низкое" in recs[0] or "Среднее" in recs[0]
        # Хорошее покрытие
        recs = analyzer.get_recommendations(["a", "b", "c"], gaps)
        assert "Отличное" in recs[0]

    def test_priority_categories(self):
        weights = {"high": 100, "mid": 50, "low": 10}
        analyzer = GapAnalyzer(weights)
        # Искусственно подправим пороги для теста
        analyzer.HIGH_IMPORTANCE = 0.6
        analyzer.MEDIUM_IMPORTANCE = 0.2
        gaps = analyzer.analyze_gap([])
        assert gaps["high_priority"][0]["skill"] == "high"
        assert gaps["medium_priority"][0]["skill"] == "mid"
        assert gaps["low_priority"][0]["skill"] == "low"
def test_gap_analyzer_init_and_dynamic_thresholds(gap_analyzer):
    assert gap_analyzer.total_weight > 0
    assert gap_analyzer.HIGH_IMPORTANCE > gap_analyzer.MEDIUM_IMPORTANCE

def test_analyze_gap_returns_categories(gap_analyzer):
    result = gap_analyzer.analyze_gap(["Python", "SQL"])
    assert "high_priority" in result
    assert "medium_priority" in result
    assert "low_priority" in result
    assert result["total_gaps"] >= 0

def test_coverage_calculates_correctly(gap_analyzer):
    coverage, details = gap_analyzer.coverage(["Python", "SQL", "Git"])
    assert 0 <= coverage <= 100
    assert details["covered_weight"] > 0
    assert abs(details["coverage_percent"] - coverage) < 0.01