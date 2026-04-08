# tests/analyzers/test_gap_analyzer.py
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
    # Фикс floating-point
    
    assert abs(details["coverage_percent"] - coverage) < 0.01
def test_gap_analyzer_thresholds(gap_analyzer):
    """Проверка динамических порогов (покрывает строки 60-63)"""
    assert hasattr(gap_analyzer, "high_threshold")
    assert hasattr(gap_analyzer, "medium_threshold")
    assert gap_analyzer.high_threshold > gap_analyzer.medium_threshold


def test_gap_analyzer_calculate_gap(gap_analyzer, sample_student):
    """Расчёт gap между студентом и вакансией"""
    vacancy_skills = ["python", "fastapi", "docker"]
    gaps = gap_analyzer._calculate_stats(sample_student, vacancy_skills)
    
    assert isinstance(gaps, dict)
    assert any("python" in k for k in gaps.keys())  # хотя бы один навык студента


def test_gap_analyzer_categorize_gap(gap_analyzer):
    """Категоризация gap (HIGH / MEDIUM / LOW)"""
    assert gap_analyzer._calculate_stats(0.15) == "HIGH"
    assert gap_analyzer._calculate_stats(0.10) == "MEDIUM"
    assert gap_analyzer._calculate_stats(0.05) == "LOW"