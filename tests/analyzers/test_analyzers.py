import pytest

def test_gap_analyzer_calculates_deficit(gap_analyzer, sample_student, sample_vacancies):
    result = gap_analyzer.analyze(sample_student, sample_vacancies)
    assert "deficit_score" in result
    assert "missing_skills" in result

def test_embedding_comparator(embedding_comparator):
    sim = embedding_comparator.compare_student_to_market(
        ["Python", "SQL", "ML"],
        ["Python", "Data Science", "Scikit-learn"]
    )
    assert 0 <= sim <= 1