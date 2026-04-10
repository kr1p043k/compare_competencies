# tests/analyzers/test_analyzers.py
import pytest

def test_embedding_comparator(embedding_comparator, sample_student):
    """Метод требует предварительной индексации рынка"""
    all_market_skills = [
        "python", "javascript", "react", "node.js", "next.js",
        "fastapi", "docker", "postgresql", "machine learning", "html", "mlops"
    ]
    embedding_comparator.build_market_index(all_market_skills)
   
    student_skills = ["python", "react", "fastapi"]
    comparison = embedding_comparator.compare_student_to_market(student_skills)
   
    assert comparison is not None
    assert isinstance(comparison, dict)
    assert "avg_similarity" in comparison
    assert "matches" in comparison
    assert "missing" in comparison
    # старые атрибуты similarity_score / gap_score больше не используются


def test_gap_analyzer_calculates_deficit(gap_analyzer, sample_student, sample_vacancies):
    result = gap_analyzer.analyze_gap(sample_student.skills)
    assert "high_priority" in result
    assert "medium_priority" in result
    assert "low_priority" in result
    assert result["total_gaps"] >= 0