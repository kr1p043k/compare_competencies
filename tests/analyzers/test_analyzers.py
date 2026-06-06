# tests/analyzers/test_analyzers.py

from src import Ok


def _unwrap(res):
    match res:
        case Ok(d):
            return d
        case _:
            raise AssertionError(f"Expected Ok, got {res}")


def test_skill_filter_filters_generic(skill_filter):
    """SkillFilter удаляет generic слова"""
    weights = {"frontend": 0.9, "python": 0.8, "web разработка": 0.7, "sql": 0.6}
    filtered = _unwrap(skill_filter.filter_weights(weights, min_weight=0.01))
    assert "python" in filtered
    assert "frontend" not in filtered


def test_skill_level_analyzer_analyzes_vacancies(skill_level_analyzer):
    """SkillLevelAnalyzer корректно анализирует вакансии"""
    vacancies = [
        {"skills": ["python", "sql"], "experience": "junior"},
        {"skills": ["python", "docker"], "experience": "middle"},
        {"skills": ["docker", "k8s"], "experience": "senior"},
    ]
    skill_level_analyzer.analyze_vacancies(vacancies)
    assert "python" in skill_level_analyzer.skill_by_level
    assert skill_level_analyzer.skill_by_level["python"]["junior"] == 1
    assert skill_level_analyzer.skill_by_level["python"]["middle"] == 1

    level = skill_level_analyzer.get_skill_level("python")
    assert level in ("junior", "middle", "all_levels")


def test_embedding_comparator_initialization(embedding_comparator):
    """Проверяет инициализацию EmbeddingComparator из фикстуры"""
    assert embedding_comparator is not None
    assert embedding_comparator.model is not None


def test_embedding_comparator_build_index(embedding_comparator):
    """Метод build_market_index работает корректно"""
    all_market_skills = [
        "python",
        "javascript",
        "react",
        "node.js",
        "next.js",
        "fastapi",
        "docker",
        "postgresql",
        "machine learning",
        "html",
        "mlops",
    ]
    from unittest.mock import patch
    with patch("src.analyzers.comparison.embedding_comparator.joblib.dump", return_value=None):
        embedding_comparator.build_market_index(all_market_skills)

    student_skills = ["python", "react", "fastapi"]
    comparison = _unwrap(embedding_comparator.compare_student_to_market(student_skills))
    assert "matches" in comparison
    assert "missing" in comparison


def test_gap_analyzer_initialization():
    """GapAnalyzer нормально инициализируется с skill_weights_by_level"""
    from src.analyzers.gap.gap_analyzer import GapAnalyzer

    weights_by_level = {
        "junior": {"python": 0.8, "sql": 0.6, "git": 0.4},
        "middle": {"python": 0.9, "docker": 0.7, "sql": 0.5},
        "senior": {"python": 0.9, "docker": 0.9, "k8s": 0.8, "sql": 0.3},
    }
    ga = GapAnalyzer(weights_by_level)

    assert ga.skill_weights_by_level == weights_by_level

    # set_weights_by_level записывает в skill_weights_by_level
    ga.set_weights_by_level(weights_by_level)
    assert ga.skill_weights_by_level == weights_by_level
