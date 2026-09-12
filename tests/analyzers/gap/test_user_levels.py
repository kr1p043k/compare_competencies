"""Profile skill levels must match market keys case-insensitively."""
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.models.student import StudentProfile


def _evaluator():
    weights = {"junior": {"sql": 0.8}, "middle": {"sql": 0.8}, "senior": {"sql": 0.8}}
    return ProfileEvaluator(skill_weights={}, vacancies_skills=[], vacancies_skills_dict=[],
                            use_clustering=False, skill_weights_by_level=weights)


def _gap(evaluator, skills):
    student = StudentProfile(profile_name="t", competencies=[], skills=skills, target_level="middle")
    return evaluator.evaluate_profile(student).unwrap()["skill_metrics"]["sql"]["gap_m"]


def test_uppercase_profile_skill_recognized():
    e = _evaluator()
    assert abs(_gap(e, ["SQL"]) - 0.2) < 1e-9


def test_case_insensitive_equal():
    e = _evaluator()
    assert abs(_gap(e, ["SQL"]) - _gap(e, ["sql"])) < 1e-9
