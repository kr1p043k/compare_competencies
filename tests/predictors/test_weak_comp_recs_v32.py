"""Unit (v32): weak-competency targeted recs, flag-gated."""
import src.predictors.curriculum_recommender as rec_module
from src.models.teacher_analysis import CompetencyCoverage, DisciplineCoverage
from src.predictors.curriculum_recommender import CurriculumRecommender


def _cov():
    return DisciplineCoverage(
        discipline_id="t1",
        discipline_name="test",
        top_matched=[],
        gaps_list=[],
        truly_missing=[],
        cross_references=[],
        competencies=[
            CompetencyCoverage(code="UK-1", total_skills=10, matched_skills=3,
                               coverage=0.3, gap_skills=["a", "b", "c", "d"]),
            CompetencyCoverage(code="UK-2", total_skills=10, matched_skills=0,
                               coverage=0.0, gap_skills=["x"]),
            CompetencyCoverage(code="UK-3", total_skills=10, matched_skills=9,
                               coverage=0.9, gap_skills=["y"]),
            CompetencyCoverage(code="UK-4", total_skills=10, matched_skills=2,
                               coverage=0.2, gap_skills=[]),
        ],
        coverage_ratio=0.3,
    )


def test_explicit_off(monkeypatch):
    monkeypatch.setattr(rec_module, "weak_comp_recs_enabled", lambda: False)
    recs = CurriculumRecommender().generate(_cov(), cooc=None).ok()
    weak = [r for r in recs if r.skill_name == "UK-1" and r.type == "add_new_content"]
    assert weak == []


def test_weak_only(monkeypatch):
    monkeypatch.setattr(rec_module, "weak_comp_recs_enabled", lambda: True)
    recs = CurriculumRecommender().generate(_cov(), cooc=None).ok()
    by_code = {}
    for r in recs:
        if r.type == "add_new_content" and r.skill_name.startswith("UK-"):
            by_code.setdefault(r.skill_name, []).append(r)
    assert set(by_code) == {"UK-1"}  # zero/strong/empty-gaps excluded
    msg = by_code["UK-1"][0].message
    assert "UK-1" in msg and "30%" in msg
    assert "a" in msg and "b" in msg and "c" in msg and "d" not in msg
