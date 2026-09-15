"""Unit (v36): fringe fuzzy hits fall through to mapped; exact-fringe still returned."""
from src.analyzers.coverage_analyzer import MIN_MATCH_FREQ
from src.analyzers.skill_matcher import MARKET_MIN_FREQ, SkillMatcher

MARKET = {
    'документация': 1,
    'техническая документация': 63,
    'гост': 63,
    'python': 3016,
}


def test_threshold_single_source():
    assert MIN_MATCH_FREQ == MARKET_MIN_FREQ == 5


def test_fringe_fuzzy_falls_through_to_mapped():
    m = SkillMatcher(market_skills=dict(MARKET))
    w, ty, s = m.match('отчетная документация').unwrap()
    assert (w, ty) == ('техническая документация', 'mapped')
    assert abs(s - 0.85) < 1e-9


def test_exact_fringe_still_returned_for_downstream_gap():
    m = SkillMatcher(market_skills=dict(MARKET))
    assert m.match('документация').unwrap() == ('документация', 'exact', 1.0)


def test_healthy_fuzzy_untouched():
    m = SkillMatcher(market_skills=dict(MARKET))
    w, ty, _ = m.match('работа с python').unwrap()
    assert (w, ty) == ('python', 'fuzzy')
