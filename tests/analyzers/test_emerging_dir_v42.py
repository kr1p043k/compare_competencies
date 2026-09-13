"""Unit (v42): direction-level emerging capped."""
from src.analyzers.skill_matcher import EMERGING_MAX_FREQ, SkillMatcher

MARKET = {'linux': 3020, 'sql': 3519, 'kubernetes': 1199, 'erp': 1092}


def test_direction_call_pattern_capped():
    m = SkillMatcher(market_skills=dict(MARKET))
    got = m.get_emerging(set(), top_n=15, max_freq=EMERGING_MAX_FREQ).unwrap()
    names = [s for s, _, _ in got]
    assert 'linux' not in names and 'sql' not in names
    assert names[0] == 'kubernetes'
