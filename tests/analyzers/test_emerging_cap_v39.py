"""Unit (v39): emerging giant-cap."""
from src.analyzers.skill_matcher import EMERGING_MAX_FREQ, SkillMatcher

MARKET = {
    'linux': 3020, 'sql': 3519, 'kubernetes': 1199,
    'erp': 1092, 'niche-tool': 45,
}


def test_cap_value():
    assert EMERGING_MAX_FREQ == 1200


def test_giants_capped_mid_kept():
    m = SkillMatcher(market_skills=dict(MARKET))
    got = m.get_emerging(set(), top_n=10, max_freq=EMERGING_MAX_FREQ).unwrap()
    names = [s for s, _, _ in got]
    assert 'linux' not in names and 'sql' not in names
    assert names == ['kubernetes', 'erp', 'niche-tool']


def test_no_cap_default_unchanged():
    m = SkillMatcher(market_skills=dict(MARKET))
    names = [s for s, _, _ in m.get_emerging(set(), top_n=10).unwrap()]
    assert names == ['sql', 'linux', 'kubernetes', 'erp', 'niche-tool']
