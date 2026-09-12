"""Fringe matches (market freq < MIN_MATCH_FREQ) count as gaps, not coverage."""
from src import Ok
from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.skill_matcher import SkillMatcher, normalize


class _Stub(SkillMatcher):
    def __init__(self):
        super().__init__(market_skills={'sql': 3519, 'редкий-навык': 1})

    def match(self, skill_name):
        n = normalize(skill_name)
        if n == 'sql':
            return Ok(('sql', 'exact', 1.0))
        if n == 'редкий навык':
            return Ok(('редкий-навык', 'semantic', 0.8))
        return Ok((None, 'no_match', 0.0))


def test_fringe_match_is_gap():
    cov = CoverageAnalyzer(_Stub()).analyze_discipline(
        'd1', 'Test', {'K1': ['sql', 'редкий навык', 'вообще мимо']}).unwrap()
    assert cov.market_matched == 1
    assert cov.gaps == 2
    assert abs(cov.coverage_ratio - round(1 / 3, 4)) < 1e-9
