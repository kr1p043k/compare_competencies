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


def test_action_map_grounded_resolution():
    from src.analyzers.action_map import phrase_lemmas, resolve_action_tools
    market = {'pandas': 271, 'scikit-learn': 58, 'numpy': 191, 'python': 3016}
    hit = resolve_action_tools(phrase_lemmas('Нормализовать и масштабировать числовые признаки'), market)
    assert hit is not None and hit[1] == 'grounded'
    assert hit[0] in ('pandas', 'scikit-learn', 'numpy')


def test_action_map_expert_dormant_then_active():
    from src.analyzers.action_map import phrase_lemmas, resolve_action_tools
    lemmas = phrase_lemmas('подбор гиперпараметров модели')
    assert lemmas, 'lemma backend unavailable'
    market = {'scikit-learn': 58}
    assert resolve_action_tools(lemmas, market) == ('scikit-learn', 'grounded')
    market2 = {'optuna': 5000}
    hit = resolve_action_tools(lemmas, market2)
    assert hit == ('optuna', 'expert')


def test_mapped_stage_in_matcher():
    from src.analyzers.skill_matcher import SkillMatcher
    m = SkillMatcher(market_skills={'pandas': 271, 'scikit-learn': 58})
    w, ty, s = m.match('Нормализовать пропуски и выбросы').unwrap()
    assert (w, ty) == ('pandas', 'mapped')
    assert abs(s - 0.85) < 1e-9
