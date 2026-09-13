"""Unit (v35): doc/QA mapped entries (grounded, freq-verified 13.09.2026)."""
from src.analyzers.action_map import phrase_lemmas, resolve_action_tools

MARKET = {'техническая документация': 63, 'гост': 63, 'python': 3016}


def test_otchetnaya_maps_techdocs():
    hit = resolve_action_tools(phrase_lemmas('отчетная документация'), MARKET)
    assert hit == ('техническая документация', 'grounded')


def test_standards_maps_gost_or_techdocs():
    hit = resolve_action_tools(
        phrase_lemmas('применять стандарты, нормы и правила при оформлении документации'),
        MARKET)
    assert hit is not None and hit[1] == 'grounded'
    assert hit[0] in ('гост', 'техническая документация')


def test_dormant_without_market():
    hit = resolve_action_tools(phrase_lemmas('отчетная документация'), {'python': 1})
    assert hit is None
