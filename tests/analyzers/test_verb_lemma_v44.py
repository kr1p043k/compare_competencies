"""Unit (v44): verb-strip + lemma-fuzzy."""
import os

import pytest

from src.analyzers.skill_matcher import SkillMatcher, strip_lead_verbs
from src.text import ru_morph

MARKET = {
    'анализ данных': 500, 'метод': 100, 'python': 3016, 'документация': 1,
}


def test_strip_leading_verbs():
    assert strip_lead_verbs("знать python") == "python"
    assert strip_lead_verbs("владеть методами анализа") == "методами анализа"
    assert strip_lead_verbs("методами анализа") == "методами анализа"
    assert strip_lead_verbs("знать") == "знать"
    assert strip_lead_verbs("") == ""


def test_lemma_fuzzy_declension():
    if not ru_morph.available():
        pytest.skip("pymorphy3 unavailable")
    import src.feature_flags as _ff
    old_v, old_l = os.environ.get('FF_MATCH_VERBS'), os.environ.get('FF_LEMMA_FUZZY')
    os.environ['FF_MATCH_VERBS'] = '0'
    os.environ['FF_LEMMA_FUZZY'] = '1'
    try:
        m = SkillMatcher(market_skills=dict(MARKET))
        w, ty, s = m.match('методами анализа данных').unwrap()
        assert (w, ty) == ('анализ данных', 'lemma')
        assert abs(s - 0.5) < 1e-9
    finally:
        for k, v in (('FF_MATCH_VERBS', old_v), ('FF_LEMMA_FUZZY', old_l)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_lemma_fringe_still_gap():
    if not ru_morph.available():
        pytest.skip("pymorphy3 unavailable")
    import src.feature_flags as _ff
    old_l = os.environ.get('FF_LEMMA_FUZZY')
    os.environ['FF_LEMMA_FUZZY'] = '1'
    try:
        m = SkillMatcher(market_skills=dict(MARKET))
        assert m.match('документацией владеть').unwrap()[1] in ('no_match', 'semantic')
    finally:
        if old_l is None:
            os.environ.pop('FF_LEMMA_FUZZY', None)
        else:
            os.environ['FF_LEMMA_FUZZY'] = old_l


def test_flags_default_on_after_ablation(monkeypatch):
    monkeypatch.delenv('FF_MATCH_VERBS', raising=False)
    monkeypatch.delenv('FF_LEMMA_FUZZY', raising=False)
    import importlib
    import src.feature_flags as _ff
    mm = importlib.reload(_ff)
    assert mm.market_verbs_enabled() is True
    assert mm.lemma_fuzzy_enabled() is True


def test_mapped_beats_lemma():
    if not ru_morph.available():
        pytest.skip("pymorphy3 unavailable")
    import os as _os
    old_v, old_l = _os.environ.get('FF_MATCH_VERBS'), _os.environ.get('FF_LEMMA_FUZZY')
    _os.environ['FF_MATCH_VERBS'] = '0'
    _os.environ['FF_LEMMA_FUZZY'] = '1'
    try:
        m = SkillMatcher(market_skills={'scikit-learn': 58, 'классификация': 40})
        w, ty, _ = m.match('отбор признаков для классификации').unwrap()
        assert (w, ty) == ('scikit-learn', 'mapped')
    finally:
        for k, v in (('FF_MATCH_VERBS', old_v), ('FF_LEMMA_FUZZY', old_l)):
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


def test_single_char_lemma_dropped():
    if not ru_morph.available():
        pytest.skip("pymorphy3 unavailable")
    import os as _os
    old_l = _os.environ.get('FF_LEMMA_FUZZY')
    _os.environ['FF_LEMMA_FUZZY'] = '1'
    try:
        m = SkillMatcher(market_skills={'с#': 50, 'python': 3016})
        w, ty, _ = m.match('обучение с учителем').unwrap()
        assert (w, ty) != ('с#', 'lemma')
    finally:
        if old_l is None:
            _os.environ.pop('FF_LEMMA_FUZZY', None)
        else:
            _os.environ['FF_LEMMA_FUZZY'] = old_l
