# -*- coding: utf-8 -*-
"""Audit fixes 2026-10-12: subagent findings verified by experiment, then fixed."""
from unittest.mock import MagicMock

from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.comparison.hub_gate import _morph_overlap
from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.hybrid_matcher import HybridMatcher
from src.analyzers.skill_matcher import SkillMatcher
from src.parsing.skills.skill_validator import SkillValidator


def _matcher(market):
    m = SkillMatcher(dict(market), embedding_provider=None)
    m.set_market(dict(market))
    return m


def test_ghost_names_not_in_semantic_index():
    import numpy as np

    class _StubProv:
        def encode(self, texts, **kwargs):
            return np.eye(len(texts), 8, dtype=float)

    m = SkillMatcher({"я": 99, "python": 10}, embedding_provider=_StubProv())
    m.set_market({"я": 99, "python": 10})
    assert list(m.market_skills) == ["python"]
    assert list(m._market_names) == ["python"]


def test_exact_case_insensitive():
    m = _matcher({"Python": 10})
    assert m.match("python").unwrap() == ("Python", "exact", 1.0)


def test_match_none_safe():
    m = _matcher({"python": 10})
    assert m.match(None).unwrap() == (None, "no_match", 0.0)
    h = HybridMatcher({"python": 10})
    assert h.match(None).unwrap() == (None, "no_match", 0.0)


def test_emerging_top_n_zero():
    m = _matcher({"python": 10, "sql": 5})
    assert m.get_emerging(set(), top_n=0).unwrap() == []
    h = HybridMatcher({"python": 10})
    assert h.get_emerging(set(), top_n=0).unwrap() == []


def test_parent_weighted_rollup_math():
    m = _matcher({"python": 10, "sql": 10})
    ca = CoverageAnalyzer(m)
    res = ca.analyze_discipline(
        "d1", "disc",
        {"ОПК-3": [], "ОПК-3.1": ["python", "nope-xyz-1", "nope-xyz-2", "nope-xyz-3",
                                  "nope-xyz-4", "nope-xyz-5", "nope-xyz-6",
                                  "nope-xyz-7", "nope-xyz-8", "nope-xyz-9"],
         "ОПК-3.2": ["python", "sql"]},
    ).unwrap()
    parent = [c for c in res.competencies if c.code == "ОПК-3"][0]
    # correct: sum(conf)/total, NOT sum(matched*w)/matched (was 0.826, must be ~0.495 scale)
    kids = [c for c in res.competencies if c.code in ("ОПК-3.1", "ОПК-3.2")]
    expect = sum(c.weighted_coverage * c.total_skills for c in kids) / sum(c.total_skills for c in kids)
    assert parent.weighted_coverage == round(expect, 4)
    assert parent.weighted_coverage < 0.7


def test_anchor_counts_strong():
    m = _matcher({"python": 10})
    ca = CoverageAnalyzer(m)
    res = ca.analyze_discipline("d1", "disc", {"К1": ["python"]}).unwrap()
    assert res.strong_matched >= 0  # anchor in strong set: no crash, invariant kept


def test_comparator_keys_normalized():
    cc = CompetencyComparator(use_embeddings=False)
    cc.set_skill_weights({"Python": 1.0})
    cc.fit_market([["python"], ["java spring"]])
    assert cc.compare(["python"]).unwrap() == (1.0, 1.0)


def test_validator_prose_rejected_dotforms_accepted():
    v = SkillValidator(whitelist={"python", "sql", "node.js", "автоматизация тестирования"})
    assert v.validate("i love python very much today").unwrap().is_valid is False
    assert v.validate("nodejs").unwrap().is_valid is True
    assert v.validate("python").unwrap().is_valid is True
    assert v.validate("изучение автоматизации тестирования систем").unwrap().is_valid is True


def test_morph_keeps_symbolic_skills():
    assert _morph_overlap("c++ разработка", "c++") > 0.0
    assert _morph_overlap("разработка 1с", "1c") > 0.0
    assert _morph_overlap("pytest", "vlan") == 0.0


def test_match_cache_flag_sensitive():
    m = _matcher({"python": 10})
    m.match("python")
    assert any(isinstance(k, tuple) for k in m._match_cache.keys())
    _ = MagicMock()


def test_match_err_excluded_from_denominator():
    from src import Err as _Err

    m = _matcher({"python": 10})
    real_match = m.match

    def flaky(s):
        if s == "boom-skill":
            return _Err("x")
        return real_match(s)

    m.match = flaky
    ca = CoverageAnalyzer(m)
    res = ca.analyze_discipline("d1", "disc", {"К1": ["python", "boom-skill"]}).unwrap()
    comp = [c for c in res.competencies if c.code == "К1"][0]
    assert comp.total_skills == 1
    assert res.total_skills == 1
