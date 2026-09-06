"""Unit tests for HybridMatcher (new hybrid algorithm for article)."""
import pytest

from src.analyzers.hybrid_matcher import (
    HybridMatcher,
    adaptive_threshold,
    fuzzy_confidence,
    normalize,
    _is_strict_prefix_trap,
    _word_contained,
    SEM_BASE,
    SEM_MIN,
    SEM_MAX,
)


class TestHelpers:
    def test_strict_prefix_trap_java(self):
        assert _is_strict_prefix_trap("java", "javascript") is True

    def test_not_trap_word_boundary(self):
        assert _is_strict_prefix_trap("react", "react native") is False

    def test_not_trap_equal(self):
        assert _is_strict_prefix_trap("python", "python") is False

    def test_not_trap_typo(self):
        assert _is_strict_prefix_trap("reackt", "react") is False

    def test_word_contained(self):
        assert _word_contained("spring", "spring boot") is True
        assert _word_contained("java", "javascript") is False

    def test_adaptive_base(self):
        t = adaptive_threshold("python developer", "python")
        assert t == SEM_BASE

    def test_adaptive_cross_script(self):
        t = adaptive_threshold("базы данных", "postgresql")
        assert t == SEM_BASE - 0.05

    def test_adaptive_long(self):
        t = adaptive_threshold("x" * 50, "python")
        assert t == SEM_BASE - 0.05

    def test_adaptive_short(self):
        t = adaptive_threshold("ab", "abcde")
        assert t == SEM_BASE + 0.10

    def test_adaptive_clamp(self):
        t = adaptive_threshold("ab", "xy")  # short => +0.10
        assert SEM_MIN <= t <= SEM_MAX

    def test_fuzzy_confidence_bounds(self):
        assert fuzzy_confidence(70) == 0.55
        assert fuzzy_confidence(100) == 1.0
        c = fuzzy_confidence(88)
        assert 0.55 < c < 1.0


class TestHybridExact:
    def test_exact_match(self):
        m = HybridMatcher({"python": 100})
        matched, mtype, conf = m.match("Python").unwrap()
        assert matched == "python"
        assert mtype == "exact"
        assert conf == 1.0

    def test_too_short(self):
        m = HybridMatcher({"python": 100})
        matched, _, _ = m.match("ab").unwrap()
        assert matched is None


class TestHybridFuzzy:
    def test_typo_caught(self):
        """reackt -> react: RF high, not a prefix trap."""
        m = HybridMatcher({"react": 100, "python": 50})
        matched, mtype, conf = m.match("reackt").unwrap()
        assert matched == "react"
        assert mtype == "hybrid_fuzzy"
        assert conf >= 0.55

    def test_trap_rejected(self):
        """java -> javascript must NOT match (strict prefix trap)."""
        m = HybridMatcher({"javascript": 100})
        matched, mtype, _ = m.match("java").unwrap()
        assert matched is None
        assert mtype == "no_match"

    def test_word_containment_accepted(self):
        """spring -> spring boot: legitimate containment, accepted."""
        m = HybridMatcher({"spring boot": 100})
        matched, mtype, _ = m.match("spring").unwrap()
        assert matched == "spring boot"
        assert mtype == "hybrid_fuzzy"

    def test_garbage_rejected(self):
        m = HybridMatcher({"python": 100})
        matched, _, _ = m.match("xyzqweasd").unwrap()
        assert matched is None

    def test_match_type_distinct(self):
        """Hybrid uses 'hybrid_fuzzy', not legacy 'fuzzy' label."""
        m = HybridMatcher({"react": 100})
        _, mtype, _ = m.match("reackt").unwrap()
        assert mtype == "hybrid_fuzzy"


class TestHybridCache:
    def test_cache_hit(self):
        m = HybridMatcher({"python": 100})
        r1 = m.match("python").unwrap()
        r2 = m.match("python").unwrap()
        assert r1 == r2

    def test_set_market_clears_cache(self):
        m = HybridMatcher({"python": 100})
        m.match("python")
        assert len(m._match_cache) > 0
        m.set_market({"java": 50})
        assert len(m._match_cache) == 0

    def test_set_market_empty_err(self):
        m = HybridMatcher()
        assert m.set_market({}).is_err()


class TestHybridEmerging:
    def test_emerging_excludes_known(self):
        m = HybridMatcher({"python": 100, "sql": 50})
        res = m.get_emerging({"python"}, top_n=10)
        assert res.is_ok()
        skills = [s for s, _, _ in res.unwrap()]
        assert "python" not in skills
        assert "sql" in skills

    def test_emerging_empty_market(self):
        m = HybridMatcher({})
        assert m.get_emerging(set()).is_err()
