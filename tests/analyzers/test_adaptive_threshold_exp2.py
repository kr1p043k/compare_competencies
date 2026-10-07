"""Exp2: парафразный порог semantic stage (163 пары canonical-vs-raw, mpnet).

F1 пик на 0.45 (0.859); recall@0.70 лишь 0.44, @0.55 — 0.69 при prec 0.91.
Base 0.55, clamp [0.45, 0.75].
"""

import pytest

from src.analyzers.skill_matcher import SkillMatcher, adaptive_semantic_threshold


class TestAdaptiveThresholdExp2:
    def test_base_band(self):
        assert adaptive_semantic_threshold("python", "python") == pytest.approx(0.55)

    def test_cross_script_discount(self):
        assert adaptive_semantic_threshold("питон", "python") == pytest.approx(0.50)

    def test_long_query_discount(self):
        t = adaptive_semantic_threshold(
            "very long query string with many words inside here ok", "python")
        assert t == pytest.approx(0.50)

    def test_short_query_premium(self):
        assert adaptive_semantic_threshold("go", "go") == pytest.approx(0.65)

    def test_never_below_floor(self):
        t = adaptive_semantic_threshold(
            "питон очень длинный запрос из множества слов внутри", "python")
        assert t >= 0.45

    def test_cache_stats_api(self):
        m = SkillMatcher({"python": 200})
        st = m.cache_stats()
        assert st["semantic_hits"] == 0 and st["semantic_miss"] == 0
        assert st["semantic_hit_rate"] == 0.0
        assert st["match_cache_size"] == 0
