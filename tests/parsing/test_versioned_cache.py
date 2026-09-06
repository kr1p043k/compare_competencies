"""Tests for versioned normalize cache"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path.home() / "workvs" / "compare_competencies"))

def test_cache_hit():
    from src.parsing.skills.skill_normalizer import SkillNormalizer
    SkillNormalizer._normalize_cache.clear()
    SkillNormalizer._normalize_cache_version = None
    r1 = SkillNormalizer.normalize("Python 3.9")
    assert r1.is_ok()
    assert len(SkillNormalizer._normalize_cache) > 0
    r2 = SkillNormalizer.normalize("Python 3.9")
    assert r2.unwrap() == r1.unwrap()

def test_cache_invalidates_on_version_change():
    from src.parsing.skills.skill_normalizer import SkillNormalizer
    SkillNormalizer.normalize("Python")
    assert len(SkillNormalizer._normalize_cache) > 0
    # Simulate taxonomy change
    SkillNormalizer._normalize_cache_version = "stale-version"
    SkillNormalizer.normalize("Python")
    # Cache should have been cleared and rebuilt with current version
    assert SkillNormalizer._normalize_cache_version != "stale-version"

def test_taxonomy_version_format():
    from src.parsing.skills.skill_normalizer import SkillNormalizer
    v = SkillNormalizer._taxonomy_version()
    assert isinstance(v, str)
    assert len(v) > 0
