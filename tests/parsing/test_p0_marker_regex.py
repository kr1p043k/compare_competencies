"""P0+P5: границы слов в маркерах, whitelist-гейт regex, negation-контекст."""

from src.parsing.skills.skill_parser import SkillParser, SkillSource


def _texts(skills):
    return [s.text.lower() for s in skills]


class TestMarkerWordBoundary:
    def test_go_inside_django_not_matched(self):
        p = SkillParser()
        r = p._marker_search(
            "Требования: django, опыт работы с фреймворком", SkillSource.DESCRIPTION)
        assert r.is_ok()
        assert "go" not in _texts(r.unwrap())

    def test_single_char_skills_skipped_in_marker(self):
        p = SkillParser()
        r = p._marker_search("Требования: r, python", SkillSource.DESCRIPTION)
        assert r.is_ok()
        texts = _texts(r.unwrap())
        assert "r" not in texts
        assert "python" in texts

    def test_marker_negation(self):
        p = SkillParser()
        r = p._marker_search("Требования: python не требуется", SkillSource.DESCRIPTION)
        assert r.is_ok()
        assert "python" not in _texts(r.unwrap())


class TestRegexWhitelistGate:
    def test_garbage_token_dropped(self):
        p = SkillParser()
        r = p._regex_search(
            "Требуется опыт работы с молодым дружным коллективом",
            SkillSource.DESCRIPTION)
        assert r.is_ok()
        assert r.unwrap() == []

    def test_real_skill_passes_gate(self):
        p = SkillParser()
        r = p._regex_search("Требуется знание Python и SQL", SkillSource.DESCRIPTION)
        assert r.is_ok()
        texts = _texts(r.unwrap())
        assert "python" in texts and "sql" in texts

    def test_cyrillic_skill_passes_gate(self):
        """Симметрия свёртки: кириллический whitelist-навык проходит гейт."""
        from src.parsing.utils import load_it_skills
        wl = load_it_skills()
        cyr = sorted({s for s in wl if any("а" <= c <= "я" or c == "ё" for c in s.lower())})
        assert cyr, "no cyrillic skills in whitelist"
        probe = cyr[0]
        p = SkillParser()
        r = p._regex_search(f"Требуется знание {probe}", SkillSource.DESCRIPTION)
        assert r.is_ok()
        from src.parsing.skills.skill_normalizer import SkillNormalizer
        norm = SkillNormalizer.normalize(probe).unwrap()
        assert norm in _texts(r.unwrap()), f"{probe!r} -> {norm!r} missing"

    def test_regex_negation(self):
        p = SkillParser()
        r = p._regex_search(
            "Требуется опыт работы с python, знание sql не обязательно",
            SkillSource.DESCRIPTION)
        assert r.is_ok()
        texts = _texts(r.unwrap())
        assert "sql" not in texts
