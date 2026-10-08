"""key_skills junk filter: digits/punct/single-char out, R/C languages survive."""

from src.models.vacancy import KeySkill, Vacancy
from src.parsing.skills.skill_parser import SkillParser, SkillSource


def _parse_names(names):
    p = SkillParser()
    v = Vacancy(id="1", name="t", area=None, employer=None,
                key_skills=[KeySkill(name=n) for n in names])
    r = p._extract_from_key_skills(v)
    assert r.is_ok()
    return [s.text for s in r.unwrap()]


class TestKeySkillsJunkFilter:
    def test_real_skills_pass(self):
        got = _parse_names(["Python", "Node.js", "R", "C", "1c"])
        assert got == ["Python", "Node.js", "R", "C", "1c"]

    def test_junk_dropped(self):
        # Пустые имена невозможны: KeySkill валидирует на конструкции.
        assert _parse_names(["61970", "A", "x", "!!!"]) == []

    def test_source_and_confidence(self):
        p = SkillParser()
        v = Vacancy(id="1", name="t", area=None, employer=None,
                    key_skills=[KeySkill(name="Go")])
        (s,) = p._extract_from_key_skills(v).unwrap()
        assert s.source == SkillSource.KEY_SKILLS and s.confidence == 1.0
