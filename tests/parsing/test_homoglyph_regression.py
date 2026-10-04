"""Homoglyph regression: dedup key must not overwrite original skill text.

Bug: skill_parser dedup did `s.text = norm` (Cyrillic lookalikes folded to
Latin), persisting mixed-script garbage ("bыcтpaиbaть пpoцeccы") into
vacancies.parsed_skills. Fix: dedup by folded key, keep original text.
"""
from src import Ok
from src.models.vacancy import Area, Employer, KeySkill, Snippet, Vacancy
from src.parsing.skills.skill_parser import SkillParser


def _vac(*names):
    return Vacancy(
        id="test-1",
        name="test vacancy",
        area=Area(id=0, name=""),
        employer=Employer(id="0", name=""),
        key_skills=[KeySkill(name=n) for n in names],
        description="",
        snippet=Snippet(requirement=None, responsibility=None),
    )


def test_homoglyph_dupes_collapse_to_first_original():
    parser = SkillParser()
    result = parser.parse_vacancy(_vac("аванс", "aвaнc"))
    assert isinstance(result, Ok)
    texts = [s.text for s in result.unwrap()]
    assert texts == ["аванс"]


def test_distinct_skills_not_collapsed():
    # Обратная сторона дедупа: разные навыки не должны схлопываться,
    # оригиналы хранятся как есть.
    parser = SkillParser()
    result = parser.parse_vacancy(_vac("сервис", "докер", "Mysql"))
    assert isinstance(result, Ok)
    texts = [s.text for s in result.unwrap()]
    assert texts == ["сервис", "докер", "Mysql"]


def test_triple_homoglyph_variants_collapse_to_first():
    parser = SkillParser()
    result = parser.parse_vacancy(_vac("аванс", "aвaнc", "аванс"))
    assert isinstance(result, Ok)
    assert [s.text for s in result.unwrap()] == ["аванс"]
