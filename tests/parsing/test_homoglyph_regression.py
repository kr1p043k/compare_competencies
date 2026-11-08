"""Контракт parsed_skills: хранятся ТОЛЬКО каноны белого списка.

Раньше dedup делал `s.text = norm` (свёртка в латиницу) и в БД попадала
смешанная каша ("bыcтpaиbaть"). Затем кратко хранили первый оригинал. Теперь
действует утверждённое правило: в parsed_skills попадает канон белого списка
в точном написании (кириллица для русских канонов, латиница для акронимов),
не-каноны отбрасываются.
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


def test_homoglyph_variants_collapse_to_single_canon():
    parser = SkillParser()
    result = parser.parse_vacancy(_vac("вебхуки", "beбxykи"))
    assert isinstance(result, Ok)
    assert [s.text for s in result.unwrap()] == ["вебхуки"]


def test_distinct_canonical_skills_not_collapsed():
    parser = SkillParser()
    result = parser.parse_vacancy(_vac("вебхуки", "базы знаний", "MySQL"))
    assert isinstance(result, Ok)
    # русские каноны - кириллицей, акроним - латиницей
    assert [s.text for s in result.unwrap()] == ["вебхуки", "базы знаний", "mysql"]


def test_non_canonical_key_skills_are_dropped():
    # Новый контракт: key_skills без канона в белом списке не попадают в вывод.
    parser = SkillParser()
    result = parser.parse_vacancy(_vac("аванс", "сервис"))
    assert isinstance(result, Ok)
    assert [s.text for s in result.unwrap()] == []


def test_triple_homoglyph_variants_collapse_to_canon():
    parser = SkillParser()
    result = parser.parse_vacancy(_vac("вебхуки", "beбxykи", "вeбхуки"))
    assert isinstance(result, Ok)
    assert [s.text for s in result.unwrap()] == ["вебхуки"]
