"""Канонизация parsed_skills: русский канон - кириллицей, акроним - латиницей.

Проверяет правило: в parsed_skills попадает только канон белого списка в точном
написании, смешанная латиница/кириллица недопустима.
"""
from __future__ import annotations

import re

from src import Ok
from src.models.vacancy import Area, Employer, Snippet, Vacancy
from src.parsing.skills.skill_parser import SkillParser

MIXED = re.compile(r"[A-Za-z][А-Яа-яЁё]|[А-Яа-яЁё][A-Za-z]")


def _vac(description: str) -> Vacancy:
    return Vacancy(
        id="c1",
        name="v",
        area=Area(id=0, name=""),
        employer=Employer(id="0", name=""),
        key_skills=[],
        description=description,
        snippet=Snippet(requirement=None, responsibility=None),
    )


def _texts(description: str) -> list[str]:
    parser = SkillParser()
    res = parser.parse_vacancy(_vac(description))
    assert isinstance(res, Ok), res
    return [s.text for s in res.unwrap()]


def test_russian_canons_stay_cyrillic():
    texts = _texts("Требования: работа в команде, вебхуки, базы знаний, тестирование")
    for canon in ("работа в команде", "вебхуки", "базы знаний", "тестирование"):
        assert canon in texts, (canon, texts)


def test_latin_acronyms_stay_latin():
    texts = _texts("Требования: CRM, Linux, JSON")
    assert "crm" in texts, texts
    assert "linux" in texts, texts
    assert "json" in texts, texts


def test_mixed_homoglyph_input_canonicalized():
    texts = _texts("Требования: beбxykи, oпepaциonnыe cиcтemы")
    assert "вебхуки" in texts, texts
    assert "операционные системы" in texts, texts


def test_no_mixed_script_tokens_in_output():
    texts = _texts("Требования: paбoтa b komanдe, beбxykи, cиcтeмнoe mышлeниe, linux")
    assert not any(MIXED.search(t) for t in texts), texts


def test_1c_left_untouched_and_clean():
    texts = _texts("Требования: 1c программирование")
    assert any(t.lower().startswith("1c") for t in texts), texts
    assert not any(MIXED.search(t) for t in texts), texts
