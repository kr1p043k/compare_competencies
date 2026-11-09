"""Регресс: односимвольные языки `c` и `r` извлекаются из описания.

Без whitelist-гейта `_SINGLE_CHAR_ALLOW` в `_marker_search` и записи `c` в
it_skills.json односимвольные языки терялись, при этом `c` не должен
цепляться внутри более длинных слов (`cd`, `cat`, `campus`) и кириллических
слов-гомоглифов (`связь`).
"""
from __future__ import annotations

from src import Ok
from src.models.vacancy import Area, Employer, Snippet, Vacancy
from src.parsing.skills.skill_parser import SkillParser


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


def test_c_extracted_from_cyrillic_c_description():
    """«Знание языка С» -> канон `c` (кириллический С - омоглиф латинского)."""
    texts = _texts("Знание языка С и опыт")
    assert "c" in texts, texts


def test_c_not_matched_inside_longer_words():
    """`c` извлекается только как отдельное слово, не внутри более длинных."""
    for word in ("cd", "cat", "связь", "campus"):
        texts = _texts(f"Требования: {word}")
        assert "c" not in texts, (word, texts)