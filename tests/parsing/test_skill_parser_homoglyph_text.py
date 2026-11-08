"""Регресс на хомоглифы: парсер должен хранить ОРИГИНАЛ навыка, не свёртку.

Раньше _marker_search/_regex_search сохраняли в ExtractedSkill.text результат
_normalize_for_matching (кириллица, свёрнутая в латиницу). Из-за этого чистый
русский текст («работа в команде») попадал в parsed_skills как «paбoтa b komanдe».
Свёртка нужна только для сопоставления; в text должно быть точное написание.
"""

from __future__ import annotations

import re

from src.parsing.skills.skill_parser import SkillParser, SkillSource

# Смешанная латиница/кириллица внутри одного слова: латинская буква рядом с
# кириллической. Именно так выглядит свёрнутый хомоглиф.
MIXED = re.compile(r"[A-Za-z][А-Яа-яЁё]|[А-Яа-яЁё][A-Za-z]")


def _texts(result) -> list[str]:
    return [s.text for s in result.unwrap()]


def test_marker_search_stores_clean_cyrillic():
    parser = SkillParser()
    res = parser._marker_search(
        "Требования:\nработа в команде, системное мышление", SkillSource.DESCRIPTION
    )
    texts = _texts(res)
    assert "работа в команде" in texts, texts
    assert not any(MIXED.search(t) for t in texts), texts


def test_marker_search_has_no_folded_latin_duplicate():
    parser = SkillParser()
    res = parser._marker_search(
        "Требования:\nработа в команде", SkillSource.DESCRIPTION
    )
    texts = _texts(res)
    assert "paбoтa b komanдe" not in texts, texts
    assert "работа в команде" in texts, texts


def test_regex_search_stores_original():
    parser = SkillParser()
    res = parser._regex_search(
        "Знание 1c программирование", SkillSource.DESCRIPTION
    )
    texts = _texts(res)
    assert any(t.lower().startswith("1c") for t in texts), texts
    assert not any(MIXED.search(t) for t in texts), texts


def test_extract_from_text_has_no_mixed_tokens():
    parser = SkillParser()
    res = parser._extract_from_text(
        "Требования:\n1c программирование, работа в команде, системное мышление, python",
        SkillSource.DESCRIPTION,
    )
    texts = _texts(res)
    assert not any(MIXED.search(t) for t in texts), texts


def test_1c_skill_preserved_and_clean():
    parser = SkillParser()
    res = parser._marker_search(
        "Требования:\n1c программирование", SkillSource.DESCRIPTION
    )
    texts = _texts(res)
    # 1c в любом виде - не мусор, должен сохраниться и быть без свёртки.
    assert any(t.lower().startswith("1c") for t in texts), texts
    assert not any(MIXED.search(t) for t in texts), texts
