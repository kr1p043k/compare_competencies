# tests/parsing/test_skill_parser_markers.py
"""Regression: `_marker_search` извлекает навыки из полного списка после маркера.

C2 fix покрывает:
- регион после КАЖДОГО вхождения маркера (был только parts[1][:600]);
- разделители `·` (U+00B7) наравне с `•`, запятой, `-`, `*`, `;`, переносом;
- длина токена >= 2 (была > 3: терялись `sql`, `git`, `go`);
- маркер «навыки» (были только «ключевые навыки» и т.п.).
"""

from src.parsing.skills.skill_parser import SkillParser, SkillSource


def _skill_set(parser: SkillParser, text: str) -> set[str]:
    result = parser._marker_search(text, SkillSource.DESCRIPTION)
    assert result.is_ok()
    return {s.text.lower() for s in result.unwrap()}


class TestMarkerSearchFixed:
    def test_marker_na_vyki_single(self):
        assert _skill_set(SkillParser(), "Навыки: Python") == {"python"}

    def test_marker_na_vyki_comma(self):
        assert _skill_set(SkillParser(), "Навыки: Python, SQL") == {"python", "sql"}

    def test_bare_bullet_list_handled_by_direct_search(self):
        # Без маркера _marker_search пуст (контракт test_marker_search_no_marker);
        # список с буллетами вытаскивает _direct_search.
        text = "Python\n- SQL\n- Git\n- Docker"
        assert _skill_set(SkillParser(), text) == set()
        result = SkillParser()._direct_search(text, SkillSource.DESCRIPTION)
        assert result.is_ok()
        assert {s.text.lower() for s in result.unwrap()} == {"python", "sql", "git", "docker"}

    def test_marker_multiline_sql_git_not_lost(self):
        assert _skill_set(SkillParser(), "Требования:\nPython\nSQL\nGit") == {
            "python",
            "sql",
            "git",
        }

    def test_repeated_marker_both_segments(self):
        # «Только первый сегмент» исправлено: оба вхождения маркера обрабатываются.
        assert _skill_set(SkillParser(), "Требования: Python. Требования: SQL") == {
            "python",
            "sql",
        }

    def test_negation_context(self):
        assert _skill_set(SkillParser(), "Требования: Python не требуется, SQL") == {"sql"}

    def test_location_word_does_not_leak(self):
        assert _skill_set(SkillParser(), "Требования: Москва, Python") == {"python"}

    def test_pure_noise(self):
        assert (
            _skill_set(SkillParser(), "Требования: дружный коллектив и приятный офис")
            == set()
        )

    def test_middle_dot_bullets(self):
        text = "Требования:\n· Python\n· SQL\n· Git\n· Docker"
        assert _skill_set(SkillParser(), text) == {"python", "sql", "git", "docker"}
