"""Property-based tests for SkillParser (backlog #51).

Four contract properties, checked by Hypothesis over generated text:

1. `_canonicalize` is idempotent: when the first pass yields a canon `c`,
   the second pass over `c` returns the same `c` (whitelist spellings are
   fixed points of the folding used for matching).
2. Parsed skill texts never mix Latin and Cyrillic scripts in adjacent
   letters: every output is a whitelist canon written in a single script.
3. `parse_vacancy` is deterministic: parsing the same Vacancy object twice
   yields the same ordered list of skill texts.
4. Homoglyph variants of one canon word collapse into a single entry
   (dedup key is the folded canon, not the raw spelling).

The parser only reads the shipped whitelist (data/reference/*.json); the
session fixture `guard_reference_data` in tests/conftest.py makes sure no
reference file is modified. Hypothesis settings stay small (max_examples=50,
deadline=None) so the file runs in seconds; style follows tests/test_result.py.
"""

import re

import pytest

# hypothesis - dev dependency (requirements-dev.txt); without it the module
# is skipped instead of breaking the whole run (style of tests/test_result.py).
pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src import Ok
from src.models.vacancy import Area, Employer, KeySkill, Snippet, Vacancy
from src.parsing.skills.skill_parser import SkillParser

# A skill text must not switch script between adjacent letters.
MIXED_SCRIPT_RE = re.compile(r"[A-Za-z][А-Яа-яЁё]|[А-Яа-яЁё][A-Za-z]")

# Latin <-> Cyrillic lookalike pairs used by _normalize_for_matching and
# _HOMOGLYPH_MAP (subset with an exact cyrillic counterpart, both lowercase).
_HOMO_PAIRS = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у",
    "x": "х", "k": "к", "m": "м", "n": "н", "b": "в",
}
_HOMO_PAIRS.update({cyr: lat for lat, cyr in _HOMO_PAIRS.items()})

_texts = st.text(max_size=64)
# KeySkill.__post_init__ rejects blank names, so filter them out up front.
_key_skill_names = st.text(min_size=1, max_size=32).filter(lambda s: s.strip())


def _vacancy(*names: str, description: str = "") -> Vacancy:
    """Minimal Vacancy, same helper shape as tests/parsing/test_homoglyph_regression.py."""
    return Vacancy(
        id="prop-1",
        name="property test",
        area=Area(id=0, name=""),
        employer=Employer(id="0", name=""),
        key_skills=[KeySkill(name=n) for n in names],
        description=description,
        snippet=Snippet(requirement=None, responsibility=None),
    )


def _to_mixed_script(word: str) -> str:
    """Swap every lowercase letter that has a Latin/Cyrillic lookalike."""
    return "".join(_HOMO_PAIRS.get(ch, ch) for ch in word)


@st.composite
def _mixed_script_canons(draw) -> str:
    """Canon words (len >= 2, lowercase mixable letters) to defold and dedup."""
    parser = SkillParser()
    pool = sorted(
        {
            word
            for word in parser._CANON_MAP.values()
            if len(word) >= 2 and any(ch in _HOMO_PAIRS for ch in word)
        }
    )
    assume(len(pool) > 0)
    return draw(st.sampled_from(pool))


class TestCanonicalizeIdempotence:
    @settings(max_examples=50, deadline=None)
    @given(_texts)
    def test_second_pass_returns_same_canon(self, text):
        parser = SkillParser()
        canon = parser._canonicalize(text)
        if canon is not None:
            assert parser._canonicalize(canon) == canon


class TestSingleScriptOutput:
    @settings(max_examples=50, deadline=None)
    @given(description=_texts, names=st.lists(_key_skill_names, max_size=3))
    def test_parsed_texts_have_no_mixed_script_pairs(self, description, names):
        result = SkillParser().parse_vacancy(_vacancy(*names, description=description))
        assert isinstance(result, Ok)
        for skill in result.unwrap():
            assert not MIXED_SCRIPT_RE.search(skill.text), repr(skill.text)


class TestOutputDeterminism:
    @settings(max_examples=50, deadline=None)
    @given(description=_texts, names=st.lists(_key_skill_names, max_size=3))
    def test_same_vacancy_parsed_twice_yields_same_texts(self, description, names):
        parser = SkillParser()
        vacancy = _vacancy(*names, description=description)
        first = parser.parse_vacancy(vacancy)
        second = parser.parse_vacancy(vacancy)
        assert isinstance(first, Ok)
        assert isinstance(second, Ok)
        assert [s.text for s in first.unwrap()] == [s.text for s in second.unwrap()]


class TestHomoglyphDedupStability:
    @settings(max_examples=50, deadline=None)
    @given(_mixed_script_canons())
    def test_original_and_mixed_variant_collapse_to_single_canon(self, canon):
        parser = SkillParser()
        expected = parser._canonicalize(canon)
        assume(expected is not None)
        mixed = _to_mixed_script(canon)
        assume(mixed != canon)
        assert parser._canonicalize(mixed) == expected
        result = parser.parse_vacancy(_vacancy(canon, mixed))
        assert isinstance(result, Ok)
        assert [s.text for s in result.unwrap()] == [expected]
