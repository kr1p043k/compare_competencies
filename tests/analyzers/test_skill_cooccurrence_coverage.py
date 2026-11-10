"""Coverage for src/analyzers/skill_cooccurrence.py - had no dedicated test file.

The module is pure in-memory statistics: no I/O to mock, all behaviour is
asserted on exact probabilities and orderings.
"""
from __future__ import annotations

import json

from src.analyzers.skill_cooccurrence import SkillCooccurrence


def _built_with_vocab() -> SkillCooccurrence:
    """python appears in 3 vacancies; co-occurs with sql in 2, docker in 1.

    'Java' is filtered out by the vocab, 'r' survives it, blanks/dups collapse.
    """
    return SkillCooccurrence().build(
        [
            ["Python", "SQL"],
            ["python", "docker"],
            ["Python", " sql "],
            ["Java"],
            ["r"],
        ],
        vocab=["python", "sql", "docker", "r"],
    )


def test_build_counts_frequencies_and_applies_vocab_filter():
    sc = _built_with_vocab()

    assert sc.freq == {"python": 3, "sql": 2, "docker": 1, "r": 1}
    assert "java" not in sc.freq  # outside vocab -> dropped despite occurrence
    assert sc.vocab == {"python", "sql", "docker", "r"}


def test_cond_returns_share_of_co_occurring_vacancies():
    sc = _built_with_vocab()

    assert sc.cond("sql", "python") == 2 / 3
    assert sc.cond("docker", "PYTHON") == 1 / 3  # lookup is case-insensitive


def test_cond_returns_zero_for_unknown_or_self_reference():
    sc = _built_with_vocab()

    assert sc.cond("python", "python") == 0.0  # skill never predicts itself
    assert sc.cond("sql", "nosuchskill") == 0.0
    assert sc.cond("", "python") == 0.0


def test_link_takes_max_probability_over_reference_skills():
    sc = _built_with_vocab()

    assert sc.link("python", ["sql", "docker", "never-heard-of"]) == 2 / 3
    assert sc.link("python", []) == 0.0


def test_top_partners_returns_most_common_first_and_respects_limit():
    sc = _built_with_vocab()

    assert sc.top_partners("python") == ["sql", "docker"]  # count desc
    assert sc.top_partners("python", n=1) == ["sql"]
    assert sc.top_partners("unknown-skill") == []


def test_build_keeps_single_letter_r_but_drops_other_one_letter_skills():
    sc = SkillCooccurrence().build([["x", "r", ""], None])

    assert sc.freq == {"r": 1}  # 'x' (1-char, not r/c) and '' are dropped
    assert sc.top == {}  # single-skill vacancy -> no pairs recorded
    assert sc.vocab is None


def test_single_letter_c_counts_toward_cooccurrence():
    sc = SkillCooccurrence().build([["c", "bash"], ["c"]])

    assert sc.freq == {"c": 2, "bash": 1}
    assert sc.cond("bash", "c") == 0.5


def test_build_with_none_input_returns_blank_state():
    sc = SkillCooccurrence().build(None)

    assert sc.freq == {}
    assert sc.top == {}
    assert sc.vocab is None
    assert sc.cond("python", "sql") == 0.0


def test_cache_roundtrip_survives_json_serialization():
    payload = _built_with_vocab().to_cache()

    assert payload["vocab"] == ["docker", "python", "r", "sql"]  # sorted for a stable cache
    restored = SkillCooccurrence.from_cache(json.loads(json.dumps(payload)))

    assert restored.freq == {"python": 3, "sql": 2, "docker": 1, "r": 1}
    assert restored.top == {
        "python": {"sql": 2, "docker": 1},
        "sql": {"python": 2},
        "docker": {"python": 1},
    }
    assert restored.vocab == {"python", "sql", "docker", "r"}


def test_from_cache_with_empty_payload_returns_blank_state():
    obj = SkillCooccurrence.from_cache({})

    assert obj.freq == {}
    assert obj.top == {}
    assert obj.vocab is None
