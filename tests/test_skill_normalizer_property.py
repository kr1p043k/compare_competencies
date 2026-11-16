"""Property-based тесты нормализации навыков (BACKLOG 7.5.3, hypothesis).

Проверяем инварианты на любом входе: идемпотентность и нечувствительность
`resolve` (быстрый alias), детерминизм `normalize`, согласованность
`normalize_batch` с одиночным `normalize` и контракт для пунктуации.

Контракт `normalize` шире, чем «значение из whitelist»: он принимает и
passthrough-термины вне whitelist/canonical_map (напр. '00') и умеет сворачивать
вариант ('r0' -> 'r'). Это зафиксировано в отчёте для владельца парсера.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.parsing.skills.skill_normalizer import SkillNormalizer

# Узкий алфавит: латиница, цифры и типовые разделители навыков (c++, c#, .net, t-sql).
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +#._-/"
TEXT = st.text(alphabet=ALPHABET, min_size=1, max_size=20)
SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _known_terms() -> set[str]:
    """Известные канонические термины: whitelist (it_skills) + canonical_map.

    _get_whitelist() - это it_skills.json, а normalize() сверяется ещё и с
    canonical_map; множества не совпадают (напр. 'iso 14001' есть в map), поэтому
    берём объединение ключей и значений.
    """
    canon = SkillNormalizer._get_canonical_map()
    return (
        set(SkillNormalizer._get_whitelist())
        | set(canon.values())
        | set(canon.keys())
    )


@SETTINGS
@given(TEXT)
def test_resolve_idempotent(raw: str):
    once = SkillNormalizer.resolve(raw)
    assert SkillNormalizer.resolve(once) == once


@SETTINGS
@given(TEXT)
def test_resolve_case_insensitive(raw: str):
    assert SkillNormalizer.resolve(raw.upper()) == SkillNormalizer.resolve(raw.lower())


@SETTINGS
@given(TEXT)
def test_resolve_ignores_surrounding_whitespace(raw: str):
    assert SkillNormalizer.resolve(f"  {raw}  ") == SkillNormalizer.resolve(raw)


@SETTINGS
@given(TEXT)
def test_normalize_is_deterministic(raw: str):
    """normalize() детерминирован: один и тот же вход -> один и тот же результат.

    Отдельно проверять «канон из whitelist» нельзя: normalize() принимает и
    passthrough-термины вне whitelist/canonical_map (напр. '00') и может свернуть
    вариант (напр. 'r0' -> 'r'), так что контракт шире whitelist. Эти расхождения
    зафиксированы в отчёте для владельца парсера.
    """
    a = SkillNormalizer.normalize(raw)
    b = SkillNormalizer.normalize(raw)
    assert a.is_ok() == b.is_ok()
    if a.is_ok():
        assert a.unwrap() == b.unwrap()

@pytest.mark.parametrize("raw", ["-", "+", ".", " ", "#", "_", "/"])
def test_punctuation_only_yields_empty_ok(raw: str):
    """Пиним текущий контракт: пунктуация -> Ok(''), а не Err.

    В коде это логируется как too_short_after_normalization. Если решат менять
    на Err - тест должен упасть и заставить осознанно обновить контракт.
    """
    res = SkillNormalizer.normalize(raw)
    assert res.is_ok()
    assert res.unwrap() == ""


@SETTINGS
@given(st.lists(TEXT, max_size=6))
def test_normalize_batch_matches_single(items: list[str]):
    expected = []
    for s in items:
        if not s:
            continue
        r = SkillNormalizer.normalize(s)
        if r.is_ok():
            expected.append(r.unwrap())

    batch = SkillNormalizer.normalize_batch(items)
    assert batch.is_ok()
    assert batch.unwrap() == expected


@SETTINGS
@given(st.sampled_from(sorted(SkillNormalizer._get_whitelist())))
def test_canonical_normalizes_ok(canonical: str):
    res = SkillNormalizer.normalize(canonical)
    assert res.is_ok()
    assert res.unwrap() in _known_terms()
