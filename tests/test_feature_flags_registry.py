"""Реестр фича-флагов: дефолты, env-оверрайды, полнота активных флагов.

Проверяем без сети/БД: только os.environ через monkeypatch.
"""

from __future__ import annotations

import pytest

import src.feature_flags as ff


def test_registry_is_consistent():
    names = [f.name for f in ff.FLAGS]
    assert len(names) == len(set(names)), "имена флагов должны быть уникальны"
    assert len(ff.FLAGS) >= 4
    for flag in ff.FLAGS:
        assert flag.name.startswith("FF_")
        assert flag.description.strip(), f"у {flag.name} пустое описание"
    assert set(ff._BY_NAME) == set(names)


def test_defaults_when_env_unset(monkeypatch):
    for flag in ff.FLAGS:
        monkeypatch.delenv(flag.name, raising=False)
        assert ff.is_enabled(flag) is flag.default


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_truthy_overrides(monkeypatch, value):
    monkeypatch.setenv(ff.FF_LEMMA_FUZZY, value)
    assert ff.is_enabled(ff.FF_LEMMA_FUZZY) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_falsy_overrides(monkeypatch, value):
    monkeypatch.setenv(ff.FF_LEMMA_FUZZY, value)
    assert ff.is_enabled(ff.FF_LEMMA_FUZZY) is False


def test_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(ff.FF_MARKET_SYNONYMS, "maybe")
    assert ff.is_enabled(ff.FF_MARKET_SYNONYMS) is True  # дефолт этого флага = True


def test_unknown_flag_raises(monkeypatch):
    with pytest.raises(KeyError):
        ff.is_enabled("FF_NOT_A_REAL_FLAG")


def test_legacy_helpers_match_registry(monkeypatch):
    monkeypatch.setenv(ff.FF_MARKET_SYNONYMS, "0")
    monkeypatch.setenv(ff.FF_MATCH_VERBS, "1")
    assert ff.market_synonyms_enabled() is ff.is_enabled(ff.FF_MARKET_SYNONYMS)
    assert ff.market_verbs_enabled() is ff.is_enabled(ff.FF_MATCH_VERBS)
    assert ff.weak_comp_recs_enabled() is ff.is_enabled(ff.FF_WEAK_COMP_RECS)
    assert ff.lemma_fuzzy_enabled() is ff.is_enabled(ff.FF_LEMMA_FUZZY)


def test_active_flags_covers_every_registered_flag(monkeypatch):
    for flag in ff.FLAGS:
        monkeypatch.delenv(flag.name, raising=False)
    snapshot = ff.active_flags()
    assert set(snapshot) == {f.name for f in ff.FLAGS}
    assert all(isinstance(v, bool) for v in snapshot.values())
