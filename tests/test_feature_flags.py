"""Unit: feature flags default ON (v32 measured), env opt-out."""
import importlib
import os

import src.feature_flags as ff


def _reload():
    return importlib.reload(ff)


def test_defaults_on(monkeypatch):
    monkeypatch.delenv("FF_MARKET_SYNONYMS", raising=False)
    monkeypatch.delenv("FF_WEAK_COMP_RECS", raising=False)
    m = _reload()
    assert m.market_synonyms_enabled() is True
    assert m.weak_comp_recs_enabled() is True
    assert m.active_flags() == {"FF_MARKET_SYNONYMS": True, "FF_WEAK_COMP_RECS": True}


def test_env_opt_out(monkeypatch):
    monkeypatch.setenv("FF_MARKET_SYNONYMS", "0")
    monkeypatch.setenv("FF_WEAK_COMP_RECS", "no")
    m = _reload()
    assert m.market_synonyms_enabled() is False
    assert m.weak_comp_recs_enabled() is False
    monkeypatch.setenv("FF_WEAK_COMP_RECS", "yes")
    m = _reload()
    assert m.weak_comp_recs_enabled() is True
