"""Hub-gate: unit-проверки логики вето (без сети/БД, таблица подменяется)."""

from src import config
from src.analyzers.comparison import hub_gate as hg


def _patch(monkeypatch, hubs, enabled=True):
    monkeypatch.setattr(hg, "_HUBS", hubs)
    monkeypatch.setattr(config, "HUB_GATE_ENABLED", enabled)
    monkeypatch.setattr(config, "HUB_GATE_HUB_T", 200)
    monkeypatch.setattr(config, "HUB_GATE_MORPH_M", 0.2)
    monkeypatch.setattr(config, "HUB_GATE_ULTRA_HUB", 1000)
    monkeypatch.setattr(config, "HUB_GATE_SIM_LO", 0.6)
    monkeypatch.setattr(config, "HUB_GATE_SIM_HI", 0.75)


def test_veto_kills_hub_junk(monkeypatch):
    _patch(monkeypatch, {"vlan": 1292})
    assert hg.veto_match("pytest", "vlan", 0.6461) is True


def test_veto_spares_low_hub(monkeypatch):
    _patch(monkeypatch, {"niche-skill": 5})
    assert hg.veto_match("pytest", "niche-skill", 0.65) is False


def test_veto_spares_morph_supported(monkeypatch):
    _patch(monkeypatch, {"анализ данных": 252})
    assert hg.veto_match("анализ данных", "анализ данных", 0.7) is False


def test_veto_outside_zone_never(monkeypatch):
    _patch(monkeypatch, {"vlan": 150})
    assert hg.veto_match("pytest", "vlan", 0.95) is False
    assert hg.veto_match("pytest", "vlan", 0.5) is False


def test_ultra_veto_high_sim_hub(monkeypatch):
    _patch(monkeypatch, {"zustand": 1489})
    assert hg.veto_match("управление временем", "zustand", 0.82) is True


def test_ultra_spares_morph_supported(monkeypatch):
    _patch(monkeypatch, {"тестирование по": 1200})
    assert hg.veto_match("тестирование", "тестирование по", 0.9) is False


def test_filter_disabled_keeps_all(monkeypatch):
    _patch(monkeypatch, {"vlan": 1292}, enabled=False)
    ms = [{"skill": "vlan", "similarity": 0.64}]
    kept, vetoed = hg.filter_matches(["pytest"], ms, comp=None)
    assert kept == ms and vetoed == []
