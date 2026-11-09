"""Дрейф распределения навыков: PSI и вердикт."""
from __future__ import annotations

import math

from src.analyzers.skills.drift import (
    detect_drift,
    population_stability_index,
)


def test_psi_zero_for_identical_distributions():
    dist = {"python": 100, "sql": 50, "docker": 25}
    assert population_stability_index(dist, dist) == 0.0


def test_psi_ignores_meta_keys():
    a = {"_meta": {"snapshot_date": "2026-01-01"}, "python": 100, "sql": 100}
    b = {"_meta": {"snapshot_date": "2026-02-01"}, "python": 100, "sql": 100}
    assert population_stability_index(a, b) == 0.0


def test_psi_detects_large_shift():
    expected = {"python": 100, "sql": 100}
    actual = {"python": 900, "sql": 100}
    psi = population_stability_index(expected, actual)
    assert psi > 0.25


def test_detect_drift_levels():
    same = {"a": 10, "b": 10}
    assert detect_drift(same, same)["level"] == "none"

    # умеренный сдвиг
    moderate = detect_drift({"a": 100, "b": 100}, {"a": 130, "b": 100}, psi_threshold=0.25)
    assert moderate["drifted"] is False
    assert moderate["level"] in {"none", "moderate"}

    # значимый сдвиг
    strong = detect_drift({"a": 100, "b": 100}, {"a": 900, "b": 100})
    assert strong["drifted"] is True
    assert strong["level"] == "significant"


def test_top_contributors_sorted_desc():
    report = detect_drift({"python": 100, "sql": 100}, {"python": 500, "sql": 90})
    pis = [c["psi"] for c in report["top_contributors"]]
    assert pis == sorted(pis, reverse=True)
    assert len(pis) == 2  # оба навыка учтены (рост и спад)


def test_psi_empty_inputs():
    assert population_stability_index({}, {"a": 1}) == 0.0
    assert population_stability_index({}, {}) == 0.0


def test_psi_symmetric_magnitude_reasonable():
    # пропорционально перевёрнутые распределения дают сопоставимый PSI
    a = {"x": 80, "y": 20}
    b = {"x": 20, "y": 80}
    assert math.isclose(population_stability_index(a, b), population_stability_index(b, a), rel_tol=1e-9)
