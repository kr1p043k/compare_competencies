"""Детект дрейфа распределения навыков между снапшотами частот.

PSI (Population Stability Index) между двумя распределениями частот навыков
(например, ``data/history/freq_market_2026-04.json`` против ``...2026-06.json``).

Пороговые значения общепринятые: PSI < 0.1 - нет дрейфа, 0.1-0.25 - умеренный,
> 0.25 - значимый сдвиг. Числа - ориентир, не догма; порог задаётся вызывающим.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

#: Минимальная доля, чтобы избежать log(0)/деления на ноль.
EPS = 1e-6

#: Поля-метаданные в снапшотах частот, которые не являются навыками.
META_KEYS = frozenset({"_meta"})


def _shares(counts: Mapping[str, float]) -> dict[str, float]:
    """Нормирует счётчики навыков в доли (мета-поля игнорируются)."""
    clean = {k: float(v) for k, v in counts.items() if k not in META_KEYS and not k.startswith("_")}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def population_stability_index(
    expected: Mapping[str, float], actual: Mapping[str, float], eps: float = EPS
) -> float:
    """PSI между двумя распределениями частот (вход - счётчики, не доли)."""
    exp = _shares(expected)
    act = _shares(actual)
    if not exp or not act:
        return 0.0
    total = 0.0
    for key in set(exp) | set(act):
        e = max(exp.get(key, 0.0), eps)
        a = max(act.get(key, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return total


def detect_drift(
    expected: Mapping[str, float],
    actual: Mapping[str, float],
    psi_threshold: float = 0.25,
    top_n: int = 10,
) -> dict:
    """Сводка дрейфа: общий PSI, вердикт и главные по вкладу навыки.

    Возвращает dict: psi, drifted, level (none|moderate|significant),
    top_contributors[{skill, psi, expected_share, actual_share}].
    """
    exp = _shares(expected)
    act = _shares(actual)
    psi = population_stability_index(expected, actual)
    if psi < 0.1:
        level = "none"
    elif psi < psi_threshold:
        level = "moderate"
    else:
        level = "significant"

    contributors = []
    for key in set(exp) | set(act):
        e = max(exp.get(key, 0.0), EPS)
        a = max(act.get(key, 0.0), EPS)
        contributors.append(
            {
                "skill": key,
                "psi": (a - e) * math.log(a / e),
                "expected_share": exp.get(key, 0.0),
                "actual_share": act.get(key, 0.0),
            }
        )
    contributors.sort(key=lambda c: c["psi"], reverse=True)
    return {
        "psi": psi,
        "drifted": psi >= psi_threshold,
        "level": level,
        "top_contributors": contributors[: max(0, top_n)],
    }
