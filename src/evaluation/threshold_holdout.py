"""Оценка порога без leakage (BACKLOG 1.4): выбор на TRAIN, метрики на TEST.

Предыстория: F1~0.99 на L3 считался на том же множестве, на котором sweeping'ом
подбирался порог 0.78 (scripts/experiments/benchmark_matching.py:367, 1115-1124).
In-sample подбор завышает метрику: здесь порог выбирается строго на train-сплите,
а Precision/Recall/F1 считаются на held-out test-сплите (детерминированный split,
фиксированный seed). Дополнительно отдаётся recall при фиксированном precision.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

#: Сетка кандидатов порога (покрывает исторические 0.45-0.85).
DEFAULT_GRID: tuple[float, ...] = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))

#: Целевой precision для recall-at-fixed-precision (требование BACKLOG 1.4).
DEFAULT_PRECISION_TARGET = 0.90

#: Фиксированный seed: сплит воспроизводим между прогонами.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Split:
    """Стратфицированный по label детерминированный сплит пар (score, label)."""

    train: tuple[tuple[float, int], ...]
    test: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class HoldoutReport:
    """Честная сводка: порог выбран на train, метрики посчитаны на test."""

    threshold: float
    train_f1: float
    test_f1: float
    test_precision: float
    test_recall: float
    test_n: int
    #: Лучший recall на test среди порогов сетки с precision >= target.
    recall_at_precision: float
    #: Фактический precision на том пороге, что дал recall_at_precision.
    precision_at_that_threshold: float
    #: Порог сетки, давший recall_at_precision (NaN-безопасно: всегда из сетки).
    recall_at_precision_threshold: float
    precision_target: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": round(self.threshold, 4),
            "train_f1": round(self.train_f1, 4),
            "test_f1": round(self.test_f1, 4),
            "test_precision": round(self.test_precision, 4),
            "test_recall": round(self.test_recall, 4),
            "test_n": self.test_n,
            "recall_at_precision": round(self.recall_at_precision, 4),
            "precision_at_that_threshold": round(self.precision_at_that_threshold, 4),
            "recall_at_precision_threshold": round(self.recall_at_precision_threshold, 4),
            "precision_target": round(self.precision_target, 4),
        }


def _precision_recall_f1(
    pairs: Sequence[tuple[float, int]], threshold: float
) -> tuple[float, float, float]:
    """(precision, recall, f1) бинарного решения score >= threshold."""
    tp = sum(1 for s, y in pairs if y == 1 and s >= threshold)
    fp = sum(1 for s, y in pairs if y == 0 and s >= threshold)
    fn = sum(1 for s, y in pairs if y == 1 and s < threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def split_scores(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    train_frac: float = 0.5,
) -> Split:
    """Детерминированный стратфицированный по label сплит.

    Каждый класс перемешивается random.Random(seed) и делится на train/test;
    при >=2 объектах класса в оба сплита попадает хотя бы по одному.
    """
    if len(scores) != len(labels):
        raise ValueError(f"scores/labels length mismatch: {len(scores)} != {len(labels)}")
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    for y in labels:
        if y not in (0, 1):
            raise ValueError(f"labels must be binary 0/1, got {y!r}")
    if len(scores) < 2:
        raise ValueError("need at least 2 pairs for a train/test split")

    train: list[tuple[float, int]] = []
    test: list[tuple[float, int]] = []
    for cls in (0, 1):
        idx = [i for i, y in enumerate(labels) if y == cls]
        if not idx:
            continue
        rng = random.Random(f"{seed}:{cls}")
        rng.shuffle(idx)
        n_train = max(1, round(len(idx) * train_frac))
        if len(idx) >= 2:
            n_train = min(n_train, len(idx) - 1)
        for pos, i in enumerate(idx):
            (train if pos < n_train else test).append((float(scores[i]), int(labels[i])))
    if not train or not test:
        raise ValueError("degenerate split: one of train/test is empty")
    return Split(train=tuple(train), test=tuple(test))


def select_threshold(
    train: Sequence[tuple[float, int]],
    grid: Sequence[float] = DEFAULT_GRID,
    *,
    min_precision: float | None = None,
) -> float:
    """Максимизировать F1 на TRAIN; tie-break: precision, затем порог выше.

    Если min_precision задан и ни один кандидат не достигает его на train,
    возврат к максимуму F1 (иначе сплит с плохим разделением ломает оценку).
    Только train-данные - test в функцию не заходит физически.
    """
    if not train:
        raise ValueError("cannot select a threshold on an empty train split")
    if not grid:
        raise ValueError("threshold grid must be non-empty")
    scored = [(t, *_precision_recall_f1(train, t)) for t in grid]
    if min_precision is not None:
        ok = [(t, p, r, f) for t, p, r, f in scored if p >= min_precision]
        if ok:
            scored = ok
    best = max(scored, key=lambda z: (z[3], z[1], z[0]))
    return float(best[0])


def recall_at_precision(
    pairs: Sequence[tuple[float, int]],
    target: float = DEFAULT_PRECISION_TARGET,
    grid: Sequence[float] = DEFAULT_GRID,
) -> tuple[float, float, float]:
    """(threshold, recall, precision): макс recall среди порогов с precision >= target.

    Tie-break: более высокий порог (консервативнее). Если target недостижим -
    (grid[0], 0.0, precision(grid[0])) честными нулями, без молчаливого отката.
    """
    if not pairs:
        raise ValueError("recall_at_precision needs non-empty pairs")
    candidates: list[tuple[float, float, float]] = []
    for t in grid:
        p, r, _f = _precision_recall_f1(pairs, t)
        if p >= target and r > 0:
            candidates.append((t, r, p))
    if not candidates:
        first = float(min(grid))
        p, _r, _f = _precision_recall_f1(pairs, first)
        return (first, 0.0, p)
    best = max(candidates, key=lambda z: (z[1], z[0]))
    return best


def evaluate_holdout(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    train_frac: float = 0.5,
    precision_target: float = DEFAULT_PRECISION_TARGET,
    grid: Sequence[float] = DEFAULT_GRID,
) -> HoldoutReport:
    """Полная оценка без leakage: порог на train, метрики на test."""
    split = split_scores(scores, labels, seed=seed, train_frac=train_frac)
    threshold = select_threshold(split.train, grid)
    _tp, _tr, train_f1 = _precision_recall_f1(split.train, threshold)
    p, r, f = _precision_recall_f1(split.test, threshold)
    ra_threshold, ra_recall, ra_precision = recall_at_precision(split.test, precision_target, grid)
    return HoldoutReport(
        threshold=threshold,
        train_f1=train_f1,
        test_f1=f,
        test_precision=p,
        test_recall=r,
        test_n=len(split.test),
        recall_at_precision=ra_recall,
        precision_at_that_threshold=ra_precision,
        recall_at_precision_threshold=ra_threshold,
        precision_target=precision_target,
    )
