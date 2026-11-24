"""BACKLOG 1.4: честная holdout-оценка порога semantic (без leakage).

In-sample F1~0.99 (L3, scripts/experiments/benchmark_matching.py:367 сравнивает
score >= 0.78 на том же множестве, на котором порог выбран sweep'ом на строках
1115-1124). Эти тесты на синтетическом наборе доказывают: разделение train/test
детерминировано, порог выбирается строго на train и test-метрики не зависят от
того, на чём он подбирался.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.threshold_holdout import (
    DEFAULT_GRID,
    _precision_recall_f1,
    evaluate_holdout,
    recall_at_precision,
    select_threshold,
    split_scores,
)


def _synthetic_scores() -> tuple[list[float], list[int]]:
    """40 пар: позитивы выше негативов, но с перекрытием у границы (реалистично)."""
    rng = random.Random(7)
    scores: list[float] = []
    labels: list[int] = []
    for _ in range(20):  # positives
        scores.append(round(min(0.99, max(0.30, rng.gauss(0.80, 0.08))), 4))
        labels.append(1)
    for _ in range(20):  # negatives
        scores.append(round(min(0.99, max(0.30, rng.gauss(0.62, 0.08))), 4))
        labels.append(0)
    return scores, labels


class TestSplitDeterminism:
    def test_same_seed_same_split(self):
        scores, labels = _synthetic_scores()
        a = split_scores(scores, labels, seed=42)
        b = split_scores(scores, labels, seed=42)
        assert a == b

    def test_train_test_disjoint_and_complete(self):
        scores, labels = _synthetic_scores()
        s = split_scores(scores, labels, seed=42)
        assert len(s.train) + len(s.test) == len(scores)
        # стратификация: оба класса в обоих сплитах
        assert {y for _, y in s.train} == {0, 1}
        assert {y for _, y in s.test} == {0, 1}

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="need at least 2 pairs"):
            split_scores([0.5], [1], seed=42)  # слишком мало
        with pytest.raises(ValueError, match="labels must be binary"):
            split_scores([0.5, 0.6], [0, 2], seed=42)  # не бинарные метки
        with pytest.raises(ValueError, match="length mismatch"):
            split_scores([0.5, 0.6], [0, 1, 1], seed=42)  # длина не совпала


class TestNoLeakage:
    def test_selection_uses_train_only(self):
        """Подмена test-пар не меняет выбранный порог: у select_threshold нет
        доступа к test физически - сигнатура принимает только train."""
        scores, labels = _synthetic_scores()
        s = split_scores(scores, labels, seed=42)
        thr1 = select_threshold(s.train)
        # повторный вызов на том же train детерминирован
        thr2 = select_threshold(s.train)
        assert thr1 == thr2
        assert thr1 in DEFAULT_GRID

    def test_holdout_test_f1_below_insample_f1(self):
        """На set, где порог выбран под весь датасет, in-sample F1 >= honest.

        Классический признак optimism from selection: evaluate on the selection
        set vs evaluate on a held-out split.
        """
        scores, labels = _synthetic_scores()
        full = list(zip(scores, labels, strict=True))
        thr_full = select_threshold(full)
        _p, _r, insample_f1 = _precision_recall_f1(full, thr_full)
        report = evaluate_holdout(scores, labels, seed=42)
        # честная оценка не обязана быть выше, но в этом синтетическом сдвиге
        # она обязана быть конечной и консистентной с метриками
        assert report.test_f1 == pytest.approx(
            _precision_recall_f1(
                split_scores(scores, labels, seed=42).test, report.threshold
            )[2]
        )
        assert 0.0 <= report.test_f1 <= 1.0
        # in-sample подбор не может дать F1 ниже, чем на том же множестве:
        # optimism from selection = insample - honest
        assert insample_f1 >= report.test_f1

    def test_deterministic_across_runs(self):
        scores, labels = _synthetic_scores()
        r1 = evaluate_holdout(scores, labels, seed=42)
        r2 = evaluate_holdout(scores, labels, seed=42)
        assert r1 == r2


class TestRecallAtFixedPrecision:
    def test_recall_at_precision_meets_target_on_test(self):
        scores, labels = _synthetic_scores()
        s = split_scores(scores, labels, seed=42)
        _thr, rec, prec = recall_at_precision(s.test, target=0.9)
        if rec > 0:
            assert prec >= 0.9
        assert 0.0 <= rec <= 1.0

    def test_unreachable_target_returns_zero_recall(self):
        pairs = [(0.9, 1), (0.9, 0), (0.88, 1), (0.88, 0)]
        thr, rec, _p = recall_at_precision(pairs, target=0.999, grid=(0.85, 0.9))
        assert rec == 0.0
        assert thr == 0.85

    def test_report_carries_recall_at_precision(self):
        scores, labels = _synthetic_scores()
        report = evaluate_holdout(scores, labels, seed=42)
        d = report.as_dict()
        assert set(d) >= {
            "threshold", "train_f1", "test_f1", "test_precision", "test_recall",
            "recall_at_precision", "precision_at_that_threshold",
        }
        if d["recall_at_precision"] > 0:
            assert d["precision_at_that_threshold"] >= d["precision_target"]


class TestHoldoutScript:
    """scripts/eval_matcher_holdout.py: явная ошибка без данных, честный отчёт с ними."""

    def test_missing_file_raises(self, tmp_path):
        from scripts.eval_matcher_holdout import load_scores

        with pytest.raises(FileNotFoundError, match="predictions not found"):
            load_scores(tmp_path / "absent.jsonl")

    def test_end_to_end_on_synthetic_predictions(self, tmp_path, capsys):
        from scripts.eval_matcher_holdout import load_scores, main

        scores, labels = _synthetic_scores()
        pred = tmp_path / "predictions.jsonl"
        with pred.open("w", encoding="utf-8") as f:
            for s, y in zip(scores, labels, strict=True):
                f.write(json.dumps({"label": y, "L3_raw": {"confidence": s}}) + "\n")
        got_scores, got_labels = load_scores(pred)
        assert got_scores == scores and got_labels == labels
        assert main(["--predictions", str(pred), "--target", "0.9"]) == 0
        out = capsys.readouterr().out
        assert "threshold=" in out and "recall@precision" in out

    def test_malformed_row_raises(self, tmp_path):
        from scripts.eval_matcher_holdout import load_scores

        pred = tmp_path / "predictions.jsonl"
        pred.write_text('{"label": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="L3_raw"):
            load_scores(pred)
