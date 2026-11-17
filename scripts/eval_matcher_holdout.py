"""Честная holdout-оценка semantic-порога (BACKLOG 1.4).

In-sample проблема: F1~0.99 на L3 считался на том же множестве, где порог 0.78
подбирался sweep'ом (scripts/experiments/benchmark_matching.py:367, 1115-1124).
Скрипт берёт сохранённые предсказки benchmark'а (predictions.jsonl: label +
L3_raw.confidence), делит детерминированно (seed 42), выбирает порог на TRAIN
и считает F1 / precision / recall / recall@precision=0.9 на TEST.

Usage:
    python scripts/eval_matcher_holdout.py [--predictions PATH] [--target 0.9]

Без файла предсказок - явная ошибка, не молчаливый ноль.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.threshold_holdout import (  # noqa: E402
    DEFAULT_PRECISION_TARGET,
    DEFAULT_SEED,
    evaluate_holdout,
)

DEFAULT_PREDICTIONS = (
    Path(__file__).resolve().parent.parent
    / "data" / "benchmark" / "article_datasets" / "predictions.jsonl"
)


def load_scores(path: Path) -> tuple[list[float], list[int]]:
    """Извлечь (L3_raw score, label) из predictions.jsonl."""
    if not path.exists():
        raise FileNotFoundError(
            f"predictions not found: {path} - сначала прогони "
            f"python -m scripts.experiments.benchmark_matching (нужна БД)")
    scores: list[float] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            label = row.get("label")
            l3 = row.get("L3_raw") or {}
            if label not in (0, 1) or "confidence" not in l3:
                raise ValueError(f"{path.name}:{line_no}: нет label/L3_raw.confidence")
            scores.append(float(l3["confidence"]))
            labels.append(int(label))
    if len(scores) < 4:
        raise ValueError(f"too few pairs for a train/test split: {len(scores)}")
    return scores, labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    ap.add_argument("--target", type=float, default=DEFAULT_PRECISION_TARGET)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    scores, labels = load_scores(args.predictions)
    report = evaluate_holdout(scores, labels, seed=args.seed, precision_target=args.target)
    d = report.as_dict()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print(f"\nthreshold={d['threshold']:.2f} выбран на train "
          f"(train F1={d['train_f1']:.3f}); test F1={d['test_f1']:.3f} "
          f"prec={d['test_precision']:.3f} rec={d['test_recall']:.3f} "
          f"(n={d['test_n']}); recall@precision>={d['precision_target']:.2f}"
          f"={d['recall_at_precision']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
