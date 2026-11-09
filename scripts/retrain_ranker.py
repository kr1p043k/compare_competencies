"""Periodic retrain orchestrator for the shipped LTR ranker.

Reuses the existing training entrypoints instead of reimplementing them:

  1. ``scripts/train_ltr_ranker.py``  - fit XGBRanker on gold + DB vacancies,
  2. ``scripts/experiments/make_ensemble.py``  - (optional) rebuild the shipped
     ensemble bundle from the freshly trained seed-models,
  3. ``scripts/check_model_manifests.py --write`` - refresh feature_names/NDCG
     sidecars from the new artifact,
  4. ``scripts/eval_frozen.py --check`` - the frozen-split regression gate; the
     retrain is rejected if NDCG@10 drops below ``data/reports/frozen_eval.json``.

Requires a live PostgreSQL (vacancies) and ``SECRET_KEY`` in the environment -
the same prerequisites as a manual training run. Nothing is faked: each step is
a real subprocess and the first non-zero exit stops the pipeline.

Usage:
    python scripts/retrain_ranker.py                 # train -> gate
    python scripts/retrain_ranker.py --ensemble      # also rebuild the bundle
    python scripts/retrain_ranker.py --gold labeling/gold_v1.csv --vacancies 5000
    python scripts/retrain_ranker.py --dry-run       # print the plan only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_steps(args: argparse.Namespace) -> list[list[str]]:
    steps: list[list[str]] = [
        [
            sys.executable,
            "scripts/train_ltr_ranker.py",
            "--gold",
            args.gold,
            "--vacancies",
            str(args.vacancies),
            "--seed",
            str(args.seed),
            "--out",
            args.out,
        ]
    ]
    if args.ensemble:
        steps.append([sys.executable, "scripts/experiments/make_ensemble.py"])
    steps.append([sys.executable, "scripts/check_model_manifests.py", "--write"])
    steps.append([sys.executable, "scripts/eval_frozen.py", "--check"])
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="labeling/gold_v1.csv")
    parser.add_argument("--vacancies", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/models/ltr_ranker_xgb_rank.joblib")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="rebuild data/models/ltr_ranker_ensemble_human.joblib after training",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    steps = build_steps(args)
    for i, step in enumerate(steps, start=1):
        printable = " ".join(step)
        print(f"[{i}/{len(steps)}] {printable}")
        if args.dry_run:
            continue
        result = subprocess.run(step, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"\nFAILED at step {i} (exit {result.returncode}): {printable}")
            return result.returncode

    if args.dry_run:
        print("\ndry-run: no commands executed")
        return 0
    print("\nretrain complete and frozen_eval --check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
