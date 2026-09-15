"""Eval tripwire (v32, minimal): golden set loads, sane classes, kappa sanity.

NOT a quality gate yet (that needs expert-confirmed round_2 + ranking preds).
It guards file corruption / schema drift of labeling/gold.csv.
"""
import csv
from pathlib import Path

from labeling.score import fleiss_kappa, load_labels

GOLD = Path(__file__).resolve().parent.parent.parent / "labeling" / "gold.csv"


def test_gold_loads_with_three_classes():
    assert GOLD.exists()
    rows = load_labels(str(GOLD))
    assert len(rows) > 400
    cats = {v for r in rows for v in r["votes"]}
    assert cats == {0, 1, 2}


def test_gold_schema_stable():
    with open(GOLD, encoding="utf-8-sig") as f:
        header = f.readline().strip().split(";")
    assert header[:4] == ["id", "role", "skill", "gold"]


def test_kappa_sanity_unanimous_is_one():
    rows = load_labels(str(GOLD))[:50]
    assert fleiss_kappa(rows) == 1.0
