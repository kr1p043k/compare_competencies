"""Score expert labeling: agreement (Fleiss' kappa) + system ranking quality.

Usage:
    1. Fill expert1..3 columns in pairs.csv (0/1/2).
    2. python labeling/score.py --labels labeling/pairs.csv [--preds predictions.csv]

predictions.csv format: role,skill,score (system ranking per role).
Without --preds only agreement + label stats are reported.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter


def load_labels(path: str):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        is_gold = "gold" in (reader.fieldnames or [])
        for r in reader:
            if is_gold:
                rows.append({"id": r["id"], "role": r["role"], "skill": r["skill"],
                             "votes": [int(r["gold"])] * 3})
                continue
            try:
                votes = [int(r["expert1"]), int(r["expert2"]), int(r["expert3"])]
            except (ValueError, TypeError, KeyError):
                continue
            if any(v not in (0, 1, 2) for v in votes):
                continue
            rows.append({"id": r["id"], "role": r["role"], "skill": r["skill"], "votes": votes})
    return rows


def fleiss_kappa(rows) -> float:
    """Fleiss' kappa for 3 raters, 3 categories."""
    n = len(rows)
    if n == 0:
        return float("nan")
    p_bar = 0.0
    cat_totals = Counter()
    for r in rows:
        c = Counter(r["votes"])
        p_bar += (sum(v * (v - 1) for v in c.values())) / (3 * 2)
        for k, v in c.items():
            cat_totals[k] += v
    p_bar /= n
    p_e = sum((cat_totals[k] / (n * 3)) ** 2 for k in (0, 1, 2))
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def majority(votes):
    c = Counter(votes)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return 1  # full disagreement -> middle
    return top[0][0]


def dcg(gains, k=10):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labeling/pairs_excel.csv")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = load_labels(args.labels)
    print(f"labeled: {len(rows)}")
    if not rows:
        return
    if len({tuple(r["votes"]) for r in rows}) > 3:
        print(f"fleiss_kappa: {fleiss_kappa(rows):.3f}")
    else:
        print("(gold standard: agreement trivially 1.0, skipped)")
    by_role: dict[str, list] = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)
    for role, rs in by_role.items():
        maj = Counter(majority(r["votes"]) for r in rs)
        print(f"[{role}] n={len(rs)} majority: " + ", ".join(f"{k}:{v}" for k, v in sorted(maj.items())))

    if not args.preds:
        print("(no --preds: ranking quality skipped)")
        return
    preds: dict[str, list] = {}
    with open(args.preds, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            preds.setdefault(r["role"], []).append((r["skill"], float(r["score"])))
    for role, rs in by_role.items():
        truth = {r["skill"]: majority(r["votes"]) for r in rs}
        ranked = sorted(preds.get(role, []), key=lambda x: -x[1])
        gains = [truth.get(s, 0) for s, _ in ranked]
        ideal = sorted(truth.values(), reverse=True)
        idcg = dcg(ideal, args.k)
        ndcg = (dcg(gains, args.k) / idcg) if idcg > 0 else 0.0
        rel = sum(1 for g in gains[: args.k] if g >= 1)
        print(f"[{role}] NDCG@{args.k}={ndcg:.3f} P@{args.k}={rel / args.k:.3f}")


if __name__ == "__main__":
    main()
