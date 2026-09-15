"""Validate transcribed votes and compute inter-rater agreement (Cohen's kappa).

Usage:
  python labeling/check_votes.py --e2 f1 f2 ... --e3 g1 g2 ...
e3 files must have argtype column (vote<->argument consistency check).
"""
import argparse
import csv
import sys
from collections import Counter

ARGMAP = {"K": 2, "S": 1, "M": 0}


def load(paths, has_argtype=False):
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                v = int(r["vote"])
                assert v in (0, 1, 2), f"bad vote {v} at id {r['id']} in {path}"
                i = int(r["id"])
                assert i not in rows, f"duplicate id {i}"
                rows[i] = {"role": r["role"], "skill": r["skill"], "vote": v,
                           "argtype": (r.get("argtype") or "").strip() if has_argtype else ""}
    return rows


def cohen(a, b):
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = Counter(a), Counter(b)
    pe = sum((pa[k] / n) * (pb[k] / n) for k in (0, 1, 2))
    return (agree - pe) / (1 - pe) if pe < 1 else 1.0, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", nargs="+", required=True)
    ap.add_argument("--e3", nargs="+", required=True)
    args = ap.parse_args()

    e2 = load(args.e2)
    e3 = load(args.e3, has_argtype=True)
    print(f"e2 rows: {len(e2)}, e3 rows: {len(e3)}")
    ids = sorted(set(e2) & set(e3))
    print(f"common ids: {len(ids)}")
    mism = [(i, e2[i]["skill"][:35], e3[i]["skill"][:35]) for i in ids
            if (e2[i]["role"], e2[i]["skill"]) != (e3[i]["role"], e3[i]["skill"])]
    print(f"skill mismatches: {len(mism)}")
    for m in mism[:10]:
        print("  MISMATCH", m)
    bad_arg = [(i, e3[i]["vote"], e3[i]["argtype"]) for i in ids
               if ARGMAP.get(e3[i]["argtype"], -1) != e3[i]["vote"]]
    print(f"e3 vote<->argtype mismatches: {len(bad_arg)}")
    for b in bad_arg[:10]:
        print("  ARG-MISMATCH", b)

    by_role: dict[str, list] = {}
    for i in ids:
        by_role.setdefault(e2[i]["role"], []).append(i)
    tall, tagree = [], 0.0
    for role, rs in by_role.items():
        a = [e2[i]["vote"] for i in rs]
        b = [e3[i]["vote"] for i in rs]
        k, agree = cohen(a, b)
        tall.append((role, len(rs), agree, k, sorted(Counter(a).items()), sorted(Counter(b).items())))
    for role, n, agree, k, ma, mb in tall:
        print(f"[{role}] n={n} agree={agree:.3f} kappa={k:.3f} e2={dict(ma)} e3={dict(mb)}")
    a = [e2[i]["vote"] for i in ids]
    b = [e3[i]["vote"] for i in ids]
    k, agree = cohen(a, b)
    print(f"OVERALL n={len(ids)} agree={agree:.3f} kappa={k:.3f}")
    conf = Counter(zip(a, b))
    print("confusion (e2 x e3):")
    for x in (0, 1, 2):
        print("  e2=%d: " % x + " ".join(f"e3={y}:{conf[(x, y)]}" for y in (0, 1, 2)))
    print("extreme flips e2=0/e3=2:")
    for i in ids:
        if e2[i]["vote"] == 0 and e3[i]["vote"] == 2:
            print(f"  id={i} :: {e2[i]['skill'][:60]}")


if __name__ == "__main__":
    main()
