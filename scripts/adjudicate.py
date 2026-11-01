"""Adjudicate 3 experts -> gold.csv + round_2.csv + agreement report.

Inputs (temp/):
  pairs_excel_expert1_careful.csv  (id;role;skill;expert1;...)
  pairs_expert2.csv                (id;role;skill;expert2;...)
  votes_e3_raw.csv                 (id;role;skill;expert3;argument)
Outputs (labeling/):
  votes_e1.csv, votes_e2.csv, votes_e3.csv (role;skill;grade)
  gold.csv (role;skill;grade) - majority; all-differ {0,1,2} -> 1
  round_2.csv - all-differ rows for re-review
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(r"C:\forCode\temp")
OUT = ROOT / "labeling"


def read_col(path: Path, col: str) -> dict[int, tuple[str, str, int]]:
    rows = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        # tolerate doubled BOM (\ufeff leftovers) in headers
        if reader.fieldnames:
            reader.fieldnames = [c.replace("\ufeff", "").strip() for c in reader.fieldnames]
        for r in reader:
            try:
                g = int((r[col] or "").strip())
            except (ValueError, AttributeError):
                continue
            if g not in (0, 1, 2):
                continue
            rows[int(r["id"])] = (r["role"].strip(), r["skill"].strip(), g)
    return rows


def fleiss(table, n_cat: int) -> float:
    N = len(table)
    n0 = sum(table[0])
    p = [sum(r[j] for r in table) / (N * n0) for j in range(n_cat)]
    pe = sum(x * x for x in p)
    po = sum(sum(c * (c - 1) for c in r) / (ni * (ni - 1)) for r, ni in zip(table, [(sum(r)) for r in table])) / N
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cohen(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = (0, 1, 2)
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    e1 = read_col(TMP / "pairs_excel_expert1_careful.csv", "expert1")
    e2 = read_col(TMP / "pairs_expert2.csv", "expert2")
    e3 = read_col(TMP / "votes_e3_raw.csv", "expert3")
    print(f"n: e1={len(e1)} e2={len(e2)} e3={len(e3)}")
    ids = sorted(set(e1) & set(e2) & set(e3))
    print(f"common ids: {len(ids)} (missing: {sorted(set(range(500)) - set(ids))})")
    mism = [(i,) for i in ids if (e1[i][0], e1[i][1]) != (e2[i][0], e2[i][1]) or (e1[i][0], e1[i][1]) != (e3[i][0], e3[i][1])]
    print(f"key mismatches: {len(mism)}", mism[:5])
    if len(ids) < 500 or mism:
        print("ALIGNMENT PROBLEM - stop"); sys.exit(1)

    E = [e1, e2, e3]
    for k, e in enumerate(E, 1):
        print(f"marginals_e{k}:", dict(sorted(Counter(e[i][2] for i in ids).items())))
    table3 = [[sum(1 for e in E if e[i][2] == c) for c in (0, 1, 2)] for i in ids]
    table2 = [[[3 - sum(1 for e in E if e[i][2] == 0), sum(1 for e in E if e[i][2] == 0)][0],
               sum(1 for e in E if e[i][2] == 0)] for i in ids]
    # binary: [n_relevant, n_zero]
    table2 = [[sum(1 for e in E if e[i][2] >= 1), sum(1 for e in E if e[i][2] == 0)] for i in ids]
    print(f"fleiss_3way={fleiss(table3, 3):.3f} fleiss_binary={fleiss(table2, 2):.3f}")
    g = [[e[i][2] for i in ids] for e in E]
    print(f"cohen e1-e2={cohen(g[0], g[1]):.3f} e1-e3={cohen(g[0], g[2]):.3f} e2-e3={cohen(g[1], g[2]):.3f}")
    bg = [[1 if x >= 1 else 0 for x in gg] for gg in g]
    print(f"cohen-binary e1-e2={cohen(bg[0], bg[1]):.3f} e1-e3={cohen(bg[0], bg[2]):.3f} e2-e3={cohen(bg[1], bg[2]):.3f}")

    # --- expert-3 focus ---
    e3_alone = [i for i in ids if g[2][ids.index(i)] not in (g[0][ids.index(i)], g[1][ids.index(i)]) and g[0][ids.index(i)] == g[1][ids.index(i)]]
    print(f"e3-alone-vs-unanimous-pair: {len(e3_alone)}")
    o2 = [i for i in ids if set([e1[i][2], e2[i][2], e3[i][2]]) == {0, 1, 2}]
    print(f"all-differ 0/1/2: {len(o2)}")
    e3_up = [i for i in e3_alone if e3[i][2] > e1[i][2]]
    e3_dn = [i for i in e3_alone if e3[i][2] < e1[i][2]]
    print(f"e3-alone: up={len(e3_up)} down={len(e3_dn)}")
    with open(OUT / "e3_outliers.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "role", "skill", "e1", "e2", "e3"])
        for i in sorted(e3_alone):
            w.writerow([i, e1[i][0], e1[i][1], e1[i][2], e2[i][2], e3[i][2]])
    print("wrote e3_outliers.csv")

    # --- votes + gold + round_2 ---
    for k, e in zip(("e1", "e2", "e3"), E):
        with open(OUT / f"votes_{k}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["role", "skill", "grade"])
            for i in sorted(ids):
                w.writerow([e[i][0], e[i][1], e[i][2]])
    gold, r2 = {}, []
    for i in sorted(ids):
        gs = [e1[i][2], e2[i][2], e3[i][2]]
        c = Counter(gs).most_common()
        if len(c) == 1 or c[0][1] > c[1][1]:
            gold[i] = c[0][0]
        else:
            gold[i] = 1
            r2.append(i)
    with open(OUT / "gold.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["role", "skill", "grade"])
        for i in sorted(ids):
            w.writerow([e1[i][0], e1[i][1], gold[i]])
    with open(OUT / "round_2.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "role", "skill", "e1", "e2", "e3"])
        for i in r2:
            w.writerow([i, e1[i][0], e1[i][1], e1[i][2], e2[i][2], e3[i][2]])
    print("gold marginals:", dict(sorted(Counter(gold.values()).items())), f"round_2={len(r2)}")


if __name__ == "__main__":
    main()
