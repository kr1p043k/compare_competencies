"""Weak gold v1: 3 transparent labelers + majority adjudication.

L1 taxonomy: 2 = in role core, 1 = in another role core, 0 = elsewhere.
L2 frequency: 1 = pair-skill in top quartile of market freq, else 0.
L3 semantic: cosine(skill, role-core centroid) tertiles per role -> 2/1/0.
Final: majority; all-differ (0,1,2) -> 1 (conservative middle).

HONEST LABEL: kind=weak-v1, NOT human. For pipeline validation and as a
prior; human round-2 still required before any production claim.

Outputs (labeling/):
  votes_weak_l1.csv, votes_weak_l2.csv, votes_weak_l3.csv,
  gold_v1.csv (role;skill;grade), gold_v1.META.json
"""
import asyncio
import csv
import json
from collections import Counter
from pathlib import Path

import asyncpg
import numpy as np

from scripts.experiments.build_labeling_pairs import ROLES, role_core_skills
from src.config import settings
from src.parsing.api.embedding_loader import get_embedding_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labeling"


def read_pairs():
    rows = []
    with open(OUT / "pairs.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append((r["role"].strip(), r["skill"].strip().lower()))
    return rows


async def market_freq(skills: set[str]) -> Counter:
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    pool = await asyncpg.create_pool(url, min_size=1, max_size=2, command_timeout=300)
    async with pool.acquire() as c:
        dbrows = await c.fetch("SELECT parsed_skills FROM vacancies WHERE parsed_skills IS NOT NULL")
    await pool.close()
    freq: Counter = Counter()
    for r in dbrows:
        try:
            vals = json.loads(r["parsed_skills"]) if isinstance(r["parsed_skills"], str) else r["parsed_skills"]
        except Exception:
            continue
        if not isinstance(vals, list):
            continue
        for s in vals:
            if isinstance(s, str):
                ns = s.strip().lower()
                if ns in skills:
                    freq[ns] += 1
    return freq


def main() -> None:
    pairs = read_pairs()
    print("pairs:", len(pairs))
    cores = role_core_skills()
    core_sets = {r: set(v) for r, v in cores.items()}
    vocab = sorted({s for _, s in pairs} | {s for v in cores.values() for s in v})
    print("vocab to encode:", len(vocab))

    model = get_embedding_model()
    embs = model.encode(vocab, convert_to_numpy=True, show_progress_bar=False)
    E = {s: e for s, e in zip(vocab, embs, strict=False)}
    centroids = {}
    for role in ROLES:
        vecs = [E[s] for s in cores[role] if s in E]
        centroids[role] = np.mean(vecs, axis=0) if vecs else np.mean(list(E.values()), axis=0)

    async def freqs():
        return await market_freq(set(vocab))
    freq = asyncio.run(freqs())
    with open(OUT / "market_freq.json", "w", encoding="utf-8") as f:
        json.dump(dict(freq.most_common(500)), f, ensure_ascii=False)
    pair_freqs = sorted(freq.get(s, 0) for _, s in pairs)
    q75 = float(np.quantile(pair_freqs, 0.75))
    print("freq q75:", q75, "nonzero skills:", sum(1 for v in freq.values() if v > 0))

    # L3: per-role cosine tertiles
    cos: dict[str, list[float]] = {}
    for role in ROLES:
        c = centroids[role]
        cn = float(np.linalg.norm(c)) or 1.0
        vals = []
        for r, s in pairs:
            if r != role:
                continue
            e = E[s]
            vals.append(float(np.dot(e, c) / ((float(np.linalg.norm(e)) or 1.0) * cn)))
        cos[role] = vals
    cuts = {r: (float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))) for r, v in cos.items()}

    votes = {"l1": {}, "l2": {}, "l3": {}}
    for role, skill in pairs:
        v1 = 2 if skill in core_sets[role] else (1 if any(skill in core_sets[r] for r in ROLES if r != role) else 0)
        v2 = 1 if freq.get(skill, 0) >= q75 else 0
        e = E[skill]
        c = centroids[role]
        cv = float(np.dot(e, c) / ((float(np.linalg.norm(e)) or 1.0) * (float(np.linalg.norm(c)) or 1.0)))
        lo, hi = cuts[role]
        v3 = 2 if cv >= hi else (0 if cv < lo else 1)
        votes["l1"][(role, skill)] = v1
        votes["l2"][(role, skill)] = v2
        votes["l3"][(role, skill)] = v3

    for name, v in votes.items():
        with open(OUT / f"votes_weak_{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["role", "skill", "grade"])
            for (role, skill) in sorted(v):
                w.writerow([role, skill, v[(role, skill)]])

    final = {}
    for key in sorted(votes["l1"]):
        gs = [votes[k][key] for k in ("l1", "l2", "l3")]
        counts = Counter(gs)
        top = counts.most_common()
        grade = top[0][0] if (len(top) == 1 or top[0][1] > top[1][1]) else 1
        final[key] = grade
    with open(OUT / "gold_v1.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["role", "skill", "grade"])
        for (role, skill), g in sorted(final.items()):
            w.writerow([role, skill, g])
    marg = Counter(final.values())
    print("gold marginals:", dict(sorted(marg.items())))
    meta = {
        "kind": "weak-v1, NOT human",
        "labelers": {
            "l1": "taxonomy (2=in role core, 1=in other role core, 0=else)",
            "l2": f"market frequency top-quartile(>={q75}) -> 1 else 0",
            "l3": "cosine to role-core centroid, per-role tertiles -> 2/1/0",
        },
        "adjudication": "majority; all-differ -> 1",
        "pairs": len(final),
        "roles": ROLES,
        "marginals": {str(k): v for k, v in sorted(marg.items())},
    }
    with open(OUT / "gold_v1.META.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("wrote gold_v1.csv + META")


if __name__ == "__main__":
    main()
