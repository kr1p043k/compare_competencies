"""Build labeling pairs: role x skill candidates for expert (or weak) labeling.

Per role: ~20 taxonomy-core + ~20 market-top (non-core) + ~20 market-tail
(random negatives) => ~60 pairs/role, 6 roles => ~360 pairs.

Output: labeling/pairs.csv (role,skill,source)
Skill names are lowercase-normalized to match the training pipeline.
"""
import asyncio
import csv
import json
import random
from collections import Counter
from pathlib import Path

import asyncpg

from src.config import settings

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
OUT = ROOT / "labeling"

ROLES = [
    "Backend Developer",
    "Frontend Developer",
    "Data Scientist",
    "DevOps Engineer",
    "QA Engineer",
    "Android Developer",
]
PER_SOURCE = 20
SEED = 42


def norm(s: str) -> str:
    return s.strip().lower()


def role_core_skills() -> dict[str, list[str]]:
    taxonomy = json.loads((REF / "profession_taxonomy.json").read_text(encoding="utf-8"))
    krm = json.loads((REF / "krm_competency_mapping.json").read_text(encoding="utf-8"))
    professions = taxonomy.get("professions", {})
    out: dict[str, list[str]] = {}
    for role in ROLES:
        prof = professions.get(role, {})
        skills: set[str] = set()
        for code in prof.get("competency_codes", []):
            for kw in krm.get(code, []):
                kwn = norm(kw)
                if kwn and len(kwn) > 2:
                    skills.add(kwn)
        out[role] = sorted(skills)
    return out


async def market_frequencies(limit_vac: int = 18844) -> Counter:
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    pool = await asyncpg.create_pool(url, min_size=1, max_size=2, command_timeout=300)
    async with pool.acquire() as c:
        rows = await c.fetch(
            "SELECT parsed_skills FROM vacancies WHERE parsed_skills IS NOT NULL LIMIT $1",
            limit_vac,
        )
    await pool.close()
    freq: Counter = Counter()
    for r in rows:
        ps = r["parsed_skills"]
        try:
            vals = json.loads(ps) if isinstance(ps, str) else ps
        except Exception:
            continue
        if not isinstance(vals, list):
            continue
        for s in vals:
            if isinstance(s, str):
                ns = norm(s)
                if ns and len(ns) > 1:
                    freq[ns] += 1
    return freq


async def main() -> None:
    rng = random.Random(SEED)
    cores = role_core_skills()
    for role, skills in cores.items():
        print(f"core[{role}] = {len(skills)}")
    freq = await market_frequencies()
    print("market distinct skills:", len(freq))
    ranked = [s for s, _ in freq.most_common()]
    tail = ranked[150:]

    pairs: list[tuple[str, str, str]] = []
    for role in ROLES:
        core = cores[role]
        core_set = set(core)
        rng.shuffle(core)
        n_core = min(PER_SOURCE, len(core))
        for s in core[:n_core]:
            pairs.append((role, s, "core"))
        market_near = [s for s in ranked if s not in core_set][: PER_SOURCE * 3]
        rng.shuffle(market_near)
        for s in market_near[:PER_SOURCE]:
            pairs.append((role, s, "market"))
        cands = [s for s in tail if s not in core_set]
        rng.shuffle(cands)
        for s in cands[:PER_SOURCE]:
            pairs.append((role, s, "random"))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pairs.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["role", "skill", "source"])
        w.writerows(pairs)
    print("wrote", OUT / "pairs.csv", "rows:", len(pairs))


if __name__ == "__main__":
    asyncio.run(main())
