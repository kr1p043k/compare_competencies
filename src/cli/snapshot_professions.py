"""Snapshot per-profession skill frequencies from the vacancies database."""
from __future__ import annotations

import argparse
import asyncio
import asyncpg
import json
from collections import Counter
from datetime import datetime

from src import config
from src.analyzers.skills.trends import TrendAnalyzer


async def _query_profession(db_url: str, query: str) -> dict[str, int]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """SELECT LOWER(TRIM(value)) AS skill, COUNT(DISTINCT v.id) AS cnt
               FROM vacancies v,
                    jsonb_array_elements_text(
                        COALESCE(v.parsed_skills, v.key_skills, '[]'::jsonb)
                    ) AS value
               WHERE v.parsed_skills IS NOT NULL
                 AND LOWER(v.name) LIKE '%' || $1 || '%'
               GROUP BY skill
               ORDER BY cnt DESC""",
            query.lower(),
        )
        return {r["skill"]: r["cnt"] for r in rows if r["cnt"] > 0}
    finally:
        await conn.close()


def main(force: bool = False) -> None:
    prof_path = config.REFERENCE_DIR / "profession_taxonomy.json"
    if not prof_path.exists():
        print("profession_taxonomy.json not found")
        return

    taxonomy = json.loads(prof_path.read_text(encoding="utf-8"))
    professions = taxonomy.get("professions", {})
    print(f"Loaded {len(professions)} professions")

    db_url = config.settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    analyzer = TrendAnalyzer({})

    for prof_name, prof_data in professions.items():
        queries = prof_data.get("hh_queries", [prof_name])
        q = queries[0]
        print(f"  {prof_name:35s} query='{q}' ... ", end="", flush=True)

        skill_freq = asyncio.run(_query_profession(db_url, q))
        if not skill_freq:
            print("no data")
            continue

        analyzer.save_snapshot(skill_freq, source_type="targeted_query", profession=prof_name, apply_whitelist=True)
        print(f"{len(skill_freq)} skills")

    print("\nDone.")


if __name__ == "__main__":
    import sys
    main("--force" in sys.argv)
