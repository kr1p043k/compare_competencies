"""Compute competency trends from trend_snapshots → competency_skills.

Aggregates per-skill trend direction/change_pct up to competency level.

Usage:
    python -m src.cli compute-competency-trends [--force]
"""

import asyncio
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select, delete

from src.database import async_session_factory
from src.models.krm_models import (
    Competency,
    CompetencySkill,
    Skill,
    TrendSnapshot,
)


async def main(force: bool = False) -> None:
    async with async_session_factory() as session:
        snaps_result = await session.execute(
            select(TrendSnapshot).order_by(TrendSnapshot.snapshot_date.asc())
        )
        snapshots = snaps_result.scalars().all()
        if len(snapshots) < 2:
            print("Need at least 2 snapshots for trends")
            return

        if force:
            await session.execute(delete(CompetencyTrend))
            await session.commit()

        pairs = list(zip(snapshots, snapshots[1:]))
        for prev_snap, cur_snap in pairs:
            prev_freq = prev_snap.skill_freq
            cur_freq = cur_snap.skill_freq
            date = cur_snap.snapshot_date

            existing = await session.execute(
                select(CompetencyTrend).where(CompetencyTrend.snapshot_date == date)
            )
            if existing.scalars().first() and not force:
                print(f"Skipping {date.date()} — already computed (use --force to recompute)")
                continue

            cs_result = await session.execute(
                select(CompetencySkill, Competency)
                .join(Competency, Competency.id == CompetencySkill.competency_id)
            )
            comp_skills_rows = cs_result.all()

            comp_skills_map: dict[str, dict] = {}
            for cs, comp in comp_skills_rows:
                if comp.id not in comp_skills_map:
                    comp_skills_map[comp.id] = {"code": comp.code, "skills": []}
                comp_skills_map[comp.id]["skills"].append(cs.skill_id)

            skill_result = await session.execute(select(Skill))
            skills = {s.id: s.name for s in skill_result.scalars().all()}

            for comp_id, info in comp_skills_map.items():
                changes = []
                for sid in info["skills"]:
                    name = skills.get(sid)
                    if not name:
                        continue
                    prev_val = prev_freq.get(name, 0)
                    cur_val = cur_freq.get(name, 0)
                    if prev_val == 0:
                        continue
                    change_pct = ((cur_val - prev_val) / prev_val) * 100
                    changes.append(change_pct)

                if not changes:
                    continue

                avg_change = sum(changes) / len(changes)
                rising = sum(1 for c in changes if c >= 5.0)
                falling = sum(1 for c in changes if c <= -5.0)
                if rising > falling:
                    direction = "rising"
                elif falling > rising:
                    direction = "falling"
                else:
                    direction = "stable"

                ct = CompetencyTrend(
                    competency_id=comp_id,
                    trend_direction=direction,
                    change_pct=round(avg_change, 2),
                    snapshot_date=date,
                    skill_count=len(changes),
                )
                session.add(ct)

            await session.commit()
            print(f"Trends computed for {date}: {len(comp_skills_map)} competencies")

    print("Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
