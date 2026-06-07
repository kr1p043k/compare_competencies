"""Compute competency embeddings by mean-pooling associated skill embeddings.

Usage:
    python -m src.cli compute-competency-vectors [--force]
"""

import asyncio
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select, update

from src.database import async_session_factory
from src.models.krm_models import Competency, CompetencySkill, Skill


async def main(force: bool = False) -> None:
    async with async_session_factory() as session:
        query = select(Competency)
        if not force:
            query = query.where(Competency.embedding.is_(None))
        result = await session.execute(query)
        competencies = result.scalars().all()
        if not competencies:
            print("No competencies to process")
            return

        for comp in competencies:
            cs_result = await session.execute(
                select(CompetencySkill).where(CompetencySkill.competency_id == comp.id)
            )
            comp_skills = cs_result.scalars().all()
            if not comp_skills:
                continue

            skill_ids = [cs.skill_id for cs in comp_skills]
            s_result = await session.execute(
                select(Skill).where(Skill.id.in_(skill_ids), Skill.embedding.isnot(None))
            )
            skills = s_result.scalars().all()
            if not skills:
                continue

            embs = np.array([s.embedding for s in skills])
            mean_emb = np.mean(embs, axis=0).tolist()
            await session.execute(
                update(Competency).where(Competency.id == comp.id).values(embedding=mean_emb)
            )

        await session.commit()
        print(f"Updated {len(competencies)} competency vectors")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
