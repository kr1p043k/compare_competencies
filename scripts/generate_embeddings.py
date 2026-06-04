"""Генерация эмбеддингов для всех навыков и запись в skills.embedding.

Использует sentence-transformers, модель paraphrase-multilingual-MiniLM-L12-v2.
Пропускает навыки, у которых embedding уже заполнен (флаг --force).

Usage:
    python scripts/generate_embeddings.py [--force]
"""

import argparse
import asyncio
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select, update

from src.database import async_session_factory
from src.models.krm_models import Skill


async def generate_embeddings(force: bool = False) -> None:
    print("Loading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"Model dim: {dim}")

    async with async_session_factory() as session:
        query = select(Skill).where(Skill.is_active == True)
        if not force:
            query = query.where(Skill.embedding.is_(None))

        result = await session.execute(query)
        skills = result.scalars().all()

        if not skills:
            print("No skills to process")
            return

        names = [s.name for s in skills]
        print(f"Generating {len(names)} embeddings...")

        embeddings: np.ndarray = model.encode(names, convert_to_numpy=True, show_progress_bar=True)

        count = 0
        for skill, emb in zip(skills, embeddings):
            stmt = (
                update(Skill)
                .where(Skill.id == skill.id)
                .values(embedding=emb.tolist())
            )
            await session.execute(stmt)
            count += 1

        await session.commit()
        print(f"Updated {count} skills")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate skill embeddings")
    parser.add_argument("--force", action="store_true", help="Re-generate all embeddings")
    args = parser.parse_args()
    asyncio.run(generate_embeddings(force=args.force))


if __name__ == "__main__":
    main()
