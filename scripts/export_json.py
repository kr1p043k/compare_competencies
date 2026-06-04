"""Экспорт всех таблиц БД в JSON-файлы (data/reference/).

Usage:
    python scripts/export_json.py
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import inspect, select, text

from src.database import async_session_factory
from src.models.krm_models import (
    Competency,
    CompetencySkill,
    CoverageAnalysis,
    Direction,
    Discipline,
    KSAEntry,
    MarketSkillMapping,
    ParseVersion,
    PDFSource,
    Recommendation,
    Skill,
    Student,
    StudentGroup,
    StudentSkill,
    User,
)

OUT = Path(__file__).parent.parent / "data" / "export"

TABLES: list[tuple[str, type]] = [
    ("directions", Direction),
    ("disciplines", Discipline),
    ("pdf_sources", PDFSource),
    ("parse_versions", ParseVersion),
    ("competencies", Competency),
    ("ksa_entries", KSAEntry),
    ("skills", Skill),
    ("competency_skills", CompetencySkill),
    ("users", User),
    ("recommendations", Recommendation),
    ("student_groups", StudentGroup),
    ("students", Student),
    ("student_skills", StudentSkill),
    ("market_skill_mappings", MarketSkillMapping),
    ("coverage_analyses", CoverageAnalysis),
]


def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    return obj


async def export_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    async with async_session_factory() as session:
        for name, model in TABLES:
            result = await session.execute(select(model))
            rows = result.scalars().all()

            # Convert ORM objects to dicts
            data = []
            for row in rows:
                d = {}
                for col in inspect(model).columns.keys():
                    val = getattr(row, col)
                    if isinstance(val, datetime):
                        d[col] = val.isoformat()
                    elif isinstance(val, list):
                        d[col] = [serialize(v) for v in val]
                    else:
                        d[col] = val
                data.append(d)

            path = OUT / f"{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            print(f"{name}: {len(data)} rows → {path}")

    print(f"\nDone. Export in {OUT}")


if __name__ == "__main__":
    asyncio.run(export_all())
