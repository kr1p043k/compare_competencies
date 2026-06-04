"""Import students from CSV into database.

Usage:
    python -m src.cli import-students students.csv
"""

import asyncio
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select

from src.database import async_session_factory
from src.models.krm_models import Direction, Skill, Student, StudentGroup, StudentSkill


async def main(csv_path: str) -> None:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Empty CSV")
        return

    async with async_session_factory() as session:
        groups_cache: dict[str, str] = {}
        skills_cache: dict[str, str | None] = {}
        total = 0

        for row in rows:
            full_name = row.get("full_name", "").strip()
            group_name = row.get("group_name", "").strip()
            direction_code = row.get("direction_code", "09.03.02")
            skills_str = row.get("skills", "").strip()
            if not full_name or not group_name:
                continue

            result = await session.execute(select(Direction).where(Direction.code == direction_code))
            direction = result.scalar_one_or_none()
            if not direction:
                print(f"  Direction {direction_code} not found, skip")
                continue

            if group_name not in groups_cache:
                result = await session.execute(select(StudentGroup).where(StudentGroup.name == group_name))
                group = result.scalar_one_or_none()
                if not group:
                    group = StudentGroup(direction_id=direction.id, name=group_name, year=2024)
                    session.add(group)
                    await session.flush()
                groups_cache[group_name] = group.id

            student = Student(group_id=groups_cache[group_name], full_name=full_name)
            session.add(student)
            await session.flush()

            if skills_str:
                for sk in skills_str.split(","):
                    sk = sk.strip().lower()
                    if not sk:
                        continue
                    if sk not in skills_cache:
                        result = await session.execute(select(Skill).where(Skill.name == sk))
                        s = result.scalar_one_or_none()
                        skills_cache[sk] = s.id if s else None
                    skill_id = skills_cache.get(sk)
                    if skill_id:
                        session.add(StudentSkill(student_id=student.id, skill_id=skill_id, source="auto_extracted", proficiency=0.5))
            total += 1

        await session.commit()
        print(f"Imported {total} students")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to CSV file")
    args = parser.parse_args()
    asyncio.run(main(args.csv))
