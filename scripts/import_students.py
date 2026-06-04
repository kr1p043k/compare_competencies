"""Импорт студентов из CSV в БД.

Формат CSV:
    full_name,group_name,direction_code,skills
    Иванов Иван,ИСИТ-31,09.03.02,"python,sql,linux"

Пропускает строки с уже существующими студентами.

Usage:
    python scripts/import_students.py students.csv
"""

import argparse
import asyncio
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select

from src.database import async_session_factory
from src.models.krm_models import Direction, Skill, Student, StudentGroup, StudentSkill


async def import_students(csv_path: str) -> None:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Empty CSV")
        return

    async with async_session_factory() as session:
        groups_cache: dict[str, str] = {}
        skills_cache: dict[str, str] = {}
        total = 0

        for row in rows:
            full_name = row.get("full_name", "").strip()
            group_name = row.get("group_name", "").strip()
            direction_code = row.get("direction_code", "09.03.02")
            skills_str = row.get("skills", "").strip()

            if not full_name or not group_name:
                continue

            # Resolve direction
            result = await session.execute(
                select(Direction).where(Direction.code == direction_code)
            )
            direction = result.scalar_one_or_none()
            if not direction:
                print(f"  Direction {direction_code} not found, skip")
                continue

            # Resolve or create group
            if group_name not in groups_cache:
                result = await session.execute(
                    select(StudentGroup).where(StudentGroup.name == group_name)
                )
                group = result.scalar_one_or_none()
                if not group:
                    group = StudentGroup(
                        direction_id=direction.id,
                        name=group_name,
                        year=2024,
                    )
                    session.add(group)
                    await session.flush()
                groups_cache[group_name] = group.id

            group_id = groups_cache[group_name]

            # Create student
            student = Student(group_id=group_id, full_name=full_name)
            session.add(student)
            await session.flush()

            # Process skills
            if skills_str:
                for sk in skills_str.split(","):
                    sk = sk.strip().lower()
                    if not sk:
                        continue

                    # Resolve skill
                    if sk not in skills_cache:
                        result = await session.execute(
                            select(Skill).where(Skill.name == sk)
                        )
                        skill = result.scalar_one_or_none()
                        skills_cache[sk] = skill.id if skill else None

                    skill_id = skills_cache.get(sk)
                    if not skill_id:
                        continue

                    ss = StudentSkill(
                        student_id=student.id,
                        skill_id=skill_id,
                        source="auto_extracted",
                        proficiency=0.5,
                    )
                    session.add(ss)

            total += 1

        await session.commit()
        print(f"Imported {total} students")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import students from CSV")
    parser.add_argument("csv", help="Path to CSV file")
    args = parser.parse_args()
    asyncio.run(import_students(args.csv))


if __name__ == "__main__":
    main()
