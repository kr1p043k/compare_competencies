"""Засев всех 13 направлений из KRM-файлов в БД.

Каждый файл data/reference/krm_disciplines_<dir_code>.json соответствует одному
направлению, где dir_code = имя файла без префикса (напр. 02.03.02_och).
Внутри файла верхний ключ может отличаться (напр. 02.03.02) — используем имя
файла как Direction.code, а данные берём из первого значения.

Использование:
    python scripts/seed_all_directions.py [--drop]
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import select, text

from src.database import Base, async_session_factory, get_engine
from src.models.krm_models import (
    Competency,
    CompetencySkill,
    Direction,
    Discipline,
    KSAEntry,
    ParseVersion,
    PDFSource,
    Skill,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
if not DATA_DIR.exists():
    DATA_DIR = Path.cwd() / "data"
REFERENCE_DIR = DATA_DIR / "reference"
IT_SKILLS_PATH = REFERENCE_DIR / "it_skills.json"
RPD_SKILLS_PATH = REFERENCE_DIR / "rpd_skills.json"

_COMP_CODE_RE = re.compile(r"^(УК|ОПК|ПК|ППК|ИП)[\s-](\d+)$")


def _parse_comp_code(code: str) -> tuple[str, str]:
    m = _COMP_CODE_RE.match(code)
    if m:
        return m.group(1), m.group(2)
    return code, "0"


def list_krm_files() -> list[tuple[str, Path]]:
    """Возвращает [(dir_code, path)] для всех non-clean KRM-файлов."""
    result = []
    for path in sorted(REFERENCE_DIR.glob("krm_disciplines_*.json")):
        if "_clean" in path.name:
            continue
        dir_code = path.name[len("krm_disciplines_"):-len(".json")]
        if not re.match(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$", dir_code):
            continue
        result.append((dir_code, path))
    return result


async def create_tables(drop_first: bool = False) -> None:
    engine = get_engine()
    if drop_first:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        await conn.run_sync(Base.metadata.create_all)
    print("Tables ready.")


async def seed_skills(session) -> dict[str, str]:
    skill_map = {}
    for source in ("it_skills", "rpd_skills"):
        path = IT_SKILLS_PATH if source == "it_skills" else RPD_SKILLS_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            names = [s.strip() for s in raw if s.strip()]
        except FileNotFoundError:
            print(f"  {path} not found, skipping")
            continue
        for name in names:
            key = name.lower()
            existing = await session.execute(select(Skill).where(Skill.name == key))
            row = existing.scalar_one_or_none()
            if row:
                skill_map[key] = row.id
                continue
            skill = Skill(name=key, source=source)
            session.add(skill)
            await session.flush()
            skill_map[key] = skill.id
    await session.commit()
    print(f"Skills: {len(skill_map)} in taxonomy")
    return skill_map


async def seed_direction(session, skill_map: dict[str, str], dir_code: str, path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        print(f"  {dir_code}: empty file, skipped")
        return
    direction_data = next(iter(data.values()))
    disciplines_raw = direction_data.get("disciplines", {})
    if not isinstance(disciplines_raw, dict):
        print(f"  {dir_code}: no disciplines, skipped")
        return

    result = await session.execute(select(Direction).where(Direction.code == dir_code))
    direction = result.scalar_one_or_none()
    if not direction:
        direction = Direction(
            code=dir_code,
            name=direction_data.get("direction_name", dir_code),
            profile=direction_data.get("profile", ""),
            opop_year=2024,
        )
        session.add(direction)
        await session.flush()
        print(f"  {dir_code}: direction created ({len(disciplines_raw)} discs)")

    pv = ParseVersion(
        direction_id=direction.id,
        version=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        opop_year=2024,
        total_disciplines=len(disciplines_raw),
        notes="Seed from parsed RPD JSON (all directions)",
    )
    session.add(pv)
    await session.flush()

    disc_count = comp_count = cs_count = ksa_count = 0
    for disc_name, disc_data in sorted(disciplines_raw.items()):
        existing = await session.execute(
            select(Discipline).where(Discipline.direction_id == direction.id, Discipline.name == disc_name)
        )
        disc = existing.scalar_one_or_none()
        if not disc:
            disc = Discipline(direction_id=direction.id, name=disc_name)
            session.add(disc)
            await session.flush()
            disc_count += 1

        competencies = disc_data.get("competencies", [])
        skills_data = disc_data.get("skills", {})
        ksa_data = disc_data.get("ksa", {})

        for comp_code in competencies:
            category, number = _parse_comp_code(comp_code)
            existing_comp = await session.execute(
                select(Competency).where(Competency.discipline_id == disc.id, Competency.code == comp_code)
            )
            comp = existing_comp.scalar_one_or_none()
            if not comp:
                comp = Competency(discipline_id=disc.id, code=comp_code, category=category, number=number, parse_version_id=pv.id)
                session.add(comp)
                await session.flush()
                comp_count += 1

            for kt in ("knowledge", "abilities", "skills"):
                for idx, ksa_text in enumerate(ksa_data.get(comp_code, {}).get(kt, [])):
                    existing_ksa = await session.execute(
                        select(KSAEntry).where(
                            KSAEntry.competency_id == comp.id,
                            KSAEntry.ksa_type == kt,
                            KSAEntry.sort_order == idx,
                            KSAEntry.original_text == ksa_text,
                        )
                    )
                    if existing_ksa.scalar_one_or_none():
                        continue
                    session.add(KSAEntry(competency_id=comp.id, ksa_type=kt, original_text=ksa_text, sort_order=idx, parse_version_id=pv.id))
                    ksa_count += 1

            for skill_name in skills_data.get(comp_code, []):
                sk = skill_name.lower()
                skill_id = skill_map.get(sk)
                if not skill_id:
                    continue
                existing_cs = await session.execute(
                    select(CompetencySkill).where(
                        CompetencySkill.competency_id == comp.id,
                        CompetencySkill.skill_id == skill_id,
                        CompetencySkill.ksa_type == "flat",
                    )
                )
                if not existing_cs.scalar_one_or_none():
                    session.add(CompetencySkill(competency_id=comp.id, skill_id=skill_id, ksa_type="flat", source_text=skill_name, match_type="fuzzy", parse_version_id=pv.id))
                    cs_count += 1

    pv.total_competencies = comp_count
    pv.total_skills = cs_count
    pv.total_ksa_items = ksa_count
    await session.commit()
    print(f"  {dir_code}: discs={disc_count} new, comps={comp_count}, links={cs_count}, ksa={ksa_count}")


async def main(drop: bool = False) -> None:
    print("Creating tables...")
    await create_tables(drop_first=drop)
    async with async_session_factory() as session:
        print("Seeding skills...")
        skill_map = await seed_skills(session)
        print("Seeding directions...")
        for dir_code, path in list_krm_files():
            await seed_direction(session, skill_map, dir_code, path)
    print("\nDone.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(drop=args.drop))
