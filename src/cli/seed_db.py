"""Seed database from JSON data + it_skills + rpd_skills.

Usage:
    python -m src.cli seed-db [--drop]
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
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
    Recommendation,
    Skill,
    User,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KRM_PATH = DATA_DIR / "reference" / "krm_disciplines_09.03.02.json"
IT_SKILLS_PATH = DATA_DIR / "reference" / "it_skills.json"
RPD_SKILLS_PATH = DATA_DIR / "reference" / "rpd_skills.json"
RECOMMENDATIONS_PATH = DATA_DIR / "reference" / "teacher_recommendations.json"

_COMP_CODE_RE = re.compile(r"^(УК|ОПК|ПК|ППК|ИП)[\s-](\d+)$")


def _parse_comp_code(code: str) -> tuple[str, str]:
    m = _COMP_CODE_RE.match(code)
    if m:
        return m.group(1), m.group(2)
    return code, "0"


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


async def seed_krm(session, skill_map: dict[str, str]) -> None:
    with open(KRM_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    direction_data = data.get("09.03.02", {})
    disciplines_raw = direction_data.get("disciplines", {})

    result = await session.execute(select(Direction).where(Direction.code == "09.03.02"))
    direction = result.scalar_one_or_none()
    if not direction:
        direction = Direction(
            code="09.03.02",
            name=direction_data.get("direction_name", "09.03.02 Информационные системы и технологии"),
            profile=direction_data.get("profile", "Перспективные информационные технологии"),
            opop_year=2024,
        )
        session.add(direction)
        await session.flush()

    pv = ParseVersion(
        direction_id=direction.id,
        version=datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S"),
        opop_year=2024,
        total_disciplines=len(disciplines_raw),
        notes="Seed from parsed RPD JSON",
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

        pdfs = [f for f in _list_pdfs() if disc_name.lower().replace(" ", "") in Path(f).stem.lower().replace(" ", "")]
        for pdf in pdfs:
            session.add(PDFSource(discipline_id=disc.id, filename=Path(pdf).name, parse_status="parsed"))

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
                for idx, text in enumerate(ksa_data.get(comp_code, {}).get(kt, [])):
                    session.add(KSAEntry(competency_id=comp.id, ksa_type=kt, original_text=text, sort_order=idx, parse_version_id=pv.id))
                    ksa_count += 1

            for skill_name in skills_data.get(comp_code, []):
                sk = skill_name.lower()
                skill_id = skill_map.get(sk)
                if not skill_id:
                    continue
                existing_cs = await session.execute(
                    select(CompetencySkill).where(CompetencySkill.competency_id == comp.id, CompetencySkill.skill_id == skill_id, CompetencySkill.ksa_type == "flat")
                )
                if not existing_cs.scalar_one_or_none():
                    session.add(CompetencySkill(competency_id=comp.id, skill_id=skill_id, ksa_type="flat", source_text=skill_name, match_type="fuzzy", parse_version_id=pv.id))
                    cs_count += 1

    await session.commit()
    pv.total_competencies = comp_count
    pv.total_skills = cs_count
    pv.total_ksa_items = ksa_count
    await session.commit()
    print(f"Disciplines: {disc_count} new / Competencies: {comp_count} / Links: {cs_count} / KSA: {ksa_count}")


async def seed_users(session) -> None:
    users_file = Path(__file__).parent.parent.parent / "users.json"
    if not users_file.exists():
        print("users.json not found, skipping users seed")
        return
    with open(users_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    from sqlalchemy import text as sa_text
    created = 0
    for email, info in raw.items():
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none():
            continue
        result = await session.execute(
            sa_text("SELECT crypt(:pw, gen_salt('bf')) AS pw_hash"),
            {"pw": info["password"]},
        )
        pw_hash = result.scalar_one()
        session.add(User(
            email=email,
            password_hash=pw_hash,
            full_name=info.get("name", email.split("@")[0]),
            role=info["role"],
        ))
        created += 1
        print(f"  User created: {email} ({info['role']})")
    if created:
        await session.commit()
    print(f"Users: {created} created, rest already exist")


async def seed_recommendations(session) -> None:
    try:
        with open(RECOMMENDATIONS_PATH, "r", encoding="utf-8") as f:
            recs = json.load(f)
    except FileNotFoundError:
        print("No recommendations to seed")
        return
    count = 0
    for rec in recs:
        disc_name = rec.get("discipline", "")
        result = await session.execute(select(Discipline).where(Discipline.name == disc_name))
        disc = result.scalar_one_or_none()
        if not disc:
            continue
        comp_code = rec.get("competency")
        comp_id = None
        if comp_code:
            result = await session.execute(select(Competency).where(Competency.discipline_id == disc.id, Competency.code == comp_code))
            comp = result.scalar_one_or_none()
            if comp:
                comp_id = comp.id
        session.add(Recommendation(discipline_id=disc.id, competency_id=comp_id, suggestion=rec.get("suggestion", ""), suggestion_type=rec.get("type", "modify")))
        count += 1
    await session.commit()
    print(f"Recommendations: {count}")


def _list_pdfs() -> list[str]:
    pdf_dir = DATA_DIR.parent / "temp" / "rpd_pdfs"
    return [str(f) for f in pdf_dir.glob("*.pdf")] if pdf_dir.exists() else []


async def main(drop: bool = False, version: str | None = None) -> None:
    print("Creating tables...")
    await create_tables(drop_first=drop)
    async with async_session_factory() as session:
        print("Seeding skills...")
        skill_map = await seed_skills(session)
        print("Seeding KRM data...")
        await seed_krm(session, skill_map)
        print("Seeding users...")
        await seed_users(session)
        print("Seeding recommendations...")
        await seed_recommendations(session)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    parser.add_argument("--version")
    args = parser.parse_args()
    asyncio.run(main(drop=args.drop, version=args.version))
