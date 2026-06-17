"""Populate vacancies.parsed_skills by re-parsing all vacancies with NULL or empty parsed_skills."""
import asyncio
import json
import sys
from datetime import datetime

sys.path.insert(0, str(__file__).rsplit("\\", 4)[0])

from sqlalchemy import text

from src import Ok, Err
from src.database import async_session_factory
from src.models.vacancy import Area, Employer, KeySkill, Snippet, Vacancy
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.skills.skill_parser import SkillParser


async def main():
    parser = SkillParser()
    async with async_session_factory() as session:
        # count
        total = await session.execute(
            text("SELECT COUNT(*) FROM vacancies WHERE parsed_skills IS NULL OR jsonb_array_length(parsed_skills) = 0")
        )
        total_count = total.scalar()
        if total_count == 0:
            print("All vacancies already have parsed_skills")
            return
        print(f"Found {total_count} vacancies to parse")

        BATCH = 100
        offset = 0
        processed = 0
        while offset < total_count:
            rows = await session.execute(
                text("""
                    SELECT id, name, key_skills, description,
                           snippet_requirement, snippet_responsibility,
                           employer_name, area_name
                    FROM vacancies
                    WHERE parsed_skills IS NULL OR jsonb_array_length(parsed_skills) = 0
                    ORDER BY id
                    LIMIT :lim OFFSET :off
                """),
                {"lim": BATCH, "off": offset},
            )
            batch = rows.fetchall()
            if not batch:
                break

            updates = []
            for r in batch:
                try:
                    ks_raw = r.key_skills if isinstance(r.key_skills, list) else json.loads(r.key_skills) if isinstance(r.key_skills, str) else []
                    key_skills = [KeySkill(name=s) if isinstance(s, str) else KeySkill(name=s.get("name", str(s))) for s in ks_raw]

                    vac = Vacancy(
                        id=str(r.id),
                        name=r.name or "",
                        area=Area(id=0, name=r.area_name or ""),
                        employer=Employer(id="0", name=r.employer_name or ""),
                        key_skills=key_skills,
                        description=r.description,
                        snippet=Snippet(
                            requirement=r.snippet_requirement,
                            responsibility=r.snippet_responsibility,
                        ),
                    )

                    match parser.parse_vacancy(vac):
                        case Ok(skills):
                            texts = [s.text for s in skills if s.text]
                        case _:
                            texts = []

                    if texts:
                        norm_result = SkillNormalizer.normalize_batch(texts)
                        if norm_result.is_ok():
                            normed = norm_result.unwrap()
                        else:
                            normed = texts
                        normed = list(dict.fromkeys(normed))
                    else:
                        normed = []

                    updates.append((json.dumps(normed, ensure_ascii=False), r.id))
                except Exception as exc:
                    print(f"  ERROR vacancy {r.id}: {exc}")
                    updates.append(('[]', r.id))

            for skills_json, vid in updates:
                await session.execute(
                    text("UPDATE vacancies SET parsed_skills = CAST(:skills AS jsonb) WHERE id = :id"),
                    {"skills": skills_json, "id": vid},
                )

            processed += len(batch)
            offset += BATCH
            print(f"  parsed {processed}/{total_count} ({processed * 100 // total_count}%)")
            await session.commit()

        print(f"Done! Updated {processed} vacancies")


if __name__ == "__main__":
    asyncio.run(main())
