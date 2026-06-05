import asyncio
from src.db import create_pool, close_pool, get_pool

async def main():
    await create_pool()
    pool = get_pool()

    # Check rpd_skills in competency_skills
    rows = await pool.fetch("""
        SELECT s.name AS skill, d2.name AS disc
        FROM skills s
        JOIN competency_skills cs ON cs.skill_id = s.id
        JOIN competencies c ON c.id = cs.competency_id
        JOIN disciplines d2 ON d2.id = c.discipline_id
        WHERE s.source = 'rpd_skills'
        ORDER BY d2.name, s.name
    """)
    print(f"rpd_skills linked to competencies: {len(rows)}")
    by_disc = {}
    for r in rows:
        by_disc.setdefault(r["disc"], []).append(r["skill"])
    for dn, skills in sorted(by_disc.items()):
        print(f"  {dn}: {skills}")
    
    # Also check what the original RPD PDF produced vs cleaned
    rows2 = await pool.fetch("""
        SELECT s.source, COUNT(DISTINCT cs.skill_id) AS linked
        FROM skills s
        LEFT JOIN competency_skills cs ON cs.skill_id = s.id
        GROUP BY s.source
        ORDER BY s.source
    """)
    print("\nSkills by source with link count:")
    for r in rows2:
        print(f"  {r['source']}: {r['linked']} linked")

    await close_pool()

asyncio.run(main())
