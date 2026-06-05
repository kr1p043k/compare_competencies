"""Check which disciplines have 'linux' as a skill."""
import asyncio, asyncpg

async def main():
    pool = await asyncpg.create_pool(
        "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
        min_size=1, max_size=2)
    
    # Find linux links
    rows = await pool.fetch("""
        SELECT d.name AS disc_name, s.name AS skill_name, s.source,
               c.code AS comp_code
        FROM competency_skills cs
        JOIN skills s ON s.id = cs.skill_id
        JOIN competencies c ON c.id = cs.competency_id
        JOIN disciplines d ON d.id = c.discipline_id
        WHERE LOWER(s.name) = 'linux'
        ORDER BY d.name
    """)
    
    print("=== 'linux' in competency_skills ===")
    for r in rows:
        print(f"  DISC: {r['disc_name']} | COMP: {r['comp_code']} | SOURCE: {r['source']}")
    
    # Count ALL market-source skills per discipline
    print("\n=== Market-source skills per discipline (top 10) ===")
    mrows = await pool.fetch("""
        SELECT d.name AS disc_name, COUNT(*) AS cnt
        FROM competency_skills cs
        JOIN skills s ON s.id = cs.skill_id
        JOIN competencies c ON c.id = cs.competency_id
        JOIN disciplines d ON d.id = c.discipline_id
        WHERE s.source = 'market'
        GROUP BY d.name
        ORDER BY cnt DESC
        LIMIT 10
    """)
    for r in mrows:
        print(f"  DISC: {r['disc_name']} | market skills: {r['cnt']}")
    
    # Foreign language specific market skills
    print("\n=== All market skills linked to foreign language disciplines ===")
    frows = await pool.fetch("""
        SELECT d.name AS disc_name, s.name AS skill_name, c.code AS comp_code
        FROM competency_skills cs
        JOIN skills s ON s.id = cs.skill_id
        JOIN competencies c ON c.id = cs.competency_id
        JOIN disciplines d ON d.id = c.discipline_id
        WHERE s.source = 'market'
          AND d.name ILIKE '%иностран%'
        ORDER BY s.name
    """)
    for r in frows:
        print(f"  DISC: {r['disc_name']} | SKILL: {r['skill_name']} | COMP: {r['comp_code']}")
    
    await pool.close()

asyncio.run(main())
