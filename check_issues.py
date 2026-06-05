import asyncio
from src.db import create_pool, close_pool, get_pool

async def main():
    await create_pool()
    pool = get_pool()

    # Issue 1: find what in "Иностранный язык для деловой коммуникации" matches "linux"
    rows = await pool.fetch("""
        SELECT d2.name AS disc, s.name AS skill, k.original_text AS ksa
        FROM directions d
        JOIN disciplines d2 ON d2.direction_id = d.id
        JOIN competencies c ON c.discipline_id = d2.id
        LEFT JOIN competency_skills cs ON cs.competency_id = c.id
        LEFT JOIN skills s ON s.id = cs.skill_id
        LEFT JOIN ksa_entries k ON k.competency_id = c.id
        WHERE d.code = $1 AND d2.name ILIKE $2
          AND (s.name IS NOT NULL OR k.original_text IS NOT NULL)
        ORDER BY c.code
    """, "09.03.02", "%деловой%")
    seen = set()
    for r in rows:
        sn = r["skill"] or ""
        ksa = str(r["ksa"] or "")
        print(f"  skill={sn:<30} ksa={ksa[:70]}")
        if sn:
            seen.add(sn)
        if ksa:
            seen.add(ksa)
    print()

    # Word-boundary check for "linux" against all text from this discipline
    import re
    def wm(a, b):
        return bool(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b))

    for t in seen:
        if wm("linux", t):
            print(f"  WORD_MATCH('linux', '{t}')")
        # Check what could false-positive — maybe 'lin' prefix + something?
        # 'linguistics', 'link', 'linear'? Let's check
        if "linux" in t.lower():
            print(f"  CONTAINS 'linux': {t}")


    # Issue 2: skill count per discipline
    rows2 = await pool.fetch("""
        SELECT d2.name AS disc,
               COUNT(DISTINCT cs.skill_id) AS skill_count,
               COUNT(DISTINCT k.id) AS ksa_count
        FROM directions d
        JOIN disciplines d2 ON d2.direction_id = d.id
        LEFT JOIN competencies c ON c.discipline_id = d2.id
        LEFT JOIN competency_skills cs ON cs.competency_id = c.id
        LEFT JOIN ksa_entries k ON k.competency_id = c.id
        WHERE d.code = $1
        GROUP BY d2.name
        ORDER BY skill_count DESC, ksa_count DESC
    """, "09.03.02")
    print(f"\n{'Discipline':<50} {'Skills':>6} {'KSA':>6}")
    print("-"*62)
    total_skills = 0
    total_ksa = 0
    zero_skills = 0
    for r in rows2:
        sc = r["skill_count"] or 0
        kc = r["ksa_count"] or 0
        total_skills += sc
        total_ksa += kc
        if sc == 0 and kc == 0:
            zero_skills += 1
        print(f"{r['disc'][:48]:<48} {sc:>6} {kc:>6}")
    print(f"\nTotal: {total_skills} skills, {total_ksa} KSA, {zero_skills} disciplines with 0 both")

    # Check source of skills per discipline (RPD vs it_skills)
    rows3 = await pool.fetch("""
        SELECT d2.name AS disc,
               COUNT(DISTINCT cs.skill_id) FILTER (WHERE s.source = 'rpd') AS rpd_skill_count,
               COUNT(DISTINCT cs.skill_id) FILTER (WHERE s.source = 'it_skills') AS it_skill_count
        FROM directions d
        JOIN disciplines d2 ON d2.direction_id = d.id
        LEFT JOIN competencies c ON c.discipline_id = d2.id
        LEFT JOIN competency_skills cs ON cs.competency_id = c.id
        LEFT JOIN skills s ON s.id = cs.skill_id
        WHERE d.code = $1
        GROUP BY d2.name
        ORDER BY rpd_skill_count DESC, it_skill_count DESC
    """, "09.03.02")
    print(f"\n{'Discipline':<50} {'RPD':>4} {'IT':>4}")
    print("-"*58)
    total_rpd = 0
    total_it = 0
    for r in rows3:
        rpd = r["rpd_skill_count"] or 0
        its = r["it_skill_count"] or 0
        total_rpd += rpd
        total_it += its
        if rpd > 0 or True:
            print(f"{r['disc'][:48]:<48} {rpd:>4} {its:>4}")
    print(f"\nTotal: RPD={total_rpd}, IT={total_it}")

    # Also check which disciplines have skills from RPD at all
    rpd_disciplines = [r for r in rows3 if (r["rpd_skill_count"] or 0) > 0]
    print(f"\nDisciplines with RPD skills: {len(rpd_disciplines)}")
    for r in rpd_disciplines:
        print(f"  {r['disc'][:50]} ({r['rpd_skill_count']} RPD, {r['it_skill_count']} IT)")

    # Check all sources in skills table
    rows4 = await pool.fetch("SELECT DISTINCT source, COUNT(*) FROM skills GROUP BY source ORDER BY source")
    print(f"\nSkills by source:")
    for r in rows4:
        print(f"  {r['source']}: {r['count']}")

    # Check actual skill sources for Иностранный язык для деловой коммуникации
    rows5 = await pool.fetch("""
        SELECT DISTINCT s.name, s.source
        FROM competencies c
        JOIN competency_skills cs ON cs.competency_id = c.id
        JOIN skills s ON s.id = cs.skill_id
        JOIN disciplines d2 ON d2.id = c.discipline_id
        WHERE d2.name ILIKE $1
        ORDER BY s.name
    """, "%деловой%")
    print(f"\nSkills in Иностранный язык для деловой коммуникации:")
    for r in rows5:
        print(f"  {r['name']} (source={r['source']})")

    # Check KSA text for that discipline that could match linux
    rows6 = await pool.fetch("""
        SELECT k.original_text
        FROM competencies c
        JOIN ksa_entries k ON k.competency_id = c.id
        JOIN disciplines d2 ON d2.id = c.discipline_id
        WHERE d2.name ILIKE $1
    """, "%деловой%")
    print(f"\nKSA text in Иностранный язык для деловой коммуникации:")
    for r in rows6:
        txt = r["original_text"]
        match_linux = "linux" in txt.lower()
        print(f"  {'[LINUX]' if match_linux else '       '} {txt[:80]}")

    await close_pool()

asyncio.run(main())
