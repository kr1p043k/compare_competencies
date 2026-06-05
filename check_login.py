import asyncio
from src.db import create_pool, close_pool, get_pool

async def main():
    await create_pool()
    pool = get_pool()
    
    # Check what passwords/hashes exist
    rows = await pool.fetch("SELECT email, password_hash, role FROM users ORDER BY email")
    for r in rows:
        h = r["password_hash"]
        print(f"{r['email']:45} role={r['role']:8} hash={h[:40] if h else 'NULL'}...")
    
    # Try verifying with crypt
    for email, pw in [("teacher@compare-competencies.local", "prepod"), ("admin@compare-competencies.local", "admin")]:
        row = await pool.fetchrow(
            "SELECT email, password_hash = crypt($1, password_hash) AS ok FROM users WHERE email = $2",
            pw, email
        )
        if row:
            print(f"\n{email} / '{pw}' → {'✅ OK' if row['ok'] else '❌ FAIL'}")
        else:
            print(f"\n{email} → NOT FOUND")
    
    await close_pool()

asyncio.run(main())
