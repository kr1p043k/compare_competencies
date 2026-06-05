import asyncio, asyncpg

async def main():
    pool = await asyncpg.create_pool(
        "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
        min_size=1, max_size=2
    )
    if pool is None:
        print("POOL IS NONE")
        return
    row = await pool.fetchrow(
        "SELECT id, email, role, full_name, password_hash FROM users WHERE email = $1 AND is_active = true",
        "admin@compare-competencies.local",
    )
    if row is None:
        print("ROW IS NONE")
        return
    print(f"Row id type: {type(row['id']).__name__} = {row['id']}")
    match = await pool.fetchval(
        "SELECT password_hash = crypt($1, password_hash) FROM users WHERE id = $2",
        "admin", row["id"],
    )
    print(f"Match: {match!r} (type={type(match).__name__})")
    await pool.close()

asyncio.run(main())
