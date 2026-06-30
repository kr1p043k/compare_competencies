"""Seed users from users.json into PostgreSQL via asyncpg."""
import asyncio
import json
from pathlib import Path

import asyncpg


async def main():
    import os
    url = os.environ.get("DATABASE_URL", "postgresql://postgres:@localhost:5432/compare_competencies")
    conn = await asyncpg.connect(url)
    try:
        users_file = Path(__file__).parent / "users.json"
        with open(users_file, encoding="utf-8") as f:
            users = json.load(f)
        created = 0
        for email, info in users.items():
            row = await conn.fetchrow("SELECT id FROM users WHERE email = $1", email)
            if row:
                print(f"  SKIP {email} — already exists")
                continue
            pw_hash = await conn.fetchval("SELECT crypt($1, gen_salt('bf'))", info["password"])
            await conn.execute(
                "INSERT INTO users (email, password_hash, full_name, role, is_active) VALUES ($1, $2, $3, $4, true)",
                email, pw_hash, info.get("name", email.split("@")[0]), info["role"],
            )
            print(f"  CREATED {email} ({info['role']})")
            created += 1
        print(f"Done: {created} users created")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
