"""Seed users from users.json into PostgreSQL via asyncpg.

Пароли — ТОЛЬКО из env (SEED_*_PASSWORD через поле password_env) или из
users.json (untracked, локальный). Известные дефолты (admin/teacher123/
student/...) отклоняются fail-closed. См. users.example.json.
"""
import asyncio
import json
import os
from pathlib import Path

import asyncpg

BLOCKED_PASSWORDS = {
    "admin", "teacher123", "teacher", "student", "student123",
    "password", "password123", "123456", "qwerty", "",
}


def _resolve_password(email: str, info: dict) -> str | None:
    env_key = info.get("password_env", "")
    if env_key and os.environ.get(env_key):
        return os.environ[env_key]
    pw = info.get("password", "")
    if not pw or pw.strip().lower() in BLOCKED_PASSWORDS:
        return None
    return pw


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
            pw = _resolve_password(email, info)
            if pw is None:
                print(f"  REFUSED {email} — set {info.get('password_env', 'a real password')} "
                      f"(defaults blocked)")
                continue
            pw_hash = await conn.fetchval("SELECT crypt($1, gen_salt('bf'))", pw)
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
