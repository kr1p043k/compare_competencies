"""CLI для создания пользователей.

Usage:
    python scripts/create_user.py admin@edu.ru mypass --role teacher --name "Иван Иванов"
"""

import argparse
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select, text

from src.database import async_session_factory
from src.models.krm_models import User


async def create_user(email: str, password: str, role: str, full_name: str) -> None:
    async with async_session_factory() as session:
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none():
            print(f"User {email} already exists")
            return

        result = await session.execute(
            text("SELECT crypt(:pw, gen_salt('bf')) AS pw_hash"),
            {"pw": password},
        )
        pw_hash = result.scalar_one()

        user = User(email=email, password_hash=pw_hash, full_name=full_name, role=role)
        session.add(user)
        await session.commit()
        print(f"User created: {email} ({role})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a user")
    parser.add_argument("email", help="Email address")
    parser.add_argument("password", help="Password")
    parser.add_argument("--role", default="teacher", choices=["admin", "teacher"])
    parser.add_argument("--name", default="", help="Full name")
    args = parser.parse_args()

    name = args.name or args.email.split("@")[0]
    asyncio.run(create_user(args.email, args.password, args.role, name))


if __name__ == "__main__":
    main()
