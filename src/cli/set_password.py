"""Смена пароля существующего пользователя.

Нужна для эксплуатации: админский аккаунт с дефолтным паролем - это дыра,
а CLI умел только создавать пользователей.

Usage:
    python -m src.cli set-password admin@example.com 'новый-пароль'
    python -m src.cli set-password admin@example.com --generate

Генерация печатает пароль один раз и сохраняет его в .secrets/, чтобы не
держать секрет в истории командной строки.
"""

from __future__ import annotations

import contextlib
import secrets
import string
import sys
from pathlib import Path

from sqlalchemy import text as sa_text

from src.database import async_session_factory

#: Без визуально неоднозначных символов (l/1/I, O/0), чтобы пароль можно было
#: продиктовать по телефону или внести руками.
_ALPHABET = (
    "".join(c for c in string.ascii_letters if c not in "lIOo")
    + "".join(c for c in string.digits if c not in "01")
    + "!@#$%^&*-_=+"
)


def generate_password(length: int = 24) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def _save_secret(email: str, password: str) -> Path:
    """Сохранить пароль локально (каталог .secrets в .gitignore)."""
    root = Path(__file__).resolve().parents[2]
    out_dir = root / ".secrets"
    out_dir.mkdir(exist_ok=True)
    name = email.replace("@", "-at-").replace(".", "-")
    path = out_dir / f"password-{name}.txt"
    path.write_text(
        f"# Пароль для {email}\n# Сгенерирован src.cli set-password\n{password}\n",
        encoding="utf-8",
    )
    return path


async def main(email: str, password: str | None, generate: bool = False) -> int:
    """Установить пароль. Возвращает 0 при успехе, 1 если пользователь не найден."""
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    if generate and not password:
        password = generate_password()
    if not password:
        print("Укажите пароль или --generate")
        return 2

    async with async_session_factory() as session:
        found = await session.execute(
            sa_text("SELECT id, role FROM users WHERE lower(email) = lower(:e)"),
            {"e": email},
        )
        row = found.first()
        if row is None:
            print(f"Пользователь не найден: {email}")
            return 1

        # bcrypt cost 12 - как в остальном проекте (логин перехеширует cost 6).
        await session.execute(
            sa_text(
                "UPDATE users SET password_hash = crypt(:pw, gen_salt('bf', 12)),"
                " failed_login_count = 0, locked_until = NULL,"
                " last_failed_login_at = NULL"
                " WHERE lower(email) = lower(:e)"
            ),
            {"pw": password, "e": email},
        )
        await session.commit()

    print(f"Пароль обновлён: {email} (роль {row.role})")
    if generate:
        path = _save_secret(email, password)
        print(f"Пароль: {password}")
        print(f"Сохранён: {path} (вне git)")
    return 0


def build_parser(sub) -> None:
    p = sub.add_parser("set-password", help="Сменить пароль пользователя")
    p.add_argument("email")
    p.add_argument("password", nargs="?", default=None)
    p.add_argument(
        "--generate",
        action="store_true",
        help="Сгенерировать пароль и сохранить в .secrets (не передавать в argv)",
    )
    p.set_defaults(func=lambda a: _run(a))


def _run(args) -> None:
    import asyncio

    raise SystemExit(
        asyncio.run(main(args.email, args.password, generate=args.generate))
    )
