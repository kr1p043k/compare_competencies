"""Шифрование секретов в состоянии покоя (SSO-токены в БД).

Ключ выводится из SECRET_KEY приложения: отдельного секрета не заводим, ротация
SECRET_KEY инвалидирует сохранённые SSO-токены (пользователь входит заново).
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from src import config


def _fernet() -> Fernet:
    digest = hashlib.sha256(config.get_secret_key().encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(value: str) -> str:
    """Шифрует строку для хранения в БД."""
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str | None) -> str | None:
    """Расшифровывает; None, если значение повреждено или записано открытым текстом."""
    if not value:
        return None
    try:
        return _fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeError):
        return None
