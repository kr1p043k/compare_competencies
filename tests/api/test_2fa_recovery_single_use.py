"""Recovery-код 2FA гасится ровно один раз.

Проверка атомарности: код изымается условным UPDATE (recovery_codes @> код).
Если строку уже погасил параллельный запрос, UPDATE вернёт 0 строк, и код
должен считаться неверным, а не погаситься повторно.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

import src.api_pkg.routers.auth as auth_mod

CODE = "abcdef0123456789"


class _Pool:
    """Пул, отдающий заданный rowcount для условного UPDATE recovery-кода."""

    def __init__(self, consumed_rows: str) -> None:
        self.consumed_rows = consumed_rows
        self.recovery_updates = 0

    async def execute(self, sql: str, *_args: Any) -> str:
        if "recovery_codes @>" in sql:
            self.recovery_updates += 1
            return f"UPDATE {self.consumed_rows}"
        return "UPDATE 0"


def _state_with_code() -> dict[str, Any]:
    return {
        "secret": "not-a-valid-fernet-token",
        "confirmed": True,
        "recovery_codes": [auth_mod._hash_recovery_code(CODE)],
        "failed_attempts": 0,
    }


@pytest.mark.asyncio
async def test_present_code_is_accepted_and_consumed() -> None:
    pool = _Pool(consumed_rows="1")
    # Не должно бросать: код найден и успешно изъят (1 строка).
    await auth_mod._check_totp_or_recovery(pool, "uid-1", _state_with_code(), CODE)
    assert pool.recovery_updates == 1


@pytest.mark.asyncio
async def test_already_consumed_code_is_rejected() -> None:
    pool = _Pool(consumed_rows="0")
    # Условный UPDATE не нашёл код (его погасил другой запрос) -> отказ.
    with pytest.raises(HTTPException) as exc:
        await auth_mod._check_totp_or_recovery(pool, "uid-1", _state_with_code(), CODE)
    assert exc.value.status_code == 400
    assert pool.recovery_updates == 1


@pytest.mark.asyncio
async def test_unknown_code_never_touches_recovery_row() -> None:
    pool = _Pool(consumed_rows="1")
    with pytest.raises(HTTPException) as exc:
        await auth_mod._check_totp_or_recovery(pool, "uid-1", _state_with_code(), "0000000000000000")
    assert exc.value.status_code == 400
    assert pool.recovery_updates == 0
