"""Сессионный токен уникален для каждого входа.

Без случайного nonce токен зависел только от (uid, почта, роль, секунда), и
два входа одного пользователя в одну секунду давали один и тот же токен -
одну сессию на всех. Тогда выход на одном устройстве выкидывал второе, а
отзыв сессий (например, при включении 2FA) не различал их.
"""
from __future__ import annotations

import src.api_pkg.routers.auth as auth_mod


def test_same_user_same_second_gets_distinct_tokens() -> None:
    first = auth_mod._make_token("uid-1", "user@example.test", "admin")
    second = auth_mod._make_token("uid-1", "user@example.test", "admin")
    assert first != second, "две сессии не должны совпадать"

    decoded = auth_mod._decode_token(first)
    assert decoded is not None
    assert decoded["uid"] == "uid-1"
    assert decoded["r"] == "admin"
