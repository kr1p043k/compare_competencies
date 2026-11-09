"""Tri-state чтения мастер-флага 2FA: enabled + read_ok.

Инвариант: ошибка чтения НЕ отключает 2FA. read_two_factor отдаёт
три-состояние (ok/empty/error), булев helper fail-closed: на error -> True,
чтобы сбой БД не снимал второй фактор.
"""

from __future__ import annotations

import weakref

import src.api_pkg.security_settings as ss


class _Pool:
    def __init__(self, *, raw=None, execute_fails: bool = False, fetch_fails: bool = False):
        self.raw = raw
        self.execute_fails = execute_fails
        self.fetch_fails = fetch_fails

    async def execute(self, _sql: str, *args):
        if self.execute_fails:
            raise RuntimeError("permission denied for schema public")

    async def fetchval(self, _sql: str, *args):
        if self.fetch_fails:
            raise RuntimeError("connection reset by peer")
        return self.raw


def _reset():
    ss._table_ready = False
    ss._ensure_locks = weakref.WeakKeyDictionary()


async def test_status_enabled_when_row_says_true():
    _reset()
    st = await ss.read_two_factor(_Pool(raw='{"enabled": true}'))  # type: ignore[arg-type]
    assert st.enabled is True
    assert st.read_ok is True
    assert st.state is ss.TwoFactorReadState.OK


async def test_status_disabled_but_read_ok_when_row_missing():
    _reset()
    st = await ss.read_two_factor(_Pool(raw=None))  # type: ignore[arg-type]
    # Строки нет - это осознанное "выключено", а не сбой чтения.
    assert st.enabled is False
    assert st.read_ok is True
    assert st.state is ss.TwoFactorReadState.EMPTY


async def test_status_read_not_ok_on_db_error():
    _reset()
    st = await ss.read_two_factor(_Pool(fetch_fails=True))  # type: ignore[arg-type]
    assert st.enabled is False
    assert st.read_ok is False
    assert st.state is ss.TwoFactorReadState.ERROR


async def test_status_read_not_ok_when_table_cannot_be_created():
    _reset()
    st = await ss.read_two_factor(_Pool(execute_fails=True))  # type: ignore[arg-type]
    assert st.enabled is False
    assert st.read_ok is False
    assert st.state is ss.TwoFactorReadState.ERROR


async def test_status_read_not_ok_on_malformed_value():
    _reset()
    st = await ss.read_two_factor(_Pool(raw="[1, 2, 3]"))  # type: ignore[arg-type]
    assert st.enabled is False
    assert st.read_ok is False
    assert st.state is ss.TwoFactorReadState.ERROR


async def test_enabled_helper_is_fail_closed_on_read_error():
    """Ошибка чтения настройки НЕ отключает 2FA: helper отвечает True.

    Раньше было fail-open (ошибка -> False -> второй фактор не спрашивался):
    сбой БД тихо снимал защиту. Теперь булев helper fail-closed.
    """
    _reset()
    assert await ss.get_two_factor_enabled(_Pool(raw='{"enabled": true}')) is True  # type: ignore[arg-type]
    assert await ss.get_two_factor_enabled(None) is False
    _reset()
    assert await ss.get_two_factor_enabled(_Pool(fetch_fails=True)) is True  # type: ignore[arg-type]
    _reset()
    assert await ss.get_two_factor_enabled(_Pool(raw="[1, 2, 3]")) is True  # type: ignore[arg-type]


async def test_enabled_helper_false_when_row_absent():
    """Нет строки настройки - это не ошибка чтения, 2FA выключена."""
    _reset()
    assert await ss.get_two_factor_enabled(_Pool(raw=None)) is False  # type: ignore[arg-type]
