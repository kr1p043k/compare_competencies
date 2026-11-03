"""Шина инвалидации кэшей между пакетами.

Зачем: роутеры держат в памяти вычисленные ответы (прогнозы, тренды), чтобы не
пересчитывать их на каждый запрос. Но после пересчёта пайплайна или записи
нового снимка кэш обязан сброситься - иначе пользователь до 10 минут видит
устаревшие цифры (было именно так: TTL 300-600 с без инвалидации).

Связь односторонняя и без импортов: `src/pipeline` только объявляет факт
изменения (`invalidate`), а `src/api_pkg` подписывается при импорте. Поэтому
нет циклической зависимости pipeline -> api_pkg.

Использование:
    # в роутере, на уровне модуля
    register("forecast", _clear_forecast_caches)

    # там, где данные изменились
    invalidate("forecast")
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

#: имя кэша -> функция сброса (без аргументов)
_INVALIDATORS: dict[str, list] = {}


def register(name: str, callback) -> None:
    """Подписать функцию сброса на событие `name`. Идемпотентно по (name, fn)."""
    callbacks = _INVALIDATORS.setdefault(name, [])
    if callback not in callbacks:
        callbacks.append(callback)


def invalidate(name: str) -> int:
    """Вызвать все сбросы для `name`. Возвращает число успешно сброшенных.

    Ошибка одного сброса не должна мешать остальным и не должна ломать запись
    в БД: логируем и идём дальше.
    """
    callbacks = list(_INVALIDATORS.get(name, ()))
    ok = 0
    for cb in callbacks:
        try:
            cb()
            ok += 1
        except Exception as exc:  # pragma: no cover - защитный путь
            logger.warning("cache_invalidate_failed", cache=name, error=str(exc))
    if callbacks:
        logger.info("cache_invalidated", cache=name, callbacks=len(callbacks), ok=ok)
    return ok


def invalidate_all() -> None:
    """Сбросить все зарегистрированные кэши (используется в тестах)."""
    for name in list(_INVALIDATORS):
        invalidate(name)


def registered() -> dict[str, int]:
    """Имена кэшей и число подписчиков (для диагностики и тестов)."""
    return {name: len(cbs) for name, cbs in _INVALIDATORS.items()}
