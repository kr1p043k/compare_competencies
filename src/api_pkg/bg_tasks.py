"""Fire-and-forget tasks with a strong reference.

`asyncio.create_task`/`ensure_future` без сохранения ссылки: задача может быть
собрана GC до завершения (RUF006). `spawn` держит ссылку до завершения.
"""
from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_tasks: set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Запустить корутину в фоне, удерживая ссылку до её завершения."""
    task = asyncio.ensure_future(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
