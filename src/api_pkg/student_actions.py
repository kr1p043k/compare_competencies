"""Structured student action log (analysis runs, searches, etc.) - persisted in PostgreSQL."""
import asyncio
from datetime import datetime, timezone
from typing import Any

MAX_ACTIONS = 500

_action_buffer: list[dict[str, Any]] = []


async def _get_pool():
    from src.db import pool as _db_pool
    return _db_pool


async def _ensure_table():
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS student_actions (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                action_type TEXT NOT NULL DEFAULT '',
                profession TEXT DEFAULT '',
                region TEXT DEFAULT '',
                vacancies_found INTEGER DEFAULT 0,
                result_ref TEXT DEFAULT '',
                profile TEXT DEFAULT '',
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)


def log_action(username: str, action_type: str, profession: str = "",
               region: str = "", vacancies_found: int = 0,
               result_ref: str = "", profile: str = "") -> None:
    entry = {
        "username": username, "action_type": action_type,
        "profession": profession, "region": region,
        "vacancies_found": vacancies_found, "result_ref": result_ref,
        "profile": profile,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _action_buffer.append(entry)
    if len(_action_buffer) >= 10:
        asyncio.ensure_future(_flush_buffer())


async def _flush_buffer():
    entries = _action_buffer[:]
    _action_buffer.clear()
    pool = await _get_pool()
    if pool is None:
        _action_buffer.extend(entries)
        return
    try:
        async with pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO student_actions
                   (username, action_type, profession, region, vacancies_found, result_ref, profile)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                [(e["username"], e["action_type"], e["profession"],
                  e["region"], e["vacancies_found"], e["result_ref"], e["profile"])
                 for e in entries]
            )
    except Exception:
        _action_buffer.extend(entries)


def get_actions(username: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    try:
        import asyncio
        pool_future = asyncio.run_coroutine_threadsafe(
            _get_pool(), asyncio.get_event_loop()
        )
        pool = pool_future.result(timeout=2)
        if pool is None:
            return _get_from_buffer(username, limit)
        coro = _query_db(username, limit)
        future = asyncio.run_coroutine_threadsafe(coro, asyncio.get_event_loop())
        return future.result(timeout=5)
    except Exception:
        return _get_from_buffer(username, limit)


async def _query_db(username: str | None, limit: int) -> list[dict[str, Any]]:
    pool = await _get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        if username:
            rows = await conn.fetch(
                """SELECT * FROM student_actions
                   WHERE username = $1
                   ORDER BY timestamp DESC LIMIT $2""",
                username, limit
            )
        else:
            rows = await conn.fetch(
                """SELECT * FROM student_actions
                   ORDER BY timestamp DESC LIMIT $1""",
                limit
            )
    return [
        {
            "timestamp": r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"]),
            "action_type": r["action_type"], "profession": r["profession"],
            "region": r["region"], "vacancies_found": r["vacancies_found"],
            "result_ref": r["result_ref"], "profile": r["profile"],
        }
        for r in rows
    ]


def _get_from_buffer(username: str | None, limit: int) -> list[dict[str, Any]]:
    entries = list(_action_buffer)
    if username:
        entries = [e for e in entries if e["username"] == username]
    return entries[-limit:]
