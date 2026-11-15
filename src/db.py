import asyncpg

from src.config import settings

DATABASE_URL = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
# Поля объявлены в Settings (см. комментарий там); .env теперь работает.
DB_POOL_MIN = settings.DB_POOL_MIN
DB_POOL_MAX = settings.DB_POOL_MAX

pool: asyncpg.Pool | None = None


async def create_pool() -> asyncpg.Pool:
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=settings.DB_POOL_MIN,
            max_size=settings.DB_POOL_MAX,
            command_timeout=settings.DB_COMMAND_TIMEOUT,
            timeout=settings.DB_POOL_ACQUIRE_TIMEOUT,
            max_inactive_connection_lifetime=settings.DB_POOL_MAX_INACTIVE_LIFETIME,
        )
    return pool


async def close_pool():
    global pool
    if pool is not None:
        await pool.close()
        pool = None


def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database pool not initialized")
    return pool
