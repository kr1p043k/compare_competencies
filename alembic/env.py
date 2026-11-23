"""Настройки Alembic для асинхронного PostgreSQL."""

import asyncio
from logging.config import fileConfig

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Модели должны быть импортированы, иначе Base.metadata пуста и autogenerate/
# `alembic check` видит «дрейф на удаление всех таблиц» (ложное срабатывание).
from alembic import context
from src.config import settings
from src.database import Base
from src.models import extra_models, krm_models  # noqa: F401  регистрирует ORM-таблицы в Base.metadata

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    """Фильтр сравнения для `alembic check`.

    Индексы и уникальные/FK-ограничения управляются миграциями: в БД имена
    (`idx_*`, `*_key`, сокращения) расходятся с именами, которые генерирует
    SQLAlchemy (`ix_*`, `uq_*`), при том что набор колонок совпадает. Их сравнение
    отключено, чтобы гейт не падал на переименованиях (backlog 6.3.2).
    Таблицы, колонки и типы по-прежнему сравниваются.

    Исключение - таблицы RAG: они создаются сырым DDL в миграции
    `add_rag_tables` (HNSW-индекс pgvector, GIN по tsv, партиционирование
    поколений), ORM-моделей у них нет намеренно, иначе пришлось бы дублировать
    типы `vector`/`tsvector` в SQLAlchemy-моделях ради одного гейта.
    """
    if type_ in ("index", "unique_constraint", "foreign_key_constraint"):
        return False
    return not (type_ == "table" and (name or "").startswith("rag_"))


def run_migrations_offline():
    """Запуск миграций в офлайн-режиме (без подключения к БД)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, include_object=include_object
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Выполнение миграций через синхронное соединение."""
    context.configure(connection=connection, target_metadata=target_metadata, include_object=include_object)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    """Запуск миграций в асинхронном режиме."""
    connectable = create_async_engine(settings.DATABASE_URL)
    async with connectable.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(do_run_migrations)


def run_migrations_online():
    """Точка входа для онлайн-миграций."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
