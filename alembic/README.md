# Alembic — управление миграциями базы данных

Alembic используется для управления изменениями схемы PostgreSQL в проекте **Competency Platform**.

---

# Содержание

* [Быстрый старт](#-быстрый-старт)
* [Структура миграций](#-структура-миграций)
* [Основные команды](#-основные-команды)
* [Работа с миграциями](#-работа-с-миграциями)
* [Устранение проблем](#-устранение-проблем)
* [Лучшие практики](#-лучшие-практики)
* [Production-деплой](#-production-деплой)
* [Интеграция с CI/CD](#-интеграция-с-cicd)
* [Мониторинг миграций](#-мониторинг-миграций)

---

# Быстрый старт

## Локальная разработка

### 1. Установка зависимостей

```bash
pip install alembic
```

### 2. Инициализация Alembic

```bash
alembic init alembic
```

### 3. Настройка подключения к БД

Отредактируйте `alembic.ini`:

```ini
sqlalchemy.url = postgresql+asyncpg://user:pass@localhost:5432/dbname
```

### 4. Создание миграции

```bash
alembic revision -m "description"
```

### 5. Применение миграций

```bash
alembic upgrade head
```

### 6. Откат последней миграции

```bash
alembic downgrade -1
```

---

## Работа внутри Docker-контейнера

### Подключение к backend-контейнеру

```bash
docker exec -it compare-competencies-api bash
```

### Проверка текущего состояния

```bash
alembic current
```

### Применение всех миграций

```bash
alembic upgrade head
```

### Просмотр истории

```bash
alembic history
```

---

# Структура миграций

```text
alembic/
├── versions/                    # Файлы миграций
│   ├── 90d02040d3d1_add_parsed_skills_to_vacancies.py
│   └── ...
├── env.py                       # Конфигурация Alembic (async support)
├── script.py.mako               # Шаблон новых миграций
└── README.md                    # Документация
```

---

# Основные команды

## Просмотр информации

### Текущая версия БД

```bash
alembic current
```

### История миграций

```bash
alembic history
```

### Посмотреть SQL без выполнения

```bash
alembic upgrade head --sql
```

### Детальный лог

```bash
alembic -v upgrade head
```

---

## Применение миграций

### До последней версии

```bash
alembic upgrade head
```

### На один шаг вперед

```bash
alembic upgrade +1
```

### До конкретной миграции

```bash
alembic upgrade 90d02040d3d1
```

---

## Откат миграций

### Откатить последнюю

```bash
alembic downgrade -1
```

### До определенной версии

```bash
alembic downgrade 90d02040d3d1
```

### Полный откат

```bash
alembic downgrade base
```

---

## Создание миграций

### Пустая миграция

```bash
alembic revision -m "add_new_table"
```

### Автоматическая миграция

```bash
alembic revision --autogenerate -m "auto_migration"
```

### Миграция с собственным ID

```bash
alembic revision -m "description" --rev-id custom_id
```

---

#Работа с миграциями

## Создание новой миграции вручную

Создайте файл:

```bash
alembic revision -m "add_user_preferences_table"
```

### Пример миграции

```python
"""add user preferences table"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "90d02040d3d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("preferences", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
    )

    op.create_index(
        "idx_user_prefs_user",
        "user_preferences",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_user_prefs_user")
    op.drop_table("user_preferences")
```

---

## Автоматическая генерация миграций

### 1. Измените SQLAlchemy-модели

### 2. Сгенерируйте миграцию

```bash
alembic revision --autogenerate -m "update_models"
```

### 3. Проверьте сгенерированный код

### 4. Примените миграцию

```bash
alembic upgrade head
```

---

## Безопасные операции

### Добавление колонки с проверкой

```python
def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    columns = [
        col["name"]
        for col in inspector.get_columns("vacancies")
    ]

    if "parsed_skills" not in columns:
        op.add_column(
            "vacancies",
            sa.Column(
                "parsed_skills",
                sa.JSON(),
                nullable=True,
            ),
        )
```

---

### Создание индекса с проверкой

```python
def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    indexes = [
        idx["name"]
        for idx in inspector.get_indexes("vacancies")
    ]

    if "idx_vacancies_parsed_pub" not in indexes:
        op.create_index(
            "idx_vacancies_parsed_pub",
            "vacancies",
            ["published_at"],
            postgresql_where=sa.text(
                "parsed_skills IS NOT NULL"
            ),
        )
```

---

### Добавление данных

```python
def upgrade() -> None:
    op.execute(
        """
        INSERT INTO directions (
            code,
            name,
            profile,
            opop_year
        )
        VALUES (
            '09.03.03',
            'Прикладная информатика',
            'Бизнес-аналитика',
            2025
        )
        ON CONFLICT (code)
        DO NOTHING
        """
    )
```

---

# Устранение проблем

## Проблема №1. Миграция не применяется

### Проверить текущую версию

```bash
alembic current
```

### Принудительно установить состояние

```bash
alembic stamp head
```

### Выполнить полный пересозданный путь

```bash
alembic downgrade base
alembic upgrade head
```

---

## Проблема №2. Конфликт версий

### Просмотреть историю

```bash
alembic history
```

### Откатиться к стабильной версии

```bash
alembic downgrade <stable_revision>
```

### Создать новую миграцию

```bash
alembic revision -m "fix_conflict"
```

---

## Проблема №3. Ошибка при autogenerate

Убедитесь, что все модели импортированы в `env.py`:

```python
from src.database import Base

import src.models.krm_models

target_metadata = Base.metadata
```

---

## Проблема №4. Async PostgreSQL

```python
from sqlalchemy.ext.asyncio import create_async_engine


async def run_async_migrations():
    connectable = create_async_engine(
        settings.DATABASE_URL
    )

    async with connectable.connect() as connection:
        await connection.run_sync(
            do_run_migrations
        )
```

---

# Лучшие практики

## 1. Понятные названия миграций

### Хорошо

```text
add_user_preferences_table
add_parsed_skills_to_vacancies
update_competency_embedding_index
fix_foreign_key_cascade
```

### Плохо

```text
update
fix_bug
temp_migration
```

---

## 2. Проверка перед применением

### Сгенерировать SQL

```bash
alembic upgrade head --sql > migration.sql
```

### Проверить SQL

```bash
cat migration.sql | grep -E "CREATE|ALTER|DROP"
```

### Создать резервную копию

```bash
pg_dump -U user dbname > backup_$(date +%Y%m%d).sql
```

---

## 3. Идемпотентность

```python
def upgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if not inspector.has_table("new_table"):
        op.create_table("new_table", ...)

    columns = [
        col["name"]
        for col in inspector.get_columns("users")
    ]

    if "preferences" not in columns:
        op.add_column(
            "users",
            sa.Column(
                "preferences",
                sa.JSON(),
            ),
        )
```

---

## 4. Git Workflow

### Коммитьте миграции

```bash
git add alembic/versions/
git commit -m "Add migration: add_parsed_skills_to_vacancies"
```

### Не изменяйте уже примененные миграции

Плохая практика:

```text
Редактировать существующую миграцию
```

Правильный подход:

```text
Создать новую миграцию для исправлений
```

---

# Production-деплой

### docker-compose.prod.yml

```yaml
backend:
  command: >
    sh -c "
      alembic upgrade head &&
      uvicorn src.api_pkg:app
      --host 0.0.0.0
      --port 8000
    "
```

---

# Интеграция с CI/CD

## GitHub Actions

```yaml
- name: Run database migrations
  run: |
    docker exec -i competency-postgres psql \
      -U ${{ secrets.DB_USER }} \
      -d ${{ secrets.DB_NAME }} \
      < backup.sql

    docker exec -i compare-competencies-api \
      alembic upgrade head
```

---

## Pre-commit hook

Файл:

```bash
.git/hooks/pre-commit
```

```bash
#!/bin/bash

echo "Checking for unapplied migrations..."

alembic current > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Migration check failed"
    echo "Run 'alembic upgrade head'"
    exit 1
fi
```

---

# Мониторинг миграций

## История миграций в БД

```sql
SELECT *
FROM alembic_version;
```

## Список примененных миграций

```sql
SELECT *
FROM alembic_version
ORDER BY version_num DESC;
```
