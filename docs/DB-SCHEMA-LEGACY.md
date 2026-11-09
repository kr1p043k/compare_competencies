# Дрейф схемы и legacy-объекты БД

Проверено по `information_schema` живой БД, `alembic heads` и `git grep`
на 2026-11-09. Документ описывает **фактическое** состояние: часть объектов
из чернового списка legacy оказалась живой, это отмечено явно.

## 1. Дрейф версии миграций (подтверждён)

- `alembic heads` в коде: **`add_llm_cost`**.
- `alembic_version` в живой БД: **`add_chat_storage`** (отстаёт).

Следствие: в живой БД пока нет колонок из неприменённых ревизий:

| Объект | Откуда | В живой БД |
|---|---|---|
| `pipeline_runs.idempotency_key` | `add_llm_cost` | отсутствует |
| `llm_interactions.cost_usd` | `add_llm_cost` | отсутствует |

Лечение (отдельным согласованным шагом, только на **локальной** БД):

```bash
alembic upgrade head
```

Прод не трогать: схему прода меняет только оркестратор.

## 2. Объекты: проверенный статус

### Мёртвые (в коде нет ссылок)

| Объект | Проверка |
|---|---|
| таблица `saved_files` | `git grep` - 0 совпадений |
| таблица `timelog` | `git grep` - 0 совпадений |
| таблица `todos` | `git grep` - 0 совпадений |
| колонка `competency_skills.is_flat` | `git grep` - 0 совпадений |
| колонка `disciplines.direction_codes` | в коде только функция `_user_direction_codes`, сама колонка не читается |

> УДАЛЕНО миграцией `drop_legacy_dead_objects` (2026-11-09): таблицы `saved_files`, `timelog`, `todos` и колонки `competency_skills.is_flat`, `disciplines.direction_codes`. Перед удалением повторно проверено: `git grep` по src/tests/alembic/scripts/frontend - 0 ссылок; локальная БД обновлена (`alembic upgrade head`), `tests/api` 873 passed.


### Живые (НЕ legacy, вопреки черновому списку)

| Объект | Где используется |
|---|---|
| `admin_totp.last_totp_step` | `src/api_pkg/routers/auth.py:955` - защита TOTP от повторного использования кода; колонка создаётся кодом (auth.py:783,793) |
| `market_skill_mappings` (таблица) | читается опционально в `src/pipeline/teacher_analysis_runner.py:566-567` (веса спроса последнего периода, в try/except); пишется `scripts/snapshot_market_skills.py`; правится `scripts/find_skill_duplicates.py` |
| `ksa_entries.embedding` | заполняется `scripts/backfill_embeddings.py` |

**Противоречие для отдельного решения:** `sql/006_market_skill_mappings_drop.sql`
дропает `market_skill_mappings` с комментарием «0 Python-ссылок», но это уже
неверно - читатель и писатель есть. Либо их переносят на новый источник
(веса спроса), либо `sql/006` применять нельзя. Пока не согласовано -
таблицу не удалять.

### Кандидаты в deprecated (в БД есть, в коде не читаются)

| Объект | Проверка |
|---|---|
| `pipeline_runs.user_id` | в БД есть; владелец прогона теперь `owner_id`, колонка не используется |
| `ksa_entries.source` | в БД есть; прямой ссылки в коде нет (в отличие от `ksa_entries.embedding`, который пишет backfill) |

Удалять оба - отдельной миграцией после согласования.

## 3. Таблицы, создаваемые кодом при старте (не дрейф)

Отсутствуют в alembic по замыслу - их поднимает код:

- `app_settings` - `src/api_pkg/security_settings.py::_ensure_table`;
- `audit_logs`, `student_actions`, `admin_totp` - DDL в коде на старте.

На чистой БД `/ready` отдаёт 200 именно поэтому.

## 4. Политика

1. `data/reference/*.json` - поставляемые справочники; тесты их не пишут
   (снимок и проверка усадки в `tests/conftest.py`).
2. «Мёртвое» из раздела 2 удалять отдельной миграцией
   `refactor: drop legacy columns`, не в общей ветке и не «по пути».
3. Перед деплоем сверять `alembic heads` с версией в целевой БД.
4. Объекты из «Противоречия» и раздел «Не найдено» - сначала решение
   (перенос / deprecated-пометка), потом действие.
