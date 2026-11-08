# Админ-консоль: API, аудит и эксплуатация

Справочник для администраторов и дежурного: эндпоинты админ-консоли, их эффекты,
записи в аудит, срок хранения и риски. Пути - с префиксом `/api`, `{id}` - UUID.
Код: `routers/admin.py`, `routers/auth.py`, `routers/audit.py`, `request_logger.py`.

## Новые и изменённые эндпоинты

| Метод | Путь | Роль | Побочный эффект | Аудит |
|---|---|---|---|---|
| POST | /auth/login | public | токен сессии либо 2FA-challenge | нет |
| POST | /auth/2fa/setup | admin | создать секрет TOTP | auth.2fa.setup |
| POST | /auth/2fa/confirm | admin | включить 2FA, вернуть 10 кодов | auth.2fa.enabled |
| POST | /auth/2fa/verify | admin | проверить код, выдать токен | auth.2fa.login |
| POST | /auth/2fa/recovery | admin | вход по recovery-коду (одноразовый) | auth.2fa.recovery_used |
| POST | /auth/2fa/disable | admin | отключить 2FA | auth.2fa.disabled |
| GET | /auth/2fa/status | admin | состояние 2FA текущего админа (мастер-флаг + личное подключение) | нет |
| GET | /auth/2fa/policy | public | статус 2FA (вкл/выкл) — показывать ли шаг кода на экране входа | нет |
| GET | /admin/security/2fa/settings | admin | состояние мастер-переключателя 2FA | нет |
| PUT | /admin/security/2fa/settings | admin | включить/выключить 2FA | admin.2fa.toggle |
| GET | /admin/users | admin | чтение списка (q, role, is_active) | нет |
| POST | /admin/users/create | admin | создать пользователя | admin.user.create |
| GET | /admin/users/{id} | admin | карточка: поля, сессии, 2FA | нет |
| PATCH | /admin/users/{id} | admin | роль/статус/имя; step-up + advisory-lock | admin.user.update |
| POST | /admin/users/{id}/password | admin | новый пароль, отзыв чужих сессий | admin.user.password_reset |
| POST | /admin/users/{id}/sessions/revoke | admin | отозвать открытые сессии | admin.user.sessions_revoke |
| POST | /admin/users/{id}/2fa/reset | admin | удалить admin_totp | admin.user.2fa_reset |
| PUT | /admin/users/{id}/directions | admin | заменить направления РОП | admin.user.directions |
| GET | /admin/sessions | admin | чтение сессий (user, limit, offset) | нет |
| GET | /admin/audit | admin | чтение журнала (фильтры, пагинация) | нет |
| GET | /admin/audit/actions | admin | уникальные action для UI | нет |
| POST | /admin/db/seed | admin | наполнить БД (фон) | admin.db.seed |
| POST | /admin/db/backup | admin | бэкап БД | admin.db.backup |
| GET | /admin/export/db | admin | выгрузка БД (фон) | admin.db.export |
| POST | /admin/embeddings/generate | admin | генерация эмбеддингов (фон) | admin.embeddings.generate |
| POST | /admin/students/import | admin | импорт студентов | admin.students.import |
| GET | /admin/export/excel | admin | выгрузка в Excel | admin.export.excel |
| GET | /admin/export/full-report | admin | полный отчёт | admin.export.full-report |
| POST | /admin/whitelist/backup | admin | бэкап it_skills.json | admin.skills.whitelist.backup |
| POST | /admin/skills/extend | admin | расширить whitelist (фон) | skills.extend.start |
| POST | /admin/skills/categorize | admin | правка таксономии навыков | skills.categorize |
| PUT | /admin/scheduler/settings | admin | сохранить настройки планировщика | admin.scheduler.settings |
| POST | /admin/scheduler/collect/now | admin | ручной сбор, 409 если занят | admin.scheduler.collect |
| POST | /admin/scheduler/gap/now | admin | ручной gap-анализ, 409 если занят | admin.scheduler.gap |
| GET | /admin/features | admin | legacy: JSON-переключатели функций, не используются для 2FA | нет |

## Дополнительные эндпоинты

### Список студентов

**`GET /admin/students`** — список профилей студентов из `data/students/*.json`
(`routers/admin.py:222`). Возвращает `filename`, `skills_count` и `modified`
(ISO 8601) для каждого файла; `total`. Если директория не существует — пустой
массив. Роль: `admin`; rate limit — 30/мин.

```bash
curl -s "$BASE/admin/students" -H "Authorization: Bearer $TOKEN"
```

### Направления KRM

**`GET /admin/directions`** — список всех направлений (`dir_code`) из KRM-файлов,
используется для привязки РОП (`routers/admin.py:515`). Вызывает
`_list_krm_directions()` из `routers/teacher.py:148`. Без параметров, rate limit —
30/мин.

```bash
curl -s "$BASE/admin/directions" -H "Authorization: Bearer $TOKEN"
```

### Логи запросов

**`GET /admin/logs`** — объединённые логи (in-memory буфер + flushed-строки из
PostgreSQL) `routers/admin.py:1179`. Параметры:

| Параметр | Тип | По умолч. | Описание |
|---|---|---|---|
| `user` | str \|\| None | None | Фильтр по email; `"all"` = все пользователи |
| `limit` | int | 100 | Макс. кол-во (1-1000) |
| `action` | str \|\| None | None | Префикс-фильтр по audit-действию (напр. `krm.recommendation.add`) |

Возвращает `logs` (массив объектов с `method`, `path`, `status`, `duration_ms`,
`user`, `source`, `detail`, `timestamp`) и `total`. Только для `admin`.
Rate limit — 30/мин.

```bash
# логи за последний час по конкретному пользователю
curl -s "$BASE/admin/logs?user=ivanov@sfedu.ru&limit=50" \
  -H "Authorization: Bearer $TOKEN"

# audit-строки по префиксу действия
curl -s "$BASE/admin/logs?action=pipeline.trigger&limit=100" \
  -H "Authorization: Bearer $TOKEN"
```

Формат `detail` в AUDIT-строках: `action | target | by <email> (<role>)`.

### Системный лог (tail)

**`GET /admin/logs/file`** — хвост файла `logs/backend.log` `routers/admin.py:1213`.
Параметры:

| Параметр | Тип | По умолч. | Описание |
|---|---|---|---|
| `lines` | int | 300 | Кол-во строк хвоста (1-5000) |
| `level` | str | "all" | Уровень: `error`, `warning`, `info`, `debug` |

Возвращает `file` (путь), `lines` (массив строк JSON-логи), `total`. Фильтр
по уровню — по подстроке `level=<уровень>` или префиксу строки. Только `admin`.

```bash
# последние 100 строк только error
curl -s "$BASE/admin/logs/file?lines=100&level=error" \
  -H "Authorization: Bearer $TOKEN"
```

### Real-time стрим логов (WebSocket)

**`WS /admin/logs/ws`** — стрим `logs/backend.log` в реальном времени
(`routers/admin.py:1601`, роутер `ws_log_router`, tags `logs-ws`).

Получает события JSON:

| `type` | Описание |
|---|---|
| `init` | начальный хвост 400 строк (массив `lines`) |
| `lines` | новые строки (пакет до 200 шт.) |
| `rotated` | файл вращён — отправлен весь текущий хвост |
| `error` | ошибка стриминга |

Авторизация — через cookie `token` или заголовок `Authorization: Bearer <token>`.
Только роль `admin` (иначе 4003). Проверка сессии через `resolve_session` +
валидация origin. Интервал опроса — 0.7 с.

```javascript
// клиентский WebSocket
const ws = new WebSocket(
  'ws://localhost:8000/api/admin/logs/ws',
  { headers: { Authorization: `Bearer ${TOKEN}` } }
);
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  console.log(msg.type, msg.lines?.length);
};
```

### Мониторинг (метрики Prometheus)

**`GET /admin/monitoring`** — JSON-мониторинг семейств метрик Prometheus
`routers/admin.py:1025`. Без параметров отдавал ВСЕ семейства (~435 строк и
растёт); теперь с фильтрацией:

| Параметр | Тип | По умолч. | Описание |
|---|---|---|---|
| `names` | str | "" | Фильтр по именам метрик (через запятую) |
| `limit` | int | 100 | Страница (1-1000) |
| `offset` | int | 0 | Смещение страницы |

Возвращает `total` (число семейств), `limit`, `offset`, `families` (объект с
`name`, `type`, `samples`). Без параметров — все семейства.

```bash
# латентность по endpoint-ам
curl -s "$BASE/admin/monitoring?names=api_request_duration_seconds_bucket" \
  -H "Authorization: Bearer $TOKEN"
```

### Сводка здоровья (dashboard summary)

**`GET /admin/monitoring/summary`** — сводка для панели здоровья
`routers/admin.py:1129`. Собирает из Prometheus и БД:

| Поле | Источник | Описание |
|---|---|---|
| `latency_p95_by_endpoint` | Prometheus histograms | p95 по топу-5 endpoint-ов |
| `errors_1h` | `request_logs` | кол-во 4xx и 5xx за 1 час |
| `sessions_active` | `sessions` WHERE logged_out_at IS NULL AND last_activity > 15m | |
| `sessions_total_open` | `sessions` WHERE logged_out_at IS NULL | |
| `recent_runs` | `pipeline_runs` ORDER BY started_at DESC LIMIT 10 | последние прогоны |
| `freshness` | `vacancies` MIN/MAX published_at | свежесть данных вакансий |

Rate limit — 30/мин. При недоступной БД или Prometheus — секция пропускается.

```bash
curl -s "$BASE/admin/monitoring/summary" -H "Authorization: Bearer $TOKEN"
```

## Аудит-действия

Строки ниже - реальные значения колонки `action` (проверяются префиксом в
`GET /admin/audit?action=...`). Полный перечень также отдаёт `GET /admin/audit/actions`.

- Пользователи: `admin.user.create`, `admin.user.update`, `admin.user.password_reset`,
  `admin.user.sessions_revoke`, `admin.user.2fa_reset`, `admin.user.directions`.
- БД и выгрузки: `admin.db.seed`, `admin.db.backup`, `admin.db.export`,
  `admin.embeddings.generate`, `admin.students.import`, `admin.export.excel`,
  `admin.export.full-report`.
- Планировщик: `admin.scheduler.settings`, `admin.scheduler.collect`, `admin.scheduler.gap`.
- Навыки: `admin.skills.whitelist.backup`, `skills.whitelist.add`, `skills.whitelist.remove`,
  `skills.extend.start`, `skills.categorize`, `admin.skills.suggestion.approve`,
  `admin.skills.suggestion.reject`.
- 2FA: `auth.2fa.setup`, `auth.2fa.enabled`, `auth.2fa.login`, `auth.2fa.disabled`,
  `auth.2fa.recovery_used`, `admin.2fa.toggle`.

Формат строки: `action | target | by <email> (<role>)`; из `target` берутся
`target_type`/`target_id` (`uid=`/`email=` дают `target_type=user`).

## Хранение, миграция, индексы

- Миграция `alembic/versions/add_audit_logs.py`, `down_revision = app_settings`.
- Таблица `audit_logs`: `actor_id/email/role`, `action`, `target_type/id`, `target`,
  `meta` (JSONB), `ip`, `request_id`, `created_at`.
- Индексы: `idx_audit_created`, `idx_audit_action_created`, `idx_audit_actor`,
  `idx_audit_target`, автоиндекс `actor_id`.
- 2FA мастер-флаг хранится в таблице `app_settings` (ключ `security.two_factor`,
  значение JSONB `{"enabled": boolean}`), обновляется через `updated_at` и
  `updated_by`. Миграция таблицы - `alembic/versions/app_settings_table.py`
  (`add_audit_logs` идёт после неё); дополнительно есть рантайм-страховка
  `_ensure_table` (CREATE TABLE IF NOT EXISTS). Чтение мастера fail-open:
  любая ошибка БД = `False`, то есть при сбое чтения флага доступ по паролю
  не блокируется - это осознанный размен, но его стоит поверхностно проверять в мониторинге.

## Retention-CLI: чистка устаревших данных

Модуль `src/cli/retention.py` удаляет устаревшие строки из трёх таблиц БД.

Запуск:

```bash
# dry-run: показать, что БУДЕТ удалено
python -m src.cli retention

# реально удалить (обязательно --apply, одна транзакция)
python -m src.cli retention --apply

# кастомные сроки
python -m src.cli retention --vacancies-days 730 --logs-days 90 --apply
```

Параметры:

| Параметр | По умолч. | Таблица | Описание |
|---|---|---|---|
| `--vacancies-days` | 730 | `vacancies` | Удалить по `published_at < NOW() - interval` |
| `--logs-days` | 90 | `request_logs` | Удалить по `created_at < NOW() - interval` |
| `--run-days` | 365 | `pipeline_runs` | Удалить по `started_at < NOW() - interval` |

Важно: вакансии удаляются только те, что НЕ участвуют в последних прогонах (чтобы
не сломать сравнение с историей). Все три удаления в одной транзакции `psycopg2` —
либо всё commitится, либо всё откатывается. Значения < 1 отклоняются с кодом 2.

## Контракт self-healing audit_logs

Система аудита — `request_logger.py` — реализует контракт самовосстановления:
при недоступности БД данные не теряются.

### In-memory буфер

- `_log_buffer: deque[LogEntry]` — кольцевой буфер maxlen=2000 (`request_logger.py:108`).
- `LogEntry` содержит: `method`, `path`, `status`, `duration_ms`, `user_email`,
  `source`, `detail`, `timestamp` (UTC, tz-aware).
- `FLUSH_INTERVAL = 10` сек, `FLUSH_BATCH = 100` — фоновый `_periodic_flush()`
  (запускается через `start_log_flusher()` при старте сервера).

### Flush с откатом

`_flush_to_db()` (request_logger.py:127) вставляет порциями по 200 строк
(лимит параметров драйвера). При ошибке вставки — запись откатывается в начало
буфера (`_log_buffer.appendleft`), а не теряется (request_logger.py:157-162).
Если колонка `detail` отсутствует (миграция не применена) — выполняется slim-вставка.

### Merge buffer + DB

`get_logs_merged()` (request_logger.py:228) объединяет:
1. in-memory буфер (`get_logs()` — последние 2000 записей).
2. flushed-строки из PostgreSQL (`RequestLog`, ORDER BY created_at DESC, limit=limit*2).
3. Сортировка по `timestamp` DESC, срез по `limit`.

При недоступной БД — возвращается только буфер без предупреждения клиенту
(request_logger.py:257).

### Email-маскирование

Полный email хранится в буфере и БД (нужен для атрибуции действий).
В структурированные логи (файлы, сборщики) email маскируется: `ivanov@sfedu.ru`
→ `i***@sfedu.ru` (`mask_email()`, request_logger.py:33-47).

### Аудит-действия

`audit_action()` (request_logger.py:296) записывает AUDIT-строку в буфер
после успешной мутации. Формат `detail`: `action | target | by <email> (<role>)`.
При ошибке — `anonymous`, `?`.

## Операционные риски

- Step-up: смена роли/статуса, сброс пароля и сброс 2FA другого администратора
  требуют `caller_password` в теле (`_assert_caller_password`, 403 при отсутствии
  или неверном пароле). Для не-админов подтверждение не нужно.
- Понижение/деактивация админа выполняется под
  `pg_advisory_xact_lock('admin_guard')`: нельзя оставить систему без
  активного администратора (409).
- Пока миграция не применена, запись в `audit_logs` отключена флагом
  `_audit_table_missing` (`request_logger.py`): события живут только в буфере
  `request_logs`. Сбои записи логируются на уровне ERROR.
- Аудит best-effort: при недоступной БД ответ эндпоинта не ломается, durable-строки
  может не быть.

## Примеры curl

```bash
BASE=http://localhost:8000/api

# вход -> токен (или {"status":"totp_required","challenge_id":"..."})
TOKEN=$(curl -s -X POST $BASE/auth/login -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"secret"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

# список студентов
curl -s "$BASE/admin/students" -H "Authorization: Bearer $TOKEN"

# список KRM-направлений
curl -s "$BASE/admin/directions" -H "Authorization: Bearer $TOKEN"

# логи запросов по префиксу действия
curl -s "$BASE/admin/logs?action=pipeline.trigger&limit=50" \
  -H "Authorization: Bearer $TOKEN"

# системный лог: 200 строк, только error
curl -s "$BASE/admin/logs/file?lines=200&level=error" \
  -H "Authorization: Bearer $TOKEN"

# метрики: фильтр по имени + пагинация
curl -s "$BASE/admin/monitoring?names=api_request_duration_seconds_bucket&limit=50" \
  -H "Authorization: Bearer $TOKEN"

# сводка здоровья
curl -s "$BASE/admin/monitoring/summary" \
  -H "Authorization: Bearer $TOKEN"

# список пользователей
curl -s "$BASE/admin/users?q=ivan&limit=25" -H "Authorization: Bearer $TOKEN"

# изменить пользователя (caller_password нужен только для другого админа)
curl -s -X PATCH $BASE/admin/users/<UID> -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"role":"teacher","caller_password":"secret"}'

# журнал аудита: префикс действия и период
curl -s "$BASE/admin/audit?action=admin.user&since=2026-01-01T00:00:00Z" \
  -H "Authorization: Bearer $TOKEN"

# отозвать сессии пользователя
curl -s -X POST $BASE/admin/users/<UID>/sessions/revoke \
  -H "Authorization: Bearer $TOKEN"

# сброс 2FA другого администратора (step-up)
curl -s -X POST $BASE/admin/users/<UID>/2fa/reset -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"caller_password":"secret"}'

# dry-run retention
python -m src.cli retention

# применить retention
python -m src.cli retention --apply
```

## Известные ограничения и backlog v2

- Инциденты пока не собраны в отдельный экран: смотреть журнал аудита и
  уведомления (`src/notifications/system.py`, Telegram).
- Бэкапы БД только создаются: списка и восстановления из UI нет.
- Фоновые операции (seed, embeddings, export) сообщают только о старте: без
  прогресса и истории.
- Нет вкладки «Сессии»: общий список только по API, персональные - в карточке
  пользователя.