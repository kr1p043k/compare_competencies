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
- Индексы: `idx_audit_created`, `idx_audit_action_created`, `idx_audit_actor`, `idx_audit_target`, автоиндекс `actor_id`.
- 2FA мастер-флаг хранится в таблице `app_settings` (ключ `security.two_factor`,
  значение JSONB `{"enabled": boolean}`), обновляется через `updated_at` и
  `updated_by`. Миграция таблицы - `alembic/versions/app_settings_table.py`
  (`add_audit_logs` идёт после неё); дополнительно есть рантайм-страховка
  `_ensure_table` (CREATE TABLE IF NOT EXISTS). Чтение мастера fail-open:
  любая ошибка БД = `False`, то есть при сбое чтения флага доступ по паролю
  не блокируется - это осознанный размен, но его стоит поверхностно проверять в мониторинге.
- Retention: `DEFAULT_AUDIT_DAYS = 730`; `python -m src.cli retention --audit-days 730`
  (dry-run; удаление с `--apply`). Шаг `audit_logs` пропускается без таблицы.

## Операционные риски

- Step-up: смена роли/статуса, сброс пароля и сброс 2FA другого администратора
  требуют `caller_password` в теле (`_assert_caller_password`, 403 при отсутствии
  или неверном пароле). Для не-админов подтверждение не нужно.
- Понижение/деактивация админа выполняется под `pg_advisory_xact_lock('admin_guard')`:
  нельзя оставить систему без активного администратора (409).
- Пока миграция не применена, запись в `audit_logs` отключена флагом
  `_audit_table_missing` (`request_logger.py`): события живут только в буфере
  `request_logs`. Сбои записи логируются на уровне ERROR.
- Аудит best-effort: при недоступной БД ответ эндпоинта не ломается, durable-строки может не быть.

## Примеры curl

```bash
BASE=http://localhost:8000/api

# вход -> токен (или {"status":"totp_required","challenge_id":"..."})
TOKEN=$(curl -s -X POST $BASE/auth/login -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"secret"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

# список пользователей
curl -s "$BASE/admin/users?q=ivan&limit=25" -H "Authorization: Bearer $TOKEN"

# изменить пользователя (caller_password нужен только для другого админа)
curl -s -X PATCH $BASE/admin/users/<UID> -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"role":"teacher","caller_password":"secret"}'

# журнал аудита: префикс действия и период
curl -s "$BASE/admin/audit?action=admin.user&since=2026-01-01T00:00:00Z" \
  -H "Authorization: Bearer $TOKEN"

# отозвать сессии пользователя
curl -s -X POST $BASE/admin/users/<UID>/sessions/revoke -H "Authorization: Bearer $TOKEN"

# сброс 2FA другого администратора (step-up)
curl -s -X POST $BASE/admin/users/<UID>/2fa/reset -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"caller_password":"secret"}'
```

## Известные ограничения и backlog v2

- Инциденты пока не собраны в отдельный экран: смотреть журнал аудита и
  уведомления (`src/notifications/system.py`, Telegram).
- Бэкапы БД только создаются: списка и восстановления из UI нет.
- Фоновые операции (seed, embeddings, export) сообщают только о старте: без прогресса и истории.
- Нет вкладки «Сессии»: общий список только по API, персональные - в карточке пользователя.
