# RUNBOOK - эксплуатация

Практическое руководство: как запустить, остановить, выкатить, откатить и что
делать при инцидентах. Проект - Competency Gap Analyzer (`compare_online` /
`compare_competencies`). Все пути - от корня репозитория.

> Быстрая проверка перед всем: `powershell -ExecutionPolicy Bypass -File
> tools\release.ps1 -Action Precheck` (ruff, pytest, tsc/vitest/build, alembic,
> pip-audit). Дольше 15-20 минут.

---

## 1. Запуск и остановка (локально, без Docker)

Нужны три процесса: PostgreSQL, backend (FastAPI), frontend (Vite).

### PostgreSQL

Служба Windows `postgresql-x64-17`. Запуск/остановка требуют прав
администратора (UAC):

```powershell
# Статус
Get-Service postgresql-x64-17

# Запуск/остановка (из PowerShell, запущенного от имени администратора)
Start-Service postgresql-x64-17
Stop-Service  postgresql-x64-17
```

Параметры подключения - в `.env` (`DATABASE_URL`, `POSTGRES_*`).

### Backend

```powershell
uvicorn src.api_pkg:app --host 0.0.0.0 --port 8000 --reload
```

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Prometheus (raw): http://localhost:8000/metrics
- Health: http://localhost:8000/health

### Frontend

```powershell
cd frontend
npm install          # один раз
npm run dev          # http://localhost:3000
```

Dev-сервер Vite проксирует `/api` и `/ws` на `http://localhost:8000`
(`frontend/vite.config.ts`, порт 3000).

### Docker (всё сразу)

```bash
docker compose build
docker compose up -d
docker compose logs -f
docker compose down          # остановка
docker compose down -v       # остановка + удаление томов БД
```

Сервисы и порты - в `README.md` (раздел «Запуск через Docker»).

---

## 2. PostgreSQL не стартует

Служба `postgresql-x64-17` в статусе `Stopped` или не поднимается.

1. **Права.** `Start-Service` из обычного PowerShell падает с «Access denied» -
   запустите консоль от администратора (UAC) и повторите.
2. **Порт занят.** Проверьте, не слушает ли 5432 другой процесс (в т.ч. Docker-
   контейнер `competency-postgres`):
   ```powershell
   netstat -ano | findstr :5432
   ```
   Если порт держит контейнер, локальная служба не нужна - работайте через
   Docker или смените порт в `.env`.
3. **Данные повреждены / служба не поднимается после падения.** Смотрите лог
   PostgreSQL:
   ```
   C:\Program Files\PostgreSQL\17\data\log\
   ```
   Часто помогает проверка `pg_ctl` и свободного места на диске.
4. **Служба вообще не создана.** Переинициализация - вне этого runbook; сначала
   восстановите из бэкапа (раздел 6).

Диагностика подключения:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -p 5432 -U postgres -d compare_competencies -c "SELECT 1;"
```

(Пароль - `POSTGRES_PASSWORD` из `.env`.)

---

## 3. Деплой

Релиз делается скриптом `tools/release.ps1`, который гоняет полный гейт и
переносит изменения в целевую ветку (по умолчанию `for_Nikita`):

```powershell
powershell -ExecutionPolicy Bypass -File tools\release.ps1 -Action All
```

`-Action All` = Precheck -> Merge -> Push. Гейт останавливается на первой
проблеме (грязный worktree, регресс ruff, падение тестов, ошибки tsc/build,
несколько alembic-head, новая уязвимая зависимость).

Отдельные шаги:

```powershell
tools\release.ps1 -Action Precheck          # только проверки
tools\release.ps1 -Action Merge             # слить в for_Nikita
tools\release.ps1 -Action Push              # push в origin/for_Nikita
tools\release.ps1 -Action Precheck -SkipBackend   # быстрый прогон без pytest (не для релиза)
```

Для Docker-развёртывания в проде есть `update-deployment.sh` (бэкап ->
git pull -> compose down/build/up -> health check -> авто-rollback при ошибке).

---

## 4. Откат на предыдущий коммит

Сначала посмотрите, куда откатываться:

```bash
git log --oneline -20
git status            # worktree должен быть чистым
```

### Безопасный откат (обратный коммит, история сохраняется)

Рекомендуется для уже запушенной в `for_Nikita` истории:

```bash
git revert <bad_sha>          # создаёт коммит, отменяющий изменения
git push origin for_Nikita
```

### Жёсткий откат ветки (перезапись истории, только если ветка не расшарена)

```bash
git reset --hard <good_sha>   # локальная ветка на нужный коммит
git push --force origin for_Nikita
```

`--force` запрещён без явного согласования - это необратимо для тех, кто уже
забрал ветку.

### Откат Docker-деплоя

`update-deployment.sh` сохраняет предыдущий образ и откатывается автоматически,
если health check не прошёл. Вручную:

```bash
docker compose down
git checkout <good_sha>
docker compose up -d --build
```

---

## 5. Инциденты

### 5xx на API

Сначала - прогон всех GET-эндпоинтов на 5xx:

```powershell
python tools\smoke-get.py
python tools\smoke-get.py --base http://127.0.0.1:8000
```

Скрипт логинится под admin и обходит все GET, печатая 500-е с именем
эндпоинта. Дальше - лог backend'а (structlog) и стек по конкретному пути.
Типовые причины: ошибка в ридере данных, отсутствующий артефакт в `data/`,
недоступная внешняя зависимость (hh.ru, Ollama).

### 401/403 у всех после выката

Признак: массовые «Unauthorized» сразу после релиза, хотя пользователи не
менялись. Почти всегда это **смена `SECRET_KEY`**.

- `SECRET_KEY` подписывает JWT/cookie-сессии. При его изменении все выданные
  токены становятся недействительными, плюс срабатывает deny-list.
- **Проверьте `.env`: значение `SECRET_KEY` должно совпадать с предыдущим
  деплоем.** Не генерируйте новый ключ при обычном выкате.
- Восстановление: вернуть прежний `SECRET_KEY` (из бэкапа `.env` / секрет-
  хранилища) и перезапустить backend. Если прежнего ключа нет - пользователям
  придётся залогиниться заново.

### Зависшие прогоны пайплайна

«Зависший» прогон - это задача в статусе `running`, которая не двигается.

1. Посмотреть активные задачи: `GET /api/pipeline/active` (admin/teacher/rop)
   или `data/cache/pipeline_tasks.json`.
2. Отменить конкретную: `POST /api/pipeline/cancel/{task_id}`.
3. Почистить накопленные устаревшие записи (прогоны, логи запросов):
   ```powershell
   python -m src.cli retention            # dry-run: показывает, что удалится
   python -m src.cli retention --apply    # фактическое удаление
   ```
4. Если процесс реально висит на внешнем вызове - смотрите лог и метрики
   `pipeline_*` в Prometheus; при необходимости перезапустите backend.

---

## 6. Бэкапы

### Где лежат

- `data/backup/` - основной каталог дампов (скрипты проверки смотрят его;
  при пустоте - запасной `data/backups/`).
- `data/backups/` - каталог, который использует CLI `src/cli/backup_db.py`.
- `./backups/` - tar-архивы Docker-томов из `backup-volumes.sh` /
  `update-deployment.sh`.
- Имя дампа: `backup_YYYYMMDD_HHMMSS.sql`.

### Создать дамп

```powershell
python -m src.cli backup                 # дамп в data/backups/
```

### Проверить дамп (читает, ничего не меняет)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_backup.ps1
powershell -ExecutionPolicy Bypass -File scripts\verify_backup.ps1 -Dir data\backups
```

Проверяет: файл существует и не пуст, есть маркеры plain-SQL дампа
(`PostgreSQL database dump`) и данные (`COPY` / `INSERT INTO`), считает число
таблиц. Exit 0 - дамп пригоден; exit 1 - проблема (файла нет, пустой, битый
или нет данных).

### Проверить восстановимость (в отдельную БД, рабочие данные не трогаются)

```powershell
# Сначала dry-run: печатает план и выходит
powershell -ExecutionPolicy Bypass -File scripts\restore_check.ps1

# Реальное восстановление в cc_restore_check (создаст и удалит БД)
powershell -ExecutionPolicy Bypass -File scripts\restore_check.ps1 -Confirm
```

Скрипт поднимает временную БД `cc_restore_check`, восстанавливает в неё дамп,
проверяет что таблиц > 0, и удаляет БД (`-Keep` оставит её для разбора).

### Восстановить рабочие данные

```powershell
python -m src.cli backup --restore latest        # последний дамп
python -m src.cli backup --restore data\backups\backup_20260101_120000.sql
```

Docker-тома - вручную из `./backups/*.tar.gz`. Перед восстановлением остановите
backend, чтобы не было пишущих транзакций.

---

## 7. 2FA администратора

TOTP-двухфакторка. Секрет и recovery-коды хранятся в таблице `admin_totp`
(секрет - зашифрован).

Есть два независимых уровня:

| Что | Где | По умолчанию |
|---|---|---|
| **Мастер-переключатель** (для всей системы) | таблица `app_settings`, ключ `security.two_factor` | **выключен** |
| Личное подключение TOTP | `admin_totp`, по пользователю | не подключено |

Мастер-переключатель - единственный источник правды: его читают и вход по
паролю, и SSO. Пока он выключен, код не спрашивается ни у кого и эндпоинты
`/auth/2fa/*` отвечают `404` («отключена администратором»). Подключения
пользователей при выключении **не удаляются** - при следующем включении те же
люди снова получают запрос кода.

### Выключить / включить (одним кликом)

Админка → вкладка **Безопасность** → кнопка «Включить»/«Выключить».
API-эквивалент:

```
GET  /api/admin/security/2fa/settings   -> { "enabled": bool, "enrolled_admins": int }
PUT  /api/admin/security/2fa/settings   { "enabled": true }
```

Выключение кода не требует (это аварийный тумблер). Каждое переключение
пишется в аудит как `admin.2fa.toggle`.

Проверить состояние без авторизации (для экрана входа):

```
GET /api/auth/2fa/policy   -> { "enabled": bool }
```

### Включить 2FA себе

1. Мастер-переключатель должен быть **включён** (иначе `404`).
2. Админка → Безопасность → «Подключить приложение-аутентификатор», либо:
   ```
   POST /api/auth/2fa/setup
   ```
   В ответе - `otpauth_uri` и `secret`. Отсканируйте QR из `otpauth_uri`
   (или вбейте `secret`) в Google Authenticator / Aegis / аналоги.
3. Подтвердить одноразовым кодом:
   ```
   POST /api/auth/2fa/confirm   { "code": "123456" }
   ```
   В ответе `recovery_codes` - **сохраните их в надёжном месте**. Коды
   одноразовые, каждый срабатывает ровно раз.
4. После подключения вход становится двухшаговым:
   - `POST /api/auth/login` возвращает `challenge_id` (не токен);
   - `POST /api/auth/2fa/verify { "challenge_id": "...", "code": "..." }`
     выдаёт токен. Challenge живёт 5 минут.

### Вход при включённом переключателе

| Кто | Что вернёт `/auth/login` |
|---|---|
| подключил TOTP | `{"status": "totp_required", "challenge_id": ...}` |
| админ без TOTP | `{"status": "totp_setup_required", "enroll_id": ...}` - сначала настройка |
| не админ без TOTP | обычный токен (иначе включение заперло бы всех сразу) |

Пропуск `enroll_id` одноразовый и живёт 10 минут. Настройка по нему проходит
**без сессии**: `POST /auth/2fa/setup {"enroll_id": ...}` →
`POST /auth/2fa/confirm {"code": ..., "enroll_id": ...}`. Подтверждение сразу
отдаёт токен и recovery-коды, чтобы не гонять человека по кругу.

> SSO через академический API подчиняется тому же переключателю: выключен -
> код не спрашиваем, включён - у подключённых SSO-вход тоже двухшаговый.

### Потеря телефона

1. **Recovery-код** (резервный способ входа):
   ```
   POST /api/auth/2fa/recovery { "challenge_id": "...", "code": "<recovery-код>" }
   ```
   `challenge_id` берётся из ответа `/auth/login`. Использованный код удаляется.
2. **Отключить 2FA** (нужен действующий TOTP- или recovery-код и активная
   admin-сессия):
   ```
   POST /api/auth/2fa/disable { "code": "<код>" }
   ```
   После этого таблица `admin_totp` очищается, вход снова одношаговый -
   настройте 2FA заново на новом телефоне.
3. Если утеряны и телефон, и recovery-коды, а войти нельзя - остаётся прямой
   доступ к БД: очистить строку в `admin_totp` для нужного `user_id`, затем
   залогиниться и настроить 2FA заново.

### Защита от перебора

Неверные коды считаются: после `_TOTP_MAX_FAILURES` срабатывает блокировка с
ответом `429` и заголовком `Retry-After`.

---

## 8. Retention (очистка устаревших данных)

Удаляет старые вакансии, логи запросов и прогоны пайплайна. По умолчанию -
**dry-run**: печатает план и ничего не удаляет.

```powershell
python -m src.cli retention                                  # dry-run
python -m src.cli retention --apply                          # удалить
python -m src.cli retention --vacancies-days 730 --logs-days 90 --run-days 365 --apply
```

Все пороги должны быть >= 1. Сначала смотрите dry-run, затем запускайте
`--apply` (можно завернуть в cron/планировщик).

---

## 9. Полезные ссылки

- `tools\release.ps1 -Action Precheck` - полный гейт перед релизом
- `python tools\smoke-get.py` - обход GET-эндпоинтов на 5xx
- `scripts\verify_backup.ps1`, `scripts\restore_check.ps1` - бэкапы
- `docs\API-CHANGELOG.md` - изменения API
- `README.md` - запуск (локально и Docker)
