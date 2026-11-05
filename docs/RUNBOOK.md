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

TOTP-двухфакторка для роли `admin`. Секрет и recovery-коды хранятся в таблице
`admin_totp` (секрет - зашифрован).

### Включить

1. Залогиниться как admin и вызвать:
   ```
   POST /api/auth/2fa/setup
   ```
   В ответе - `otpauth_uri` и `secret`. Отсканируйте QR из `otpauth_uri`
   (или вбейте `secret`) в Google Authenticator / Aegis / аналоги.
2. Подтвердить одноразовым кодом:
   ```
   POST /api/auth/2fa/confirm   { "code": "123456" }
   ```
   В ответе `recovery_codes` - **сохраните их в надёжном месте**. Коды
   одноразовые, каждый срабатывает ровно раз.
3. После включения вход становится двухшаговым:
   - `POST /api/auth/login` возвращает `challenge_id` (не токен);
   - `POST /api/auth/2fa/verify { "challenge_id": "...", "code": "..." }`
     выдаёт токен. Challenge живёт 5 минут.

> Если включён SSO/admin через академический API - 2FA всё равно обязательна
> (SSO больше не обходит её).

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
