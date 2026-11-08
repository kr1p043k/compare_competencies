# RUNBOOK: compare_competencies (прод)

Аварийный справочник по прод-серверy `compareserver`. Все команды выполняются
на сервере из каталога приложения:

```bash
ssh admin_server@compareserver
cd /home/admin_server/compare_competencies
```

Смежные документы: `deploy/README.md` (как устроен деплой), `MONITORING` -
`monitoring/README.md` (правила алертов), `docs/ARCHITECTURE.md`.

## 0. Быстрая проверка: что задеплоено

```bash
cat ~/.cc-deployed-sha          # коммит, который подтвердил cc-autodeploy
git log -1 --oneline            # фактический HEAD рабочего дерева
git rev-parse origin/main       # что лежит в origin/main
journalctl -u cc-autodeploy -n 50 --no-pager
docker compose ps
curl -sf http://localhost:8000/health && echo " backend OK"
```

Как устроен деплой: systemd-таймер `cc-autodeploy.timer` раз в 5 минут запускает
`deploy/cc-autodeploy.sh`, который вызывает `scripts/deploy.sh` (маркер
`~/.cc-deployed-sha`, лог `logs/autodeploy.log`). `.github/workflows/deploy.yml` -
ручной фолбэк. `APP_DIR=/home/admin_server/compare_competencies`,
`HEALTH_URL=http://localhost:8000/health`, `HEALTH_TIMEOUT=300`.

---

## 1. Откат (rollback)

### 1.1. Остановить автодеплой, чтобы не перекатил обратно на `main`

```bash
sudo systemctl stop cc-autodeploy.timer
```

Не забыть вернуть после починки: `sudo systemctl start cc-autodeploy.timer`.

### 1.2. Быстрый откат по образам (если плохой релиз, миграции совместимы)

`deploy.sh` при живом `/health` перетегирует текущие образы:
`compare_competencies-backend:latest` -> `:latest.previous` (и то же для frontend).

```bash
docker tag compare_competencies-backend:latest.previous  compare_competencies-backend:latest
docker tag compare_competencies-frontend:latest.previous compare_competencies-frontend:latest
docker compose up -d backend frontend --force-recreate
curl -sf http://localhost:8000/health && echo " health OK"
```

### 1.3. Откат по коммиту (если нужен и код, и образ)

```bash
cd /home/admin_server/compare_competencies
git fetch origin main
git reset --hard <GOOD_SHA>
git rev-parse HEAD > ~/.cc-deployed-sha   # маркер: таймер не будет тянуть main заново
docker compose build backend
docker compose run --rm --no-deps --entrypoint alembic backend upgrade heads
docker compose up -d backend frontend --force-recreate
curl -sf http://localhost:8000/health && echo " health OK"
```

Откат миграций выполняется отдельно, см. раздел «Сбой миграции».

---

## 2. Восстановление БД (restore) с проверкой целостности

В репозитории ТРИ разных механизма бэкапа, не путать:

| Механизм | Команда / файл | Формат | Восстанавливается |
|---|---|---|---|
| CLI (основной) | `python -m src.cli backup` -> `data/backups/backup_<UTC>.sql` | pg_dump `--format=custom` | `pg_restore`, в т.ч. `python -m src.cli backup --restore latest\|PATH` |
| update-deployment | `./update-deployment.sh backup` -> `backups/db_<date>.sql` | plain SQL | только `psql -f` (НЕ `pg_restore`) |
| volume tars | `./backup-volumes.sh` -> `backups/<volume>_<date>.tar.gz` | tar каталогов томов | распаковкой, не дамп БД |

### 2.1. Проверка целостности ПЕРЕД восстановлением

```bash
cd /home/admin_server/compare_competencies
LATEST=$(ls -1t data/backups/backup_*.sql | head -n1)
echo "Проверяю: $LATEST"

# 1) дамп вообще читается как custom-format
cat "$LATEST" | docker exec -i competency-postgres pg_restore --list > /tmp/dump.list \
  && echo "dump.list OK" || { echo "ДАМП БИТЫЙ"; exit 1; }

# 2) в дампе есть таблица users (значит бэкап не пустой)
grep -q "TABLE public users" /tmp/dump.list && echo "users: OK" || echo "users: НЕ НАЙДЕНА"
```

Если проверка не прошла, восстановление не запускать: искать другой бэкап
(`ls -1t data/backups/backup_*.sql | head`).

### 2.2. Снимок текущего состояния (на случай, если бэкап тоже плохой)

```bash
python -m src.cli backup     # создаст свежий data/backups/backup_<UTC>.sql
```

### 2.3. Восстановление

Остановить backend, чтобы не было записи во время restore:

```bash
docker compose stop backend
cat "$LATEST" | docker exec -i competency-postgres pg_restore \
  --clean --if-exists --no-owner --no-privileges \
  --dbname=postgresql://postgres@localhost:5432/compare_competencies
docker compose start backend
curl -sf http://localhost:8000/health && echo " health OK"
```

Альтернатива из CLI (то же самое, но без явного DSN): `python -m src.cli backup
--restore "$LATEST"`. Всегда указывайте явный путь, а не `latest`: `latest` может
указать на свежий «страховочный» снимок из 2.2, а не на проверенный дамп.

### 2.4. Пост-проверка данных

```bash
docker exec competency-postgres psql -U postgres -d compare_competencies -tAc "SELECT count(*) FROM users;"
```

---

## 3. Сбой миграции (migration failure)

Симптом: `deploy.sh` падает на шаге `migrations (alembic upgrade heads)`, либо
backend не поднимается после миграции. Сначала остановить автодеплой (раздел 1.1).

```bash
cd /home/admin_server/compare_competencies

# где мы и куда идём; голова должна быть одна
docker compose run --rm --no-deps --entrypoint alembic backend current
docker compose run --rm --no-deps --entrypoint alembic backend heads
# ожидаемая единственная голова: retention_fk_on_delete

# кто держит блокировку (миграция ждёт ALTER TABLE)
docker exec competency-postgres psql -U postgres -d compare_competencies -c \
  "SELECT pid, state, wait_event_type, left(query,80) FROM pg_stat_activity WHERE state IS DISTINCT FROM 'idle';"
```

Если применена плохая ревизия, откатить её на одну назад и вернуть код/образ
(раздел 1.3):

```bash
docker compose run --rm --no-deps --entrypoint alembic backend downgrade -1
```

**Проверено локально на чистой базе:**

| Команда | Результат |
|---|---|
| `upgrade heads` с нуля | ok, все ревизии применяются |
| `downgrade -1` | ok, откатывает ровно одну ревизию |
| повторный `upgrade heads` | ok, возвращается на ту же голову |
| `downgrade base` | **падает**, см. ниже |

`downgrade base` не поддерживается: при откате до основания старая ревизия
возвращает ограничение `ck_user_role CHECK (role IN ('admin','teacher','student'))`,
а роль `teacher` появилась позже, и в таблице уже есть строки с этой ролью.
Ограничение не создаётся. **Откатывайтесь по одной ревизии (`downgrade -1`), а
не до `base`.** Полный откат базы делается восстановлением из бэкапа (раздел 2),
а не миграциями.

Частая причина зависания - активные читатели таблицы, которую мигрируют.
Дождаться/остановить долгие запросы, повторить `upgrade heads`. После починки
обязательно выставить маркер `git rev-parse HEAD > ~/.cc-deployed-sha` и вернуть
таймер.

---

## 4. Шторм 5xx и недоступность API

Важно: middleware пишет в `api_requests_total` ТОЛЬКО пути `/api/*`
(`src/api_pkg/request_logger.py`). `/health` и `/ready` в эти метрики НЕ попадают.

```bash
# 5xx по метрикам (нужен METRICS_TOKEN из .env; без него /metrics отдаёт 403)
TOKEN=$(grep -E '^METRICS_TOKEN=' .env | cut -d= -f2-)
curl -s -H "X-Metrics-Token: $TOKEN" http://localhost:8000/metrics \
  | grep -E '^api_requests_total'

# свежие ошибки и логи
docker logs --tail 200 compare-competencies-api
tail -n 200 logs/backend.log

# соединения и тяжёлые запросы
docker exec competency-postgres psql -U postgres -d compare_competencies -c \
  "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

Порядок действий:

1. `curl -sf http://localhost:8000/ready` - если `not_ready`, смотреть `problems`
   (`data_not_loaded`, `db_pool_unavailable`).
2. Если 5xx начался сразу после деплоя - откат по образам (раздел 1.2).
3. Если БД перегружена - снять тяжёлые запросы, затем при необходимости
   `docker compose restart backend`.
4. Если ошибки из-за размеров данных - `retention` (раздел 5).

---

## 5. Диск заполнен (disk full)

Ориентиры: `data` ~724 MB, `logs` ~48.7 MB, `uploads` ~0.1 MB. Логи backend
ротируются в `logs/backend.log.1..5`.

```bash
df -h /
du -sh data logs uploads 2>/dev/null
docker system df

# безопасная уборка
docker image prune -f
ls -1t data/backups/*.sql 2>/dev/null | tail -n +6 | xargs -r rm --   # оставить 5 свежих
rm -f backups/*_<старая_дата>.tar.gz

# чистка БД (dry-run по умолчанию, удаление только с --apply)
python -m src.cli retention
python -m src.cli retention --apply
```

`retention` удаляет устаревшие `vacancies` (2 года), `request_logs` (90 дней),
`pipeline_runs` (1 год); значения дней ограничены диапазоном 1..36500.

Не запускать `logs.sh`: он ссылается на отсутствующий `docker-compose.prod.yml`.
Смотреть логи так:

```bash
docker logs -f --tail=100 compare-competencies-api
tail -f logs/backend.log
```

---

## 6. Ротация секретов (secret rotation)

### 6.1. SECRET_KEY

Ротация `SECRET_KEY` немедленно инвалидирует:

- все сессионные токены (пользователи разлогинены);
- секреты admin TOTP (Fernet-ключ выводится из `SECRET_KEY`) - админам нужно
  заново привязать 2FA;
- academic SSO-токены, сохранённые в `sessions.sso_token`.

```bash
openssl rand -hex 32          # новый SECRET_KEY
# записать новое значение в .env -> SECRET_KEY=...
docker compose up -d backend --force-recreate
curl -sf http://localhost:8000/health && echo " health OK"
```

Предупредить пользователей об повторном входе и админов о перепривязке TOTP.

### 6.2. METRICS_TOKEN (доступ к /metrics)

```bash
openssl rand -hex 32 > prometheus/metrics_token   # файл читает Prometheus
# то же значение в .env -> METRICS_TOKEN=...
docker compose up -d backend prometheus --force-recreate
```

Проверить: `curl -s -o /dev/null -w '%{http_code}\n' -H "X-Metrics-Token: $TOKEN" http://localhost:8000/metrics`
должен вернуть `200`. Имя заголовка `X-Metrics-Token` - часть контракта
`/metrics`: Prometheus шлёт его через `http_headers` в `prometheus/prometheus.yml`,
а не через `Authorization: Bearer` (см. `monitoring/README.md`).
