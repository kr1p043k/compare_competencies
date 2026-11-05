# Monitoring: правила алертов

Каталог содержит правила Prometheus и заметки по их подключению.

- `alerts.yml` - Prometheus alerting rules (только реальные метрики; отключённые
  правила закомментированы как ТЗ).

## Как загрузить правила

Prometheus в `docker-compose.yml` сейчас монтирует только `prometheus.yml` и
`prometheus/metrics_token`, а директивы `rule_files` в `prometheus/prometheus.yml`
нет. Чтобы алерты подхватились, нужно:

1. Смонтировать каталог `monitoring/` в контейнер и добавить `rule_files` в
   `prometheus/prometheus.yml`:

   ```yaml
   # prometheus/prometheus.yml (в начало файла, рядом с global)
   rule_files:
     - /etc/prometheus/rules/alerts.yml
   ```

2. Добавить том в сервис `prometheus` в `docker-compose.yml`:

   ```yaml
   volumes:
     - prometheus-data:/prometheus
     - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
     - ./prometheus/metrics_token:/etc/prometheus/metrics_token:ro
     - ./monitoring:/etc/prometheus/rules:ro   # добавить
   ```

3. Перезапустить/перечитать Prometheus (`--web.enable-lifecycle` уже включён):

   ```bash
   docker compose up -d prometheus
   curl -X POST http://localhost:9090/-/reload
   ```

Список загруженных правил: `http://localhost:9090/rules`, активные алерты:
`http://localhost:9090/alerts`. Для доставки уведомлений нужен Alertmanager
(сейчас в compose его нет - добавить отдельно).

## Проверка правил (promtool)

Локально `promtool` не установлен, поэтому используем официальный образ:

```bash
# Linux/macOS
docker run --rm -v "$PWD/monitoring:/m" \
  --entrypoint promtool prom/prometheus:latest check rules /m/alerts.yml

# Windows PowerShell
docker run --rm -v "${PWD}/monitoring:/m" `
  --entrypoint promtool prom/prometheus:latest check rules /m/alerts.yml
```

Юнит-тесты правил (если появятся `alerts_test.yml`):

```bash
docker run --rm -v "$PWD/monitoring:/m" \
  --entrypoint promtool prom/prometheus:latest test rules /m/alerts_test.yml
```

Без Docker быстрая проверка синтаксиса YAML:

```bash
python -c "import yaml,io; yaml.safe_load(io.open('monitoring/alerts.yml',encoding='utf-8'))"
```

## Какой экспортёр нужен каждому правилу

| Правило | Метрики | Источник |
|---|---|---|
| `CcBackendDown` | `up{job="backend"}` | job `backend` в prometheus.yml |
| `CcApi5xxRateHigh`, `CcApiP95LatencyHigh` | `api_requests_total`, `api_request_duration_seconds_bucket` | backend `/metrics` |
| `CcDbConnectionsSaturated` | `pg_stat_database_numbackends`, `pg_settings_max_connections` | postgres-exporter |
| `CcDiskSpaceLow`, `CcDiskWillFillSoon` | `node_filesystem_avail_bytes`, `node_filesystem_size_bytes` | node-exporter |
| `CcLlmErrorRateHigh` | `llm_requests_total{status}` | backend `/metrics` |
| `CcPipelineErrors` | `pipeline_errors_total` | backend `/metrics` |
| (отключено) ready-проба | `probe_success` | blackbox_exporter (НЕ настроен) |
| (отключено) retention | `retention_last_success_timestamp_seconds` | НЕ публикуется |
| (отключено) пул БД | `db_pool_*` | НЕ публикуется |

Контейнеры экспортёров из `docker-compose.yml`: `postgres-exporter`
(`postgres-exporter:9187`), `node-exporter` (`node-exporter:9100`), `cadvisor`
(`cadvisor:8080`).

## ВАЖНО: несоответствие авторизации /metrics (дефект)

Backend закрывает `/metrics` заголовком `X-Metrics-Token` (константное сравнение
через `hmac.compare_digest`) либо ролью `admin` - см.
`src/api_pkg/__init__.py:359-369`. Прометею логин недоступен, поэтому он должен
слать именно `X-Metrics-Token`.

Но `prometheus/prometheus.yml` сейчас шлёт другой заголовок:

```yaml
authorization:
  type: Bearer
  credentials_file: /etc/prometheus/metrics_token
```

`authorization` всегда пишет в заголовок `Authorization: Bearer <token>`, а
backend читает `X-Metrics-Token`. Скрейп job=backend получает 403, таргет
падает (`up{job="backend"} == 0`), а `CcBackendDown` будет гореть ложно.

Починка (Prometheus поддерживает кастомные заголовки с чтением из файла -
`http_headers.<name>.files`). Заменить блок `authorization` в job `backend` на:

```yaml
    http_headers:
      X-Metrics-Token:
        files:
          - /etc/prometheus/metrics_token
```

`prometheus/metrics_token` уже смонтирован в контейнер read-only
(`docker-compose.yml`), так что содержимое файла остаётся секретом и не попадает
в git. После правки: `docker compose up -d prometheus` и
`curl -X POST http://localhost:9090/-/reload`, затем убедиться, что
`up{job="backend"} == 1` и в `http://localhost:9090/targets` у job=backend нет
`403 Forbidden`.

Проверка вручную (значение токена - в `.env`, `METRICS_TOKEN`):

```bash
TOKEN=$(grep -E '^METRICS_TOKEN=' .env | cut -d= -f2-)
curl -s -o /dev/null -w '%{http_code}\n' -H "X-Metrics-Token: $TOKEN" http://localhost:8000/metrics
# ожидаем 200
```

## Известные пробелы в метриках (документированы, пока не закрыты)

- Нет пробы `/ready`: endpoint есть (`src/api_pkg/routers/health.py:146`), но
  blackbox_exporter не развёрнут - метрики `probe_success` нет.
- Нет метрики свежести retention (последнего успешного запуска).
- Нет gauge пула соединений БД.
- `/health` и `/ready` не попадают в `api_requests_total`: middleware
  `RequestLogMiddleware` учитывает только пути `/api/*`.
