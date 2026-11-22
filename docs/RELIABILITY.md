# Reliability

Отказоустойчивость и работа с внешними сервисами: таймауты, graceful shutdown,
retention.

## Таймауты внешних вызовов (7.8.4)

Все внешние вызовы обязаны иметь явный таймаут. Текущее состояние:

| Вызов | Место | Таймаут | Ретраи |
|---|---|---|---|
| hh API (запрос вакансий) | `src/parsing/api/hh_api.py:361` | 10 с | `urllib3 Retry(backoff_factor=1)` |
| hh API (refresh token) | `src/parsing/api/hh_api.py:214` | 10 с | token-retry |
| hh API (сессия по умолчанию) | `src/parsing/api/hh_api.py:92` | 15 с | - |
| Hub publications | `src/infrastructure/hub_publications.py:290` | `config.HUB_API_TIMEOUT_S` | - |
| LLM клиент | `src/services/llm_client.py:117` | 120 с | - |

Правило: новый внешний вызов без `timeout=` не проходит ревью. Числа - стартовые,
менять только с обоснованием.

## Graceful shutdown (7.8.3)

Реализован в lifespan приложения: `src/api_pkg/__init__.py:86-107`. При остановке
(SIGTERM/SIGINT) последовательно:
1. `stop_background_collector()` - останавливает фоновый сбор;
2. `stop_log_flusher()` - сбрасывает буфер логов;
3. `stop_system_error_flusher()` - сбрасывает буфер системных ошибок;
4. `close_hub_client()` - закрывает HTTP-клиент Hub;
5. `close_pool()` - закрывает пул asyncpg;
6. `get_engine().dispose()` - освобождает SQLAlchemy-движок.

Порядок важен: буферы сбрасываются до закрытия пула, иначе запись теряется.

## Пробы живости и готовности (7.3.4)

Оркестратор различает два вопроса: «процесс жив?» и «можно ли слать трафик?».

| Проба | Канонический путь | Алиасы | Что проверяет | Код |
|---|---|---|---|---|
| liveness | `GET /health/live` | `/live`, `/api/live`, `/api/health/live` | только то, что процесс отвечает; в зависимости не ходит | всегда 200 |
| readiness | `GET /health/ready` | `/ready`, `/api/health/ready` | `deps.is_ready` (данные загружены) и `src.db.pool` (пул БД) | 200 или 503 |

Readiness - **fail-closed**: пока данных нет или пул БД не поднят, ответ 503 со
`status="not_ready"` и списком `problems` (`data_not_loaded`,
`db_pool_unavailable`). Не критичны и не роняют готовность: `evaluator`,
`recommendation_engine`, `clusterer`, `trend_analyzer` - в режиме ожидания
(`_set_waiting_mode`) они пустые, а API при этом рабочий.

Правило: liveness-проба в манифесте должна указывать на `/health/live`, иначе
временная неготовность (например, первичная загрузка вакансий) вызовет
бессмысленный рестарт контейнера. Плоские `/live` и `/ready` сохранены ради
совместимости, но новые конфиги пишем через `/health/*`.

`GET /health` - отдельный, «человеческий» отчёт (версии моделей, артефакты);
он всегда 200 и для проб не используется.

## Retention

Политика хранения - `docs/RETENTION.md` (команда `python -m src.cli retention`).

## Известное

- `tests/api/test_monitoring_grafana.py::test_unreachable_falls_back` - падает
  вне зоны D (Grafana-ветка opencode-main).
