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

## Retention

Политика хранения - `docs/RETENTION.md` (команда `python -m src.cli retention`).

## Известное

- `tests/api/test_monitoring_grafana.py::test_unreachable_falls_back` - падает
  вне зоны D (Grafana-ветка opencode-main).
