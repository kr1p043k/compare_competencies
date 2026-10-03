# LLM Phase 2 — сопровождение

## Env vars

| Var | Default | Назначение |
|---|---|---|
| `OLLAMA_URL` | `http://ollama8.r61.net:11434` | Endpoint Ollama (OpenAI-совместимый `/v1`) |
| `OLLAMA_MODEL` | `gpt-oss:120b` | Модель чата (`LLMClient.model`) |
| `OLLAMA_EMBEDDING_MODEL` | `qwen2.5:0.5b` | Модель эмбеддингов |
| `QWEN_TEMPERATURE` | `0.7` | Температура по умолчанию |
| `QWEN_MAX_TOKENS` | `2000` | Лимит токенов по умолчанию |
| `LLM_ENABLED` | `true` | Мастер-флаг Phase 2 (без саб-флагов поведения не меняет) |
| `LLM_ENHANCE_STUDENT` | `false` | LLM-дополнение рекомендаций студента |
| `LLM_ENHANCE_TEACHER` | `false` | LLM-дополнение рекомендаций преподавателя |
| `LLM_EXTRACT` | `false` | LLM-извлечение скиллов из текста |
| `LLM_TIMEOUT_S` | `60` | Таймаут вызова LLM в UX-путях, сек |

## Архитектура

```
router (/llm/chat, student/teacher wiring) -> LLMClient -> [cache] -> Ollama /v1
                                                        -> fallback (base) при любой ошибке
```

`LLMClient` (`src/services/llm_client.py`): OpenAI-клиент на
`base_url + /v1`, `api_key="ollama"`, `timeout=120.0`, `max_retries=2`.
`.chat(messages, temperature, max_tokens)`; UX-пути оборачивают вызов в
`ThreadPoolExecutor` с `LLM_TIMEOUT_S` (60с).
Метрики: `llm_requests_total{model,status}`, `llm_request_duration_seconds`,
`llm_token_usage`, `llm_response_length`.
Сервисы Phase 2: `src/services/llm_extract.py` (`extract_skills`, задача кэша
`'extract'`), `src/services/llm_recommend.py` (`enhance_student_recs`,
`enhance_teacher_recs`, задача `'recommend'`).
Врезки: `routers/profiles.py:get_recommendations` (студент),
`routers/teacher.py:get_analysis_discipline` (преподаватель) — обе за флагами,
любая ошибка LLM = silent fallback на base.

## Cache ops

- Таблицы Phase 2 (миграция `alembic/versions/add_llm_cache_tables.py`,
  созданы в БД 2026-09-27): `llm_cache(task, model, prompt_hash CHAR(64),
  prompt_text, response_text, tokens, created_at; UNIQUE(task,model,prompt_hash))`
  и `skill_embedding_cache(skill_key PK, model, embedding vector(768))`.
  Модуль: `src/services/llm_cache.py` (`get_cached/put_cached/get_embedding/
  put_embedding`; любая ошибка БД = warning + None, raise нет).
- NB: в БД есть старая таблица `llm_recommendations` (`sql/004`) — это другой
  (существовавший ранее) механизм, Phase 2 его не использует.
- NB: цепочка alembic фрагментирована (8 heads, в БД штамп несуществующей
  `merge_for_nikita_main`) — таблицы созданы прямым SQL (обратимо через DROP);
  файл миграции лежит для будущего ремонта цепочки.
- Осмотр: `SELECT task, model, created_at FROM llm_cache
  ORDER BY created_at DESC LIMIT 20;`
- Пурж: `DELETE FROM llm_cache WHERE created_at < NOW() - INTERVAL '30 days';`
- Рост: ~1 строка на уникальный `(task,model,prompt_hash)`; hit строк не добавляет.

## Failure modes

| Сбой | Что видит пользователь |
|---|---|
| Endpoint down / connection refused | Базовые (не-LLM) рекомендации без изменений |
| Timeout (`LLM_TIMEOUT_S`=20) | То же: fallback, base без изменений |
| Bad JSON от модели | Extract/enhance возвращают `[]` / base без изменений |
| Все саб-флаги OFF | Поведение байт-идентично до Phase 2 (см. `TestFlagsOff`) |

## Включение по ролям

```
LLM_ENABLED=1 LLM_ENHANCE_STUDENT=1  # только студенты
LLM_ENABLED=1 LLM_ENHANCE_TEACHER=1  # только преподаватели
LLM_ENABLED=1 LLM_EXTRACT=1          # только извлечение скиллов
```

## Rollback

Снять саб-флаги (`LLM_ENHANCE_*=0`, `LLM_EXTRACT=0`): код идёт по старым путям,
LLM не вызывается. Таблицы `llm_cache`/`skill_embedding_cache` аддитивны,
откат кода их не требует (пурж/удаление — прямым SQL).

## Тесты

`tests/api/test_llm_phase2.py` — 14 тестов на РЕАЛЬНЫХ биндингах (без шимов):
FAKE chat-клиент (протокол `.chat`), real Postgres `llm_cache` (строки чистятся),
ассерты дефолтов флагов. Без сети и без Ollama.
