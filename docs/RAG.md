# RAG: retrieval по РПД, вакансиям и публикациям

Подсистема добавляет LLM-чату второй слой контекста: **доказательства**
(цитаты из корпуса) рядом с **фактами** (агрегаты, которые уже считает
пайплайн). Числа и выводы берутся только из фактов, формулировки и ссылки —
только из доказательств.

Фича за флагом `RAG_ENABLED` (по умолчанию `false`): без него ничего не
индексируется и не ищется, `llm_chat` работает как раньше.

## Состав

```
src/rag/
  types.py      Document, Chunk, RetrievalHit, Citation, RagContext, RagQuery
  settings.py   RagSettings + PROFILES + AclScope (иерархия видимости)
  chunking/     base (упаковка/boilerplate), rpd, vacancy, card (KRM/таксономия)
  embedder.py   singleton sentence-transformers, батчи, async-обёртка
  store.py      SQL к rag_*: поколения, upsert, dense/sparse/entity-поиск
  indexer.py    оркестрация: hash-diff -> embed -> upsert -> tombstone/switch
  query.py      нормализация, леммы (pymorphy3), intent, сущности, фильтры
  retriever.py  RRF-фьюжн, дедуп, агрегация по документу, MMR
  context.py    сборка промпт-блока, цитаты [n], бюджет, санитайз injection
  service.py    фасад RagService (analyze -> retrieve -> assemble + лог)
```

## Схема БД (`alembic/versions/add_rag_tables.py`)

| Таблица | Назначение |
|---|---|
| `rag_sources` | источник + `active_generation` + `spec_version` |
| `rag_documents` | документ, `content_hash`, `acl_scope`, `direction/year`, tombstone `valid_to` |
| `rag_chunks` | чанк, `embedding vector(768)`, `tsv`, `section_path` |
| `rag_chunk_entities` | (chunk, тип, ключ) — точный поиск «всё про навык/компетенцию» |
| `rag_ingest_runs` | история ингестии со счётчиками |
| `rag_query_log` | хеш запроса, intent, hits, top_score, latency (без сырого текста) |
| `rag_feedback` | thumbs up/down — сырьё для golden-set |

Индексы: HNSW (`vector_cosine_ops`), GIN по `tsv`, составные по
`(source, generation, acl_scope)` и `(direction, year)`.

Ключевое: поиск всегда фильтрует `d.generation = s.active_generation`, поэтому
переиндексация в новое поколение не ломает работающий поиск (blue-green).

## Индексация

```bash
python -m src.cli rag-index --source all --dry-run    # только посчитать
python -m src.cli rag-index --source rpd              # инкрементально
python -m src.cli rag-index --source rpd --reindex    # новое поколение + switch
python -m src.cli rag-index --source krm
python -m src.cli rag-index --source vacancies --from-db --limit 5000
python -m src.cli rag-index --stats
```

Источники:

| source | откуда | чанкер |
|---|---|---|
| `rpd` | `data/reference/sfu_annotations_all.json` | секции РПД (цели/задачи/компетенции/содержание/ФОС) |
| `krm` | `krm_disciplines_*.json` + `skill_taxonomy.json` | синтетические карточки компетенций/дисциплин/навыков |
| `vacancies` | таблица `vacancies` (`--from-db`) или `data/raw/hh_vacancies_basic.json` | требования/обязанности/описание/навыки |

Идемпотентность: документ с неизменившимся `content_hash` пропускается;
удалённые помечаются `valid_to` (tombstone), а не удаляются.

Измерено на текущем корпусе РПД: 7 направлений, 447 документов, ~2300 чанков,
медиана 520 токенов, p90 — 670.

## Поиск

Три ветки параллельно с общим таймаутом `RAG_TIMEOUT_S`:

1. **dense** — HNSW по эмбеддингу запроса;
2. **sparse** — `ts_rank_cd` по `to_tsvector('russian', ...)`;
3. **entity** — точное совпадение по `rag_chunk_entities` (коды компетенций,
   навыки из `it_skills.json`).

Далее RRF-фьюжн → дедуп по `content_hash` → ≤3 чанка на документ → MMR.

Профили (`RagSettings.profile`):

| профиль | источники | что решает |
|---|---|---|
| `chat_qa` | все | обычный вопрос в чате |
| `rpd_gap_explain` | rpd, krm | «почему компетенция не покрыта» |
| `market_evidence` | vacancies, publications | «что требует рынок» |
| `curriculum_design` | rpd, publications, docs | «что добавить в РПД» |
| `competency_lookup` | krm | «что входит в УК-3» |

Intent определяется правилами по ключевым словам (`query.detect_intent`) —
дешево и детерминированно; профиль влияет только на источники и веса.

## Интеграция в чат

`llm_chat.assess(..., use_rag=None)`:
- `use_rag=None` → берётся `config.RAG_ENABLED`;
- retrieval выполняется по **тексту вопроса** (`message`), не по названию объекта;
- **роль передаётся из роутера** (`chat._stream` → `rag_role`), иначе ACL
  схлопнулся бы до `public` и корпус (`internal`) был бы невидим;
- профиль выбирается по intent из вопроса (правила), `rag_profile` — override;
- блок `<evidence>` с цитатами `[n]` попадает в промпт **перед** фактами
  (`_context_text`: 55% бюджета — доказательства);
- промпты требуют ссылки `[n]` только когда evidence есть;
- `answer_blocks` добавляет `citation`-блоки (фронт `ChatBlockView` уже умеет);
- **citation-блоки от самой модели отбрасываются** (`_MODEL_BLOCK_TYPES`):
  иначе LLM могла бы придумать URL и подпись;
- `rag`-метаданные в ответе: `hits`, `top_score`, `degraded`, `intent`,
  `profile`, `groundedness` (доля упомянутых ссылок, реально выданных retrieval);
- при `top_score < RAG_MIN_SCORE`, пустом индексе или ошибке retrieval ответ
  строится только по фактам, `llm_used` не меняется.

`build_context(..., rag=..., rag_query=...)` — точка внедрения для тестов и
альтернативных вызывающих; без `rag` состав контекста прежний.

## Кэш

L1 в памяти процесса: ключ = `нормализованный запрос | профиль | уровень ACL |
источники | top_n | направление | год`. TTL `RAG_CACHE_TTL_S` (60 с).
Кэшируется только непустой результат (пустой из-за сбоя ветки не залипает).
Сброс: `rag.service.clear_cache()`, подписан на `cache_bus` как `rag` —
стадия пайплайна инвалидирует после реиндексации.

## Пайплайн

`src/pipeline/rag_stage.py::RagIndexStage` (70–74%, после `ModelTrainingStage`):
1. при `RAG_ENABLED=false` — no-op;
2. инкрементально переиндексирует вакансии из БД (`--from-db`, по content_hash);
3. сбрасывает L1-кэш и `cache_bus`;
4. любая ошибка индексации → `Ok` с полем `error`: сбор рынка не падает из-за RAG.


## API

| метод | путь | роли |
|---|---|---|
| GET | `/api/rag/sources` | admin, teacher, rop |
| POST | `/api/rag/search` | admin, teacher, rop |
| POST | `/api/rag/feedback` | admin, teacher, rop, student |
| POST | `/api/rag/reindex` | admin |

`POST /api/rag/search` возвращает intent, профиль, сущности, `top_score`,
`degraded`, попадания со scores по ветками и готовый `prompt_block` — удобно
для отладки без LLM.

## Безопасность

- `AclScope.for_role` → `acl_scope IN (...)` в SQL **до** ANN: студент не может
  получить teacher-only документы даже при высоком косинусе.
- Студенческие персональные данные в индекс не попадают — только учебные
  документы и рыночные тексты.
- Текст чанков оборачивается в `<evidence>` и чистится от injection-маркеров
  (`context.sanitize_evidence`); запрос проверяется `llm_guardrails`.
- `rag_query_log` хранит `query_hash`, а не сырой текст.

## Деградация

| отказ | поведение |
|---|---|
| модель эмбеддингов не загрузилась | только FTS + entity, `degraded=dense_unavailable` |
| ветка не уложилась в таймаут | ветка отбрасывается, остальные работают |
| индекс пуст | `evidence=""`, ответ по фактам |
| LLM недоступен | детерминированные блоки (существующий механизм) |

## Конфиг

`RAG_ENABLED`, `RAG_PROFILE_DEFAULT`, `RAG_TOP_N`, `RAG_MIN_SCORE`,
`RAG_CONTEXT_BUDGET`, `RAG_TIMEOUT_S`, `RAG_DENSE_K`, `RAG_SPARSE_K`,
`RAG_ENTITY_K`, `RAG_INDEX_BATCH`, `RAG_MMR_LAMBDA`, `RAG_MAX_CHUNK_TOKENS`,
`RAG_MIN_CHUNK_TOKENS`, `RAG_ACL_DEFAULT`, `RAG_CACHE_TTL_S`. См. `.env.example`.

## Метрики (Prometheus)

`rag_retrieval_duration_seconds{profile}`, `rag_hits_total{source,profile}`,
`rag_zero_results_total{profile,reason}`, `rag_top_score{profile}`,
`rag_groundedness`, `rag_index_documents_total{source,status}`,
`rag_index_chunks_total{source}`, `rag_cache_hits_total`.

## Тесты

```bash
python -m pytest tests/rag -q
```

Покрыто: чанкеры (РПД/вакансии/boilerplate), RRF/дедуп/MMR/confidence,
сборка контекста (порог/бюджет/цитаты/санитайз), анализ запроса, оркестрация
индексации на фейковом store (идемпотентность, tombstone, поколения), wiring
в `llm_chat` (цитаты, деградация), аудит ссылок и groundedness, отсечение
выдуманных citation-блоков от модели, L1-кэш (попадание/TTL/ACL-ключ),
передача роли из роутера чата, стадия пайплайна (no-op и устойчивость к сбою).

DDL и SQL проверены на живом PostgreSQL: миграция применяется (16 стейтментов,
HNSW + GIN), upsert идемпотентен, dense/sparse/entity работают на русском FTS,
ACL-фильтр и blue-green переключение поколений корректны.


## Дальше (не сделано намеренно)

- реранк cross-encoder (сейчас RRF+MMR; реранкер требует офлайн-весов и
  отдельного замера латентности);
- профиль `publications` в ингестии (hub-клиент есть, но нужен планировщик
  сбора — логичнее в ночном пайплайне);
- golden-set и метрики recall@k/nDCG: инфраструктура (`rag_query_log`,
  `rag_feedback`) готова, раннер `rag-eval` — следующий шаг;
- `docs/ARCHITECTURE.md` ссылается на несуществующий `src/vector_search/` и
  FAISS — этот слой заменён pgvector-хранилищем, документацию надо поправить.