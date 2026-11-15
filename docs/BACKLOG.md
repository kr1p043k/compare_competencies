# Бэклог проекта Compare Online

Консолидированный список задач, отсортированный по приоритету.
Числа и факты из исходников и файлов данных; ссылки на источники в скобках.

Статус: `[ ]` открыто · `[~]` частично · `[x]` сделано.
Последняя сверка: 2026-11-14, ветка `dev`.

## 0. Сводка статуса

| # | Задача | Статус |
|---|--------|--------|
| 1.1 | Ensemble как прод-модель LTR | `[x]` авто-выбор human-артефакта, суррогат исключён |
| 1.2 | Калибровка (Platt/isotonic) | `[x]` в serving не нужна (rank-only), закрыто |
| 1.3 | Тренды: нормировка на объём | `[ ]` |
| 1.4 | Generalization матчинга (F1 0.99) | `[ ]` |
| 1.5 | Разметка teacher_labeling_task | `[ ]` |
| 1.6 | Покрытие ml/clusters.py | `[x]` тесты добавлены |
| 1.7 | Покрытие teacher_analysis_runner | `[x]` тесты добавлены |
| 2.1 | Degraded-флаг сервисов | `[~]` есть для каналов пайплайна, нет общий middleware |
| 2.2 | Coverage + Quality в UI | `[ ]` |
| 2.3 | Provenance скилла | `[ ]` |
| 2.4 | readiness → readiness_score | `[x]` остатков нет |
| 3.1 | LLM-клиент: единый /v1 | `[x]` реализовано в 7a7b690a, тесты нормализации расширены |
| 3.2 | Чат-хранилище | `[x]` миграция `add_chat_storage.py` |
| 3.3 | Лимит контекста LLM | `[x]` truncation/max_tokens в `llm_client` |
| 3.4 | Rate limit + auth эндпоинтов LLM | `[ ]` |
| 3.5 | UI-чат: plugin-блоки | `[ ]` |
| 3.6 | Тесты LLM | `[x]` 9 файлов в `tests/` |
| 4.1 | ruff | `[~]` 120 issues (было 196) |
| 4.2 | datetime.utcnow → now(UTC) | `[x]` остатков нет |
| 4.3 | .tmp и logs | `[ ]` |

---

## 1. Algorithm (допилы ML/пайплайна)

### 1.1 Продвинутая прод-модель LTR: surrogate → human/ensemble — `[x]`

`ACTIVE_MODEL` = авто-выбор из `_candidate_artifacts()`: берётся feature-совместимый
human-артефакт (`ltr_ranker_ensemble_human.joblib`, NDCG 0.7455), суррогат
`ltr_ranker_xgb_regressor.joblib` (MODEL_FILENAME) исключён из рантайма.
Валидация — `scripts/eval_frozen.py` (CI-гейт). Файл: `src/predictors/ltr_recommendation_engine.py:126-195`.

### 1.2 Калибровка: Platt сломан, isotonic — спасение — `[x]` (закрыто)

`rank_score` в продукте используется только для порядка, калибратор в serving не
подключён. Отчёт `labeling/calibration_report.json` оставлен как исторический.
В `src/` калибровка не участвует — задача снята.

### 1.3 Тренды: нормировка на объём выборки — `[ ]`

Частоты в `trend_snapshots` — абсолютные counts без деления на объём среза. При
росте числа проиндексированных вакансий drift частот ≠ drift спроса.
Файлы: `src/analyzers/skills/trends.py:46`, `src/pipeline/teacher_analysis_runner.py`.
Приоритет: P1.

### 1.4 Generalization матчинга: F1 0.99 подозрителен — `[ ]`

F1 ≈ 0.99 на L3 (semantic) в `data/benchmark/article_datasets/` — вероятен leakage
с настройкой порога 0.78. Нужно: holdout без leakage + report recall при fixed precision.
Файлы: `docs/methodology.md:80-89`, `tests/analyzers/test_adaptive_threshold_exp2.py`,
`src/analyzers/skill_matcher.py:172`. Приоритет: P1.

### 1.5 Разметка: docs/teacher_labeling_task.md — `[ ]`

Для 3 профилей (base, dc, top_dc) проставить «да/нет» на 60 оценок. Цели:
Precision@k, Recall, сравнение версий. Файл: `docs/teacher_labeling_task.md`. P2.

### 1.6 Покрытие ml/clusters.py — `[x]`

Тесты: `tests/.../test_ml_clusters_coverage.py`, `test_train_clusters.py`,
`test_vacancy_clustering.py`, `test_clusters_router_coverage.py`,
`test_clusters_plots_coverage.py`.

### 1.7 Покрытие pipeline/teacher_analysis_runner.py — `[x]`

Тесты: `tests/.../test_teacher_analysis_runner_coverage.py`, `test_teacher_analysis.py`,
`test_teacher_analysis_ksa.py` и др. (16 файлов по teacher).

---

## 2. Trust & Safety

### 2.1 Degraded-флаг для сервисов — `[~]`

`src/pipeline/channel_diag.py` отдаёт API-safe dict `{degraded, failed_channels, count}`
для каналов пайплайна. Общий middleware-деградация (DB down, embedding provider,
LTR model не загружен) с флагом в API-ответах — пока нет.
Файлы: `src/monitoring/`, `src/api_pkg/routers/`. Приоритет: P0.

### 2.2 Coverage + Quality вместе в UI — `[ ]`

`coverage_ratio` (binary) и `readiness_score` (composite) показаны, `quality`
(weighted_coverage) рядом не визуализируется. Файлы: `frontend/src/`,
`src/visualization/coverage.py:22-33`. P1.

### 2.3 Provenance скилла — `[ ]`

Нет сквозной traceability источника навыка (KSA/parsed_skills/it_skills/label).
Нужно поле `source` в `top_market_matched_skills` и `gaps`.
Файлы: `src/analyzers/skill_matcher.py`, `src/pipeline/teacher_analysis_runner.py`. P2.

### 2.4 Переименовать readiness → readiness_score — `[x]`

Остатков `"readiness"` (без `_score`) в `src/` нет.

---

## 3. LLM

### 3.1 Клиент: не задваивать /v1 — `[x]`

Единый `/v1` endpoint вместо параллельных реализаций Ollama/external.
Готово: `LLMClient` нормализует base_url (rstrip + проверка `endswith("/v1")`),
`/v1` добавляется ровно один раз; тесты в `tests/services/test_llm_client.py`.
Файлы: `src/services/llm_client.py`. P1.

### 3.2 Чат-хранилище — `[x]`

Миграция `alembic/versions/add_chat_storage.py` создаёт `chat_threads`, `chat_messages`.

### 3.3 Контекст — `[x]`

`src/services/llm_client.py` содержит truncation/`max_tokens` контроль.

### 3.4 Эндпоинты: rate limit + auth — `[ ]`

Rate limit есть у HH API (`src/parsing/api/hh_api*.py`), но не у LLM-эндпоинтов.
Нужно: auth per user + rate limit per user/IP для chat/completion/embeddings.
Файлы: `src/api_pkg/routers/`. P1.

### 3.5 UI-чат: блоки metric/citation/chart/action_list/text — `[ ]`

Типы блоков жёстко закодированы, без plugin-механизма. Файлы: `frontend/src/`. P2.

### 3.6 Тесты LLM — `[x]`

`tests/api/test_llm_chat_api.py`, `test_llm_cost.py`, `test_llm_phase2.py`;
`tests/services/test_llm_chat.py`, `test_llm_chat_guardrails.py`, `test_llm_client.py`,
`test_llm_guardrails.py`, `test_llm_cache_coverage.py`, `test_llm_pricing_coverage.py`.

---

## 4. Hygiene

### 4.1 ruff — `[~]`

`ruff check .` → 120 issues (было 196; 28 fixable). Прогресс есть, закрыть остаток.
Файлы: `pyproject.toml` (ruff config). P1.

### 4.2 datetime.utcnow() → datetime.now(UTC) — `[x]`

Вхождений `utcnow` в `src/` нет.

### 4.3 .tmp и logs — `[ ]`

`pre-commit` hook + `.dockerignore` coverage для `.tmp_*`, `*.tmp`, `logs/`, `*.log`.
Файлы: `.gitignore`. P2.
