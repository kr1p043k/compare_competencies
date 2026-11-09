# Бэклог проекта Compare Online

Консолидированный список задач, отсортированный по приоритету.
Числа и факты — из исходников и файлов данных; ссылки на источники в скобках.

---

## 1. Algorithm (допилы ML/пайплайна)

### 1.1 Продвинутая прод-модель LTR: surrogate → human/ensemble

Сейчас прод-модель — `XGBRegressor` на суррогатном таргете `0.6×relevance + 0.4×base_target` (`docs/methodology.md:56-72`). Флагманский артефакт `ltr_ranker_ensemble_human.joblib` обучен на human-оценках из `labeling/gold.csv` (4 роли × 125 пар) через `scripts/experiments/make_ensemble.py`. Задача — задеплоить ensemble как основную модель вместо surrogate, провалидировать на frozen split через `scripts/eval_frozen.py`.

- **Файлы:** `scripts/experiments/make_ensemble.py`, `scripts/eval_frozen.py`, `labeling/gold.csv`, `labeling/battery3.py`, `src/predictors/ltr_recommendation_engine.py:142`
- **Приоритет:** P1

### 1.2 Калибровка: Platt сломан, isotonic — спасение

Модуль калибровки `scripts/experiments/calibrate_ranker.py` тестирует Platt, isotonic, raw_minmax и OOF-варианты (`labeling/calibration_report.json`):

| Вариант | log_loss | Brier |
|---------|----------|-------|
| platt | 13.2348 | 0.3672 |
| isotonic | 0.7674 | 0.2380 |
| raw_minmax | 0.7138 | 0.2539 |
| oof_platt | 12.3900 | 0.3438 |
| oof_isotonic | 0.6420 | 0.2260 (winner) |

Platt (log_loss 13.23) и oof_platt (12.39) катастрофически плохо калибруют — модель выдаёт либо 0, либо 1 без градиента. Победитель — `oof_isotonic` (log_loss 0.642). Нужно: (1) убрать platt из serving, (2) добавить oof_isotonic в serving pipeline, (3) документировать.

- **Файлы:** `scripts/experiments/calibrate_ranker.py`, `labeling/calibration_report.json`
- **Приоритет:** P0

### 1.3 Тренды: нормировка на объём выборки

Частоты в `trend_snapshots` считаются как абсолютные counts без деления на объём выборки за срез. При изменении числа проиндексированных вакансий drift частот ≠ drift спроса — можно наблюдать «рост» навыка только потому что проиндексировано больше вакансий вообще.

- **Файлы:** `src/analyzers/skills/trends.py`, `src/pipeline/teacher_analysis_runner.py:1162-1188`
- **Приоритет:** P1

### 1.4 Generalization матчинга: F1 0.99 подозрителен

Ф1 ≈ 0.99 на L3 (semantic stage) в бенчмарке `data/benchmark/article_datasets/` (800 пар KSA→market skill) — подозрительно высоко. Вероятно, пары разметки близки к данным настройки порога 0.78 (`docs/methodology.md:84-89`). Текущий score.py (`labeling/score.py`) и Exp2 (`tests/analyzers/test_adaptive_threshold_exp2.py`) показывают: F1-пик на adaptive threshold 0.45 = 0.859; recall@0.70 лишь 0.44, @0.55 — 0.69. Нужно: (1) раздельный holdout без leakage, (2) report общего recall при fixed precision.

- **Файлы:** `docs/methodology.md:80-89`, `tests/analyzers/test_adaptive_threshold_exp2.py`, `src/analyzers/skill_matcher.py:172`, `src/analyzers/hybrid_matcher.py:33-35`
- **Приоритет:** P1

### 1.5 Разметка: docs/teacher_labeling_task.md

Задача валидации рекомендаций пайплайна: для 3 профилей (base, dc, top_dc) проставить «да/нет» на 60 оценок (10-15 мин/чел). Цели: Precision@k, Recall, объективное сравнение версий. Файл `docs/teacher_labeling_task.md` содержит ТЗ.

- **Файлы:** `docs/teacher_labeling_task.md`
- **Приоритет:** P2

### 1.6 Покрытие ml/clusters.py

Модуль `src/ml/clusters.py` (220 строк) — единственная точка тренировки vacancy clustering. Нет unit-тестов, покрывается только integration-тестами пайплайна. Функция `train_clusters()` содержит 3 print-блока и нет try/except вокруг ключевых операций.

- **Файлы:** `src/ml/clusters.py`
- **Приоритет:** P2

### 1.7 Покрытие pipeline/teacher_analysis_runner.py

Модуль 1521 строка — main runner teacher analysis. Огромная функция `run_teacher_analysis()` содержит ~15 этапов (DB → market skills → disciplines → matching → gap → enhanced gap → charts → write). Нет модульных тестов отдельных этапов, покрытие через integration-тесты (`tests/pipeline/`). Риск: любой stage-fail ломает весь pipeline без recover.

- **Файлы:** `src/pipeline/teacher_analysis_runner.py`
- **Приоритет:** P1

---

## 2. Trust & Safety

### 2.1 Degraded-флаг для сервисов

Нет механизма уведомления UI о деградации сервиса (DB down, embedding provider недоступен, LTR model не загружен). При partial-failure UI показывает устаревшие/неполные данные как нормальные. Нужно: middleware-деградация + флаг в API-ответах.

- **Файлы:** `src/monitoring/`, `src/api_pkg/routers/`
- **Приоритет:** P0

### 2.2 Coverage + Quality вместе в UI

Сейчас в интерфейсе показываются `coverage_ratio` (binary: сколько есть) и `readiness_score` (composite index). Отсутствует визуализация `quality` (weighted_coverage) рядом с coverage — большой разрыв «coverage высокое, quality низкий» не очевиден для пользователя.

- **Файлы:** `frontend/src/`, `src/visualization/coverage.py:22-33`
- **Приоритет:** P1

### 2.3 Provenance скилла

Нет сквозной traceability: от какого источника (KSA из БД, parsed_skills из HH, it_skills.json, human labeling) пришёл навык в recommendation. Нужно: поле `source` в `top_market_matched_skills` и `gaps`.

- **Файлы:** `src/analyzers/skill_matcher.py`, `src/pipeline/teacher_analysis_runner.py:993-1001`
- **Приоритет:** P2

### 2.4 Переименовать readiness → readiness_score

Поле в API и UI называется `readiness`, но внутри контрактов `readiness_score` (`src/models/data_contracts.py:93`, `src/models/student.py:65`). В UI это приводит к inconsistency label. Нужно унифицировать на `readiness_score` везде.

- **Файлы:** `src/models/api_responses.py:51,187`, `src/models/data_contracts.py:93`, `src/models/student.py:65`, `src/reports/pdf_report.py:136`, `grafana/provisioning/dashboards/competency-dashboard.json:631`
- **Приоритет:** P1

---

## 3. LLM

### 3.1 Клиент: не задваивать /v1

LLM-клиент должен использовать единый `/v1` endpoint, а не дублировать логику для Ollama и external providers. Сейчас `src/analyzers/` и `src/services/` могут содержать параллельные реализации.

- **Файлы:** `src/`, `docs/LLM_PHASE2.md`
- **Приоритет:** P1

### 3.2 Чат-хранилище

Нет persistent storage для LLM chat sessions. Текущая реализация не сохраняет историю диалогов. Нужно: таблица в БД + API endpoint для chat history.

- **Файлы:** `docs/LLM_PHASE2.md:40`
- **Приоритет:** P2

### 3.3 Контекст

LLM-контекст (system prompt, retrieved documents) не имеет лимита на длину. Нет mechanism для truncation при превышении context window. Нужно: `max_tokens` control + smart truncation.

- **Файлы:** `src/services/`, `src/api_pkg/routers/`
- **Приоритет:** P1

### 3.4 Эндпоинты

LLM API-эндпоинты (chat, completion, embeddings) не имеют rate limiting и auth. Нужно: API key auth per user + rate limit per user/IP.

- **Файлы:** `src/api_pkg/routers/`
- **Приоритет:** P1

### 3.5 UI-чат: блоки metric/citation/chart/action_list/text

В UI-компонентах чата жёстко закодированы типы блоков (metric, citation, chart, action_list, text) без extensibility mechanism. Нужно: plugin-architecture для кастомных блоков.

- **Файлы:** `frontend/src/`
- **Приоритет:** P2

### 3.6 Тесты LLM

Нет тестов для LLM-интеграции. Нет mocking для external providers. Нужно: unit-тесты для LLM client с mocks.

- **Файлы:** `tests/`
- **Приоритет:** P2

---

## 4. Hygiene

### 4.1 ruff: 196 issues

Ruff (через `tasks.ps1 lint`) находит ~196 issues. Нужно: `ruff check .` → fix top-N → закомитить.

- **Файлы:** `pyproject.toml` (ruff config)
- **Приоритет:** P1

### 4.2 datetime.utcnow() → datetime.now(UTC)

2 места используют deprecated `datetime.utcnow()` (`src/notifications/system.py:67`, `src/n8n/webhooks.py:133`). Нужно заменить на `datetime.now(UTC)` и проставить timezone-awareness.

- **Файлы:** `src/notifications/system.py:67`, `src/n8n/webhooks.py:133`
- **Приоритет:** P0

### 4.3 .tmp и logs

`.gitignore` (.tmp_*, *.tmp, logs/, *.log) — файлы не коммитятся, но на CI могут попадать. Нужно: pre-commit hook check + .dockerignore coverage.

- **Файлы:** `.gitignore:11-19,66,96-97`
- **Приоритет:** P2