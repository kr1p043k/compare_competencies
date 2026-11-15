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
| 3.1 | LLM-клиент: единый /v1 | `[ ]` |
| 3.2 | Чат-хранилище | `[x]` миграция `add_chat_storage.py` |
| 3.3 | Лимит контекста LLM | `[x]` truncation/max_tokens в `llm_client` |
| 3.4 | Rate limit + auth эндпоинтов LLM | `[ ]` |
| 3.5 | UI-чат: plugin-блоки | `[ ]` |
| 3.6 | Тесты LLM | `[x]` 9 файлов в `tests/` |
| 4.1 | ruff | `[x]` 120 → 0 (конфиг проекта) |
| 4.2 | datetime.utcnow → now(UTC) | `[x]` остатков нет |
| 4.3 | .tmp и logs | `[ ]` |
| 5.1 | F601: задвоенный ключ `hadoop` (баг в synonym map) | `[x]` + huggingface/html/css → F601=0 |
| 5.2 | ruff: разбивка остатка (E501/N806/SIM105/...) | `[x]` остаток 0 |
| 5.3 | E402: импорты не в начале файла (scripts/seed_all_directions.py и др.) | `[x]` |
| 5.4 | `print()` в prod-коде → structured logging (~340) | `[ ]` |
| 5.5 | Крупные модули: разбить router-файлы | `[ ]` |
| 5.6 | Frontend: `any`-типы (101 место) | `[ ]` |
| 5.7 | Аудит широких `except` (202) на проглатывание ошибок | `[ ]` |
| 6.* | Расширенный бэклог: security/API/DB/данные/фронт/ML/infra/docs (~35 подзадач) | `[~]` часть уже сделана (см. §8) |
| 7.* | Глубокий бэклог: async/frontend/observability/config/tests/UX/reliability (~35 подзадач) | `[~]` часть уже сделана (см. §8) |
| 8 | Верификация: снятые/переуточнённые задачи | — |
| 9.* | Аудит по расширенным правилам ruff (correctness/errors/logging/complexity/perf/tests) | `[ ]` |
| 10.1 | Синхронизация `main` → `dev` (ESLint + 3 коммита) | `[ ]` |

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

### 3.1 Клиент: не задваивать /v1 — `[ ]`

Единый `/v1` endpoint вместо параллельных реализаций Ollama/external.
Файлы: `src/`, `docs/LLM_PHASE2.md`. P1.

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

---

## 5. Найдено при сканировании (2026-11-14)

### 5.1 F601: задвоенный ключ `hadoop` в synonym map — `[ ]`

`ruff F601` (4 срабатывания): ключ `"hadoop"` объявлен дважды в
`SkillNormalizer.SYNONYM_MAP` — второй перекрывает первый, правила теряются.
Возникло при расширении карты (коммиты `30c483e0`, `594799b4`). Нужно: слить
значения в один ключ, перечитать `--select F601` → 0.
Файлы: `src/parsing/skills/skill_normalizer.py:170`. Приоритет: P1 (баг).

### 5.2 ruff: разбивка остатка — `[ ]`

`ruff check .` = 120 issues. Топ-правила:
`E501` line-too-long 28 · `N806` non-lowercase-var 27 · `I001` unsorted-imports 12 (*)
· `SIM105` suppressible-exception 10 · `UP007` non-pep604-union 9 (*) · `E402` 4 ·
`E741` ambiguous-name 4 · `F601` 4 · `B007` 3 · `B905` zip-without-strict 3 ·
`UP035` 3 (*) · `W292` 3 (*) · прочее 11. `(*)` — авто-фиксится.
Файлы: `pyproject.toml`. P1.

### 5.3 E402: импорты не в начале файла — `[ ]`

4 срабатывания (например `scripts/seed_all_directions.py:22` — `sys.stdout.reconfigure`
до импортов). Нужно: вынести reconfigure после импортов либо `# noqa: E402`.
Файлы: `scripts/`. P2.

### 5.4 print() в prod-коде → structured logging — `[ ]`

~340 вызовов `print(` в `src/` (пайплайн, анализаторы, api). Есть `src/logging_config.py`
(structlog + маскировка секретов), но используется точечно. Нужно: заменить `print`
на `log.*` в рантайм-путях, оставить `print` только в CLI-скриптах.
Файлы: `src/logging_config.py`, `src/`. P2.

### 5.5 Крупные модули: разбить router-файлы — `[ ]`

Размеры (KB): `api_pkg/routers/admin.py` 83 · `pipeline/teacher_analysis_runner.py` 81 ·
`predictors/ltr_recommendation_engine.py` 64 · `api_pkg/routers/teacher.py` 62 ·
`api_pkg/routers/zun.py` 62 · `routers/auth.py` 48 · `routers/pipeline.py` 47 ·
`routers/profiles.py` 45. Риск: сложность навигации/ревью, скрытые зависимости.
Нужно: вынести под-домены в отдельные модули (admin/monitoring, admin/llm, teacher/gap, ...).
Файлы: `src/api_pkg/routers/`, `src/pipeline/`. P2.

### 5.6 Frontend: `any`-типы — `[ ]`

~101 место `: any` / `as any` в `frontend/src`. Нужно: заменить на точные типы/`unknown`
+ сужение; включить строгие правила в `tsconfig`. Файлы: `frontend/src/`. P2.

### 5.7 Аудит широких `except` — `[ ]`

202 `except:`/`except Exception:` в `src/`. Часть проглатывает ошибки без лога.
Нужно: пройти по критичным путям (pipeline, api, parsing), оставить только осознанные
fallback-и с логированием; остальные — сузить до конкретных исключений.
Файлы: `src/`. P2.

---

## 6. Расширенный бэклог (2026-11-14)

Собрано сканированием кода/инфры. Формат: факт → что сделать → приоритет.

### 6.1 Безопасность

- **6.1.1 Auth-покрытие эндпоинтов** — 220 `@router.*` роутов, 138 guard-вызовов,
  только 10 роутеров задают `dependencies=` на уровне `APIRouter`. Нужно: перепись
  эндпоинтов без явного guard, добавить роутерные зависимости или `require_authenticated`.
  Файлы: `src/api_pkg/routers/*`. P1.
- **6.1.2 Аудит f-string SQL** — 6 мест с интерполяцией в `execute`/`text`
  (`admin.py:481` where_clause, `cli/populate_parsed_skills.py`), подтвердить, что все
  они на bind-параметрах и не берут пользовательский ввод. P1.
- **6.1.3 Rate limiting** — публичные/дорогие эндпоинты (chat, embedding, триггеры
  pipeline, forecast) без лимитов. Добавить per-IP/per-user. Файлы: `src/api_pkg`. P1.
- **6.1.4 Dependency audit в CI** — `pip-audit` (py) + `npm audit` (fe) как гейт.
  Файлы: `.github/workflows/ci.yml`. P2.
- **6.1.5 Аудит JWT** — TTL/refresh/revocation; проверить `require_role` на подмену роли.
  Файлы: `src/api_pkg/routers/auth.py`. P2.

### 6.2 API / backend quality

- **6.2.1 `response_model` покрытие** — проверить, что все роуты декларируют схему ответа
  (иначе дрейф контрактов). Связь: `docs/API-CHANGELOG.md`. P1.
- **6.2.2 Пагинация** — аудит списочных эндпоинтов на unbounded `LIMIT`. P2.
- **6.2.3 Единый error-envelope** — уже добавлен `code`+`request_id` (`525f35d8`);
  убедиться, что ВСЕ роутеры и middleware используют его, а не голый `HTTPException`. P2.
- **6.2.4 Идемпотентность** — повторные триггеры pipeline/collection не должны дублировать
  прогон. Ключ запроса + защита. P2.
- **6.2.5 Долгие задачи** — вынести пайплайн из `BackgroundTasks` в очередь (RQ/arq),
  иначе процесс держится и не ретраится. P2.

### 6.3 БД

- **6.3.1 Индекс-аудит** — горячие запросы (`parsed_skills`, `vacancies.raw`): проверить
  наличие индексов и `EXPLAIN`. P1.
- **6.3.2 Дрейф моделей/миграций** — `alembic check` в CI; 40 миграций, ранее были
  «дрейфовые» колонки. P1.
- **6.3.3 N+1** — аудит `teacher_analysis_runner.py` (1521 строк, ~15 этапов). P1.
- **6.3.4 Пул соединений** — размер/таймауты под нагрузку. Связь: `docs/LOAD-BASELINE.md`. P2.

### 6.4 Данные/пайплайн

- **6.4.1 `.keyskills` backfill** — в проде `key_skills` пуст; пайплайн пишет
  `extracted_skills`, но не `key_skills`. Решить: заполнять или убрать поле. P1.
- **6.4.2 Retry/backoff HH API** — сбор устойчивый к 429/5xx. Файлы: `src/parsing/api/*`. P2.
- **6.4.3 Regression-снапшоты нормализации** — зафиксировать universe навыков (1181) и
  результат нормализации тестом, чтобы правки не «разъезжали» данные. P2.
- **6.4.4 Консистентность whitelist** — `it_skills.json` / `skill_taxonomy.json` / таблица
  `skills`: единый источник правды + тест на расхождение. P2.

### 6.5 Frontend

- **6.5.1 Тест-покрытие компонентов** — 112 `.tsx`, 42 тест-файла (~37%). Довести ядро
  (Vacancies/Teacher/Admin/Student) до ≥80%. P1.
- **6.5.2 Error boundaries** — на все lazy-роуты (уже ловили «Failed to fetch dynamically
  imported module»). P1.
- **6.5.3 a11y-аудит** — axe на ключевых страницах (контраст уже правили `cfa25e07`). P2.
- **6.5.4 Bundle budget** — code-split тяжёлых чартов (recharts), лимит в CI. P2.
- **6.5.5 Empty/loading/error-состояния** — аудит таблиц: единый паттерн. P2.

### 6.6 ML

- **6.6.1 Model registry** — версионирование артефактов в serving (manifest, хэш). Связь:
  `docs/MODEL_REGISTRY.md`. P1.
- **6.6.2 Дрифт-мониторинг** — джоб + алерт (n8n `trend_alert.json` уже есть) на падение
  NDCG/смену распределения. P2.
- **6.6.3 Детерминизм обучения** — фиксировать seed; воспроизводимость `make_ensemble.py`. P2.
- **6.6.4 Кэш объяснений** — SHAP/`ml-explanations` кэшировать по (model,skill). P2.

### 6.7 Infra / CI

- **6.7.1 CI-гейты** — PR: `ruff`, `mypy`, `pytest --cov-fail-under=80`, `tsc`, `vitest`.
  Часть есть (`ci.yml`) — дополнить порогами. P1.
- **6.7.2 Docker hardening** — non-root user, healthcheck для всех сервисов, pinned base. P2.
- **6.7.3 Grafana alerting** — правила алертов (сейчас только дашборд `competency-dashboard`). P2.
- **6.7.4 Load-test пороги** — `load-test.yml`: зафиксировать SLO и падать при регрессии. P2.
- **6.7.5 n8n workflows** — 11 json: валидация, версионирование, тесты цепочек. P2.

### 6.8 Docs

- **6.8.1 API-CHANGELOG upkeep** — политика обязательного обновления при изменении роутов. P2.
- **6.8.2 RUNBOOK расширить** — сценарии инцидентов (DB down, провайдер 5xx, ML не загрузился). P2.
- **6.8.3 Архитектурная диаграмма** — drawio в `docs/` (актуализировать ARCHITECTURE.md). P2.

---

## 7. Глубокий бэклог (2026-11-14, раунд 2)

### 7.1 Async / устаревшие API

- **7.1.1 `asyncio.get_event_loop()` → `get_running_loop()`/`asyncio.run()`** — deprecated,
  ломается на 3.12+: `parsing/api/hh_api_async.py:337,400`, `pipeline/runner.py:62,664-688`,
  `notifications/system.py:312`, `api_pkg/request_logger.py:268`. P1.
- **7.1.2 Sync-bridge `run_until_complete` в async-контексте** — переписать на нативный await,
  убрать event-loop juggling. `pipeline/runner.py`, `hh_api_async.py`. P1.
- **7.1.3 Разобрать `noqa`/`# type: ignore`** — 31 место: заменить реальными фиксами,
  где возможно. P2.

### 7.2 Frontend

- **7.2.1 ESLint — уже есть в `main`, НЕТ в `dev`.** `frontend/eslint.config.js` +
  `package.json` (`lint: eslint .`, `@eslint/js`, `eslint-plugin-react-hooks`,
  `typescript-eslint`) + CI-джоба `Lint (npm run lint)` присутствуют в `origin/main`,
  но отсутствуют в `origin/dev`. Задача = **синхронизировать `main` → `dev`** (см. 7.9.1),
  а не «добавить с нуля». P1.
- **7.2.2 Дробление гигантских компонентов** — `TeacherDashboard.tsx` 103KB(!), `ProfilePages`
  59KB, `VacanciesList` 58KB, `AdminUsersTab` 54KB, `AdminDashboard` 49KB, `FaqPage` 48KB,
  `LoginPage` 34KB. Вынести секции/таблицы/формы. P1.
- **7.2.3 Порог покрытия vitest** — задать `coverage.thresholds` и гейт. P2.
- **7.2.4 Кэш/дедуп запросов** — аудит fetch-штормов (TanStack Query или ручной дедуп). P2.
- **7.2.5 Виртуализация длинных списков** — вакансии/навыки (`react-window`). P2.

### 7.3 Наблюдаемость

- **7.3.1 Prometheus `/metrics`** — подтвердить, что не публичен в проде; добавить бизнес-метрики
  (длительность пайплайна, латентность матчинга, ошибки LLM). P1.
- **7.3.2 request_id сквозной** — прокинуть в логи/ответы/ошибки (частично есть, `525f35d8`). P2.
- **7.3.3 Error-tracking (Sentry)** — сбор исключений фронта и бэка. P2.
- **7.3.4 Health split** — `/health/live` vs `/health/ready` (БД/модель/провайдер). Связь: 2.1. P2.

### 7.4 Конфиг / эксплуатация

- **7.4.1 Fail-fast валидация env** — `pydantic-settings` обязательные переменные, падать на старте
  с понятной ошибкой. P1.
- **7.4.2 Feature flags** — механизм включения/выключения фич без редеплоя. P2.
- **7.4.3 `.env.example` синхронность** — тест: все ключи из кода присутствуют в примере. P2.
- **7.4.4 Секреты не в образе** — секреты только через env/secret-store, не в Docker-слое. P2.

### 7.5 Тестирование

- **7.5.1 Backend coverage ≥80% как гейт** — включить `--cov-fail-under=80` в CI + отчёт по модулям. P1.
- **7.5.2 Contract-тесты** — ответы API валидируются против схем (`response_model`). Связь: 6.2.1. P2.
- **7.5.3 Property-based тесты** — `hypothesis` для `skill_normalizer` (идемпотентность нормализации). P2.
- **7.5.4 Frontend coverage ≥80% ядра** — связь: 6.5.1. P2.
- **7.5.5 Load/soak-тесты** — сценарии на 60 мин, деградация памяти/латентности. P2.

### 7.6 Данные / UX

- **7.6.1 Поиск навыков** — fuzzy-search endpoint (autocomplete по `skills`/`parsed_skills`). P2.
- **7.6.2 Экспорт CSV/Excel** — выгрузка таблиц (вакансии, компетенции, gap). P2.
- **7.6.3 URL-state таблиц** — сортировка/фильтр/страница в query (шарящиеся ссылки). P2.
- **7.6.4 PDF-отчёт: формат/i18n** — унифицировать `reports/pdf_report.py`, встроить шрифты. P2.

### 7.7 Docs

- **7.7.1 DESIGN-SYSTEM.md** — дописать (в файле есть placeholder/WIP). P3.
- **7.7.2 ONBOARDING актуализация** — сверить с текущим `dev` (структура/команды). P2.
- **7.7.3 CONTRIBUTING + PR-template** — процесс, чек-лист ревью, коммит-стиль. P3.
- **7.7.4 ADR-лог** — `docs/adr/` есть? вести решения по архитектуре. P3.

### 7.8 Надёжность

- **7.8.1 Проверка бэкапов в CI** — `scripts/verify_backup.ps1` в регулярный прогон. P2.
- **7.8.2 Retention enforcement** — исполнение политики из `docs/RETENTION.md`. P2.
- **7.8.3 Graceful shutdown** — дренаж in-flight запросов/задач при остановке. P2.
- **7.8.4 Таймауты исходящих HTTP** — единый httpx-клиент с connect/read timeout + ретраями. P1.

---

## 8. Верификация (2026-11-14): что уже реализовано в dev

Проверка разделов 6-7 против кода `dev` (`594799b4`). Эти пункты **сняты** или
**переуточнены** - не выдавать как открытые.

### 8.1 Уже сделано (снять)

- **6.1.3 Rate limiting - СДЕЛАНО.** `slowapi` подключён: 191 `@limiter.limit(...)`
  по роутерам + `RATELIMIT_ENABLED` (`src/config.py:82`, `src/api_pkg/__init__.py:100`).
  Осталось (если хочется): проверить покрытие именно LLM-эндпоинтов. → связь с 3.4.
- **6.1.4 Dependency audit в CI - СДЕЛАНО.** `.github/workflows/ci.yml:51-55` (`pip-audit`),
  `:80-82` (`npm audit --audit-level=high`).
- **6.2.3 Единый error-envelope - СДЕЛАНО.** `_error_body()` + `request_id` middleware
  (`src/api_pkg/__init__.py:49-56,270`), 4 `exception_handler`.
- **6.3.2 (часть) alembic в CI - СДЕЛАНО** для heads: `ci.yml:109-118` (single-head + upgrade).
  Осталось только *drift* `alembic check` (модели↔БД) - см. 8.2.
- **6.4.2 Retry/backoff HH API - СДЕЛАНО.** `urllib3.util.retry.Retry(backoff_factor=1)` +
  token-retry (`src/parsing/api/hh_api.py:13-50,200`).
- **7.3.1 `/metrics` защищён - СДЕЛАНО** (не публичен): `METRICS_TOKEN` / роль admin
  (`src/api_pkg/__init__.py:59-100,458-491`). Остаются только бизнес-метрики - см. 8.2.
- **7.3.4 Health split - СДЕЛАНО.** `/health`, `/ready`, `/live` (`src/api_pkg/routers/health.py:116,186`).
- **6.5.2 Error boundaries - СДЕЛАНО** (базово): `design/ErrorBoundary.tsx` + тест,
  используется в `App.tsx`/`main.tsx`. Осталось: покрыть ВСЕ lazy-роуты - см. 8.2.
- **7.5.1 (часть) coverage gate - ЕСТЬ**, порог `--cov-fail-under=68` (`ci.yml:160`).

### 8.2 Переуточнено (не дублировать как «сделать с нуля»)

- **6.1.2 f-string SQL (6 мест)** - везде уже передаются bind-`params`
  (`admin.py:481` - `where_clause` строится из enum-фильтров). Задача = **аудит**, не рефактор.
- **6.2.1 `response_model`** - 49 объявлений на 220 роутов. Цель сузить: закрыть списочные/схемные
  эндпоинты (не все 220).
- **6.4.1 `key_skills`** - `db_writer.py:298` **пишет** `key_skills`; пусто в проде, т.к. HH *basic*
  выдача не содержит `key_skills` (см. docstring `db_writer.py:193-194`). Задача = **backfill**
  для basic-вакансий (есть `raw.extracted_skills`), не «добавить запись поля».
- **7.4.1 fail-fast env** - уже `pydantic BaseSettings` + `field_validator` (`config.py:10-33`).
  Задача = покрыть обязательные креды явными `Field(...)` и fail-fast.
- **7.5.1** - поднять порог покрытия **68 → 80** (гейт уже есть).
- **7.8.4 HTTP-таймауты** - во многих местах уже `timeout=` задан; задача = **унифицировать**
  в один клиент, а не «добавить таймауты».
- **3.4 LLM rate limit** - проверить, покрыты ли `llm.py`/`chat.py` лимитерами (инфра slowapi есть).
- **7.2.1 ESLint** - уже реализован в `origin/main` (`frontend/eslint.config.js` + CI-джоба),
  отсутствует в `dev`. Задача = синхронизировать `main` → `dev` (см. §10.1), не «внедрять с нуля».

---

## 9. Аудит по расширенным правилам ruff (2026-11-14, раунд 3)

Все пункты - реальные текущие срабатывания на `dev` (не сделано). Команда:
`python -m ruff check . --select <семейство>`.

### 9.1 Корректность (баг-класс)

- **9.1.1 `RUF006` asyncio-dangling-task (10)** — `asyncio.create_task(...)` без сохранения
  ссылки (сборщик может убить задачу). Места: `api_pkg/request_logger.py:567,584`,
  `routers/admin.py:1874,1886`, `routers/pipeline.py:623`, `startup.py:249,314,316,361`,
  `student_actions.py:57`. P1.
- **9.1.2 `RUF012` mutable-class-default (31)** — изменяемые атрибуты класса без `ClassVar`:
  `parsing/skills/skill_normalizer.py` (много), `analyzers/skills/skill_taxonomy.py:25-26`,
  `analyzers/gap/gap_analyzer.py:21`, `logging_config.py:18`,
  `analyzers/comparison/embedding_provider.py:88`. P2.
- **9.1.3 `DTZ001/005/007` tz-naive datetime (41+39+13)** — `datetime(...)`/`datetime.now()`/
  `strptime` без tz. Сделать все UTC-aware. Инвентарь: `--select DTZ`. P1.
- **9.1.4 `PLW2901` redefined-loop-name (20)** — переиспользование имени цикла. P2.
- **9.1.5 `PLW0603` global-statement (58)** — сократить мутабельные глобалы. P2.

### 9.2 Обработка ошибок

- **9.2.1 `TRY400` error-instead-of-exception (56)** — в `except` использовать `logger.exception`
  (с трейсбеком), а не `logger.error`. P2.
- **9.2.2 `TRY003` raise-vanilla-args (163)** — длинные сообщения в `raise` без своего класса
  исключения. Ввести доменные исключения. P3.
- **9.2.3 `TRY301/TRY300` (36/48)** — `raise` внутри `try` / перенос в `else`. P3.

### 9.3 Логирование

- **9.3.1 `G004` logging-f-string (17)** — f-строки в логах вместо `%`/structlog:
  `parsing/skills/skill_embedding_cache.py:53-74`, `hybrid_weight_calculator.py:54,116`,
  `analyzers/clustering/vacancy_clustering.py:355`. P2.

### 9.4 Сложность / поддерживаемость

- **9.4.1 `PLR0912/PLR0915` (84/75)** — too-many-branches/statements: разбить функции. P2.
- **9.4.2 `PLR0913` too-many-arguments (56)** — ввести dataclass/Pydantic-params. P2.
- **9.4.3 `PLR0911` (22) + `PLR2004` magic-value (1901)** — вынести константы. P3.
- **9.4.4 `PLC0415` import-outside-top-level (1473)** — ревизия ленивых импортов (оставить
  осознанно или вынести вверх). P3.
- **9.4.5 автофиксы `PLW0108/C4*/PIE807/RET5*/RUF100`** — прогнать `ruff --fix` пачкой
  (в т.ч. `RUF100` unused-noqa 159). P3.

### 9.5 Производительность

- **9.5.1 `PERF401` (38) + `PERF102` (16)** — ручные циклы → comprehension/`dict`-итерация. P2.

### 9.6 Стиль тестов

- **9.6.1 `PT018` pytest-composite-assertion (153)** — разбить составные `assert`. P3.
- **9.6.2 `PT001/PT022/PT011/PT023`** — стиль фикстур и `pytest.raises`. P3.

---

## 10. Синхронизация веток

### 10.1 `main` → `dev` (рассинхрон) — `[ ]`

`origin/main` содержит 3 коммита (UI-фильтры/сортировки TeacherDashboard, фикс
`merge_all_heads`), которых нет в `dev`. В них, среди прочего, **ESLint-конфиг и CI-джоба**
(см. 7.2.1). Нужно: смёржить `main` в `dev`, разрешить конфликты (`TeacherDashboard.tsx`,
`merge_all_heads.py`), прогнать `npm run lint` + `tsc` + `vitest` + `pytest`, запушить.
P1. Подтверждает: ESLint есть в `main` (`frontend/eslint.config.js`), в `dev` - нет.









---

## 11. Сделано в батче 1 (2026-11-15)

Ветка `chore/backlog-batch1` -> merge в `dev` (локально, push под хуком).

1. **5.1** F601 задвоенные ключи: слиты `hadoop`, `huggingface`, `html`, `css`; `F601 = 0`.
2. **4.1 / 5.2** ruff: **120 -> 0** (проектный конфиг) - автофиксы + ручные + per-file-ignores.
3. **5.3** E402 (scripts) - убраны (`# noqa` / перенос импортов).
4. **E741** неоднозначные `l` -> `lvl` (`benchmark_matching.py`).
5. **UP031** %-формат -> f-строки (`merge_annotations_to_krm.py`, `check_votes.py`).
6. **B007** неиспользуемая переменная цикла (`benchmark_matching.py`).
7. **E702** точка с запятой -> две строки (`adjudicate.py`).
8. **N818** исключение `DSNNotConfigured` -> `DSNNotConfiguredError` (+ ссылки).
9. **SIM105** `try/except/pass` -> `contextlib.suppress` (`clean_intake_data.py`).
10. **SIM108** `if/else` -> тернарник (`merge_annotations_to_krm.py`).
11. **SIM115** контекст-менеджер/`noqa` (`calibrate_ensemble.py`).
12. **E501 (core)** длинная строка `skill_normalizer.py:116` разбита.
13. **N806/E501 (скрипты)** per-file-ignores для `scripts/**`, `alembic/**`.

Проверка: `ruff check .` = 0; `pytest tests/parsing tests/analyzers` = 1100 passed;
`pytest tests/pipeline tests/services` = 631 passed.

### Поправка к §7.1
**7.1.1 УЖЕ СДЕЛАНО**: `asyncio.get_event_loop()` в `src/` отсутствует (только в
docstring `hh_api_async.py:24`); используется хелпер с `get_running_loop`. Снять из открытых.
`run_until_complete` (`hh_api_async.py:39`, `pipeline/runner.py:62`) - осознанный sync-bridge.


---

## 12. Сделано в батче 2 (2026-11-15)

1. **9.1.1 RUF006** - fire-and-forget: src/api_pkg/bg_tasks.py:spawn() держит ссылку; 10 мест. RUF006=0.
2. **9.3.1 G004** - f-строки в логах -> structlog kwargs (7 мест). G004(src)=0.
3. **DTZ005** - datetime.now() -> now(UTC) в admin.py. DTZ005=0.
4. **9.1.2 RUF012** - мутабельные class-атрибуты -> ClassVar (7 файлов). RUF012=0.
5. **9.1.4 PLW2901** - переиспользование loop-переменной (11 мест). PLW2901(src)=0.
6. **4.3** .dockerignore - добавлены *.tmp, .tmp_*, *.log, logs_*.txt.
7. **5.4 (частично)** src/ml/clusters.py - print -> logger.
8. **6.4.3** тест идемпотентности нормализации.
9. **6.4.4** тест консистентности whitelist (замороженный дрейф).
10. **БАГ node** - дубли канонов nodejs/node.js ломали идемпотентность; канон сведён к node.js.

Проверка: ruff=0; pytest tests/parsing 432 passed (+2 xfail).
