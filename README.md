# Competency Gap Analyzer

Анализ соответствия учебных компетенций студентов требованиям IT-рынка (hh.ru).

Собирает вакансии, нормализует навыки, выполняет gap-анализ, формирует персонализированные рекомендации через ML (XGBoost + SHAP) с Prometheus/Grafana-мониторингом, n8n-автоматизацией и LLM-интеграцией (внешняя Ollama — `qwen3.6:latest`, 36B, сервер университета).

## Возможности

- **Сбор вакансий** — hh.ru API, синхронный и асинхронный клиенты, 2000 вакансий на запрос
- **Нормализация навыков** — синонимы, fuzzy-матчинг, BM25 + SentenceTransformer + PCA
- **Таксономия** — 19 категорий навыков + профессии с привязкой к доменам
- **Gap-анализ** — дефициты по уровням junior/middle/senior, приоритеты, готовность
- **ML-ранжирование** — XGBoost LTR + SHAP (всегда включён), предсказание важности навыков (0-100%), кросс-доменные объяснения
- **Кластеризация** — KMeans/HDBSCAN + авто k по silhouette, человекочитаемые имена
- **Тренды** — динамика спроса по историческим снимкам, временные ряды топ-10
- **Мониторинг** — Prometheus-метрики пайплайна, API, LTR, LLM; Grafana-дашборды (Server, Application, LLM & AI Monitoring); cAdvisor (Docker-контейнеры); административная панель во фронтенде
- **Автоматизация** — n8n-воркфлоу: nightly pipeline, student onboarding, trend alerts, weekly reports
- **Визуализация** — радары, тепловые карты, покрытие, профессии (300 DPI)

## Структура проекта

```plaintext
📁 compare_competencies/
│
├── 📁 data/
│   ├── 📁 cache/                       # Кэш извлечённых навыков, эмбеддингов, кластеров
│   │   ├── 📄 parsed_skills.joblib     # Извлечённые навыки (парсинг)
│   │   ├── 📄 gap_progress.json        # Прогресс gap-анализа (SSE)
│   │   ├── 📄 pipeline_progress.json   # Прогресс пайплайна (SSE)
│   │   ├── 📄 pipeline_tasks.json      # Фоновые задачи пайплайна
│   │   ├── 📁 embeddings/              # Эмбеддинги навыков + рынка
│   │   ├── 📁 clusters/                # KMeans-кластеры (.joblib + .manifest.json)
│   │   ├── 📁 students/                # Эмбеддинги профилей студентов
│   │   └── 📄 .hh_token_cache.json     # Кэш токена hh.ru API
│   ├── 📁 history/                     # Снимки частот навыков по датам
│   │   ├── 📄 freq_latest.json         # Текущий срез
│   │   └── 📄 freq_2026-*.json         # Исторические (апрель–май 2026, ~40 шт.)
│   ├── 📁 last_uploaded/               # Последняя загруженная матрица
│   │   └── 📄 competency_matrix.csv
│   ├── 📁 models/                      # ML-модели
│   │   ├── 📄 ltr_ranker_xgb_regressor.joblib  # XGBoost LTR-ранкер
│   │   └── 📄 ltr_feature_importance.png        # Важность признаков
│   ├── 📁 processed/                   # Обработанные данные
│   │   ├── 📄 competency_frequency.json          # Частоты навыков на рынке
│   │   ├── 📄 competency_frequency_mapped.json   # Сопоставление с учебными
│   │   ├── 📄 competency_mapping.json            # Коды компетенций -> навыки
│   │   ├── 📄 skill_weights.json                 # Очищенные веса
│   │   ├── 📄 hh_vacancies_detailed.json         # Детальные вакансии
│   │   └── 📄 vacancies_IT_Sector_Multiple.xlsx  # Excel-экспорт вакансий
│   ├── 📁 raw/                         # Сырые данные с hh.ru
│   │   ├── 📄 hh_vacancies_basic.json  # Результат поиска (HH API)
│   │   └── 📄 competency_matrix.csv    # Исходная матрица компетенций
│   ├── 📁 reference/                   # Справочники
│   │   ├── 📄 domain_map.json          # 15 доменов -> список навыков
│   │   ├── 📄 filler_words.json        # Слова-паразиты (19 шт.)
│   │   ├── 📄 generic_words.json       # Общие слова
│   │   ├── 📄 hard_skills.json         # 96 жёстких навыков (EN)
│   │   ├── 📄 it_skills.json           # 933 IT-скилла (430 RU + 503 EN)
│   │   ├── 📄 profession_taxonomy.json     # Профессии -> домены
│   │   ├── 📄 skill_blacklist.json     # Чёрный список
│   │   ├── 📄 skill_taxonomy.json      # 19 категорий навыков
│   │   ├── 📄 stop_lemmas.json         # Стоп-леммы для BM25
│   │   ├── 📄 timeframe_groups.json    # Группы для времени изучения
│   │   └── 📄 trend_hot_skills.json    # Горячие навыки
│   ├── 📁 result/                      # Графики, отчёты, рекомендации
│   │   ├── 📁 base/                    # Графики для профиля base
│   │   ├── 📁 dc/                      # Графики для профиля dc
│   │   ├── 📁 top_dc/                  # Графики для профиля top_dc
│   │   ├── 📁 trends/                  # Тренды: графики + JSON
│   │   └── 📁 reports/                 # Отчёты и экспорт
│   │       ├── 📄 spam_vacancies_report.json
│   │       ├── 📄 coverage_comparison.png
│   │       ├── 📄 profession_coverage.png
│   │       ├── 📄 domain_skill_gaps.png
│   │       ├── 📄 skill_correlation_heatmap.png
│   │       ├── 📄 skills_heatmap.png
│   │       └── 📄 vacancies_export.xlsx
│   └── 📁 students/                    # Профили студентов
│       ├── 📄 base_competency.json
│       ├── 📄 dc_competency.json
│       ├── 📄 top_dc_competency.json
│       ├── 📄 description_of_competency.txt
│       └── 📄 competency_matrix.csv
│
├── 📁 docs/
│   ├── 📄 ARCHITECTURE.md              # Архитектура системы
│   └── 📄 user_manual.md               # Полное руководство
│
├── 📁 frontend/                        # React SPA (Vite + shadcn/ui)
│   ├── 📁 src/
│   │   ├── 📁 app/
│   │   │   ├── 📁 components/
│   │   │   │   ├── 📁 figma/
│   │   │   │   │   └── 📄 ImageWithFallback.tsx
│   │   │   │   ├── 📁 ui/              # shadcn/ui (60+)
│   │   │   │   ├── 📄 AnalysisTab.tsx
│   │   │   │   ├── 📄 DataViewer.tsx
│   │   │   │   ├── 📄 Footer.tsx
│   │   │   │   ├── 📄 GapAnalysisVisualizer.tsx
│   │   │   │   ├── 📄 LoadingSpinner.tsx
│   │   │   │   ├── 📄 MetricsExplanation.tsx
│   │   │   │   ├── 📄 PipelineProgress.tsx
│   │   │   │   ├── 📄 RecommendationsReport.tsx
│   │   │   │   ├── 📄 RegionCombobox.tsx
│   │   │   │   ├── 📄 StatsCards.tsx
│   │   │   │   ├── 📄 VacanciesList.tsx
│   │   │   │   └── 📄 VacancyCard.tsx
│   │   │   └── 📄 App.tsx
│   │   ├── 📁 imports/
│   │   │   ├── 📄 pipeline_endpoints.py
│   │   │   ├── 📄 pipeline_runner.py
│   │   │   ├── 📄 README.md
│   │   │   └── 📄 user_manual.md
│   │   ├── 📁 lib/
│   │   │   └── 📄 logger.ts
│   │   ├── 📁 styles/
│   │   │   ├── 📄 fonts.css
│   │   │   ├── 📄 globals.css
│   │   │   ├── 📄 index.css
│   │   │   ├── 📄 tailwind.css
│   │   │   └── 📄 theme.css
│   │   └── 📄 main.tsx
│   ├── 📄 package.json
│   ├── 📄 vite.config.ts
│   ├── 📄 postcss.config.mjs
│   └── 📄 pnpm-workspace.yaml
│
├── 📁 src/
│   │   # Корень
│   ├── 📄 config.py                    # Pydantic Settings (пути, API, модели)
│   ├── 📄 logging_config.py            # structlog
│   ├── 📄 artifacts.py                 # Манифест артефактов
│   ├── 📄 cache_manager.py             # Менеджер кэша (JSON/joblib)
│   ├── 📄 decorators.py                # Декораторы (кэш, retry, timeout)
│   ├── 📄 errors.py                    # Кастомные исключения
│   ├── 📄 result.py                    # Result[T, E] pattern
│   └── 📄 utils.py                     # Утилиты (atomic_write, safe_read)
│
│   # Пайплайн
│   ├── 📁 pipeline/
│   │   ├── 📄 orchestrator.py          # PipelineOrchestrator
│   │   ├── 📄 stage.py                 # PipelineStage (base)
│   │   ├── 📄 stages.py                # 8 конкретных этапов
│   │   ├── 📄 progress.py              # SSE-прогресс
│   │   ├── 📄 data_source.py           # Загрузка вакансий
│   │   ├── 📄 skill_extractor.py       # Извлечение навыков
│   │   ├── 📄 weight_cleaner.py        # Фильтрация весов
│   │   ├── 📄 level_builder.py         # Уровни junior/middle/senior
│   │   ├── 📄 gap_runner.py            # Gap-анализ
│   │   ├── 📄 metric_computer.py       # Оценка профилей
│   │   ├── 📄 recommendation_runner.py # Рекомендации
│   │   └── 📄 helpers.py               # Общие функции
│
│   # Парсинг
│   ├── 📁 parsing/
│   │   ├── 📁 api/
│   │   │   ├── 📄 hh_api.py            # Синхронный клиент hh.ru
│   │   │   ├── 📄 hh_api_async.py      # Асинхронный клиент
│   │   │   └── 📄 embedding_loader.py  # SentenceTransformer
│   │   ├── 📁 skills/
│   │   │   ├── 📄 skill_parser.py      # Извлечение навыков из текста
│   │   │   ├── 📄 skill_normalizer.py  # Синонимы, fuzzy
│   │   │   ├── 📄 skill_validator.py   # Белый/чёрный списки
│   │   │   ├── 📄 vacancy_parser.py    # Фасад парсера
│   │   │   ├── 📄 bm25_ranker.py       # BM25Okapi
│   │   │   ├── 📄 hybrid_weight_calculator.py  # BM25 + эмбеддинги
│   │   │   └── 📄 skill_embedding_cache.py     # Кэш эмбеддингов
│   │   └── 📄 utils.py
│
│   # Анализ
│   ├── 📁 analyzers/
│   │   ├── 📁 comparison/
│   │   │   ├── 📄 comparator.py        # CompetencyComparator
│   │   │   ├── 📄 embedding_comparator.py  # Cosine similarity
│   │   │   ├── 📄 domain_analyzer.py   # 15 доменов
│   │   │   └── 📄 engines.py           # Jaccard, Ensemble
│   │   ├── 📁 gap/
│   │   │   ├── 📄 gap_analyzer.py      # Разрыв навыков
│   │   │   └── 📄 profile_evaluator.py # ProfileEvaluator
│   │   ├── 📁 skills/
│   │   │   ├── 📄 skill_taxonomy.py    # 19 категорий
│   │   │   ├── 📄 skill_filter.py      # Фильтрация мусора
│   │   │   ├── 📄 skill_level_analyzer.py  # Уровни
│   │   │   ├── 📄 skill_correlation.py # Jaccard-матрица
│   │   │   ├── 📄 profession_taxonomy.py  # Профессии
│   │   │   └── 📄 trends.py            # Тренды
│   │   └── 📁 clustering/
│   │       └── 📄 vacancy_clustering.py    # KMeans/HDBSCAN
│
│   # API
│   ├── 📁 api_pkg/
│   │   ├── 📄 deps.py                  # Depends
│   │   ├── 📄 startup.py               # Startup hooks
│   │   ├── 📄 n8n.py                   # n8n integration
│   │   └── 📁 routers/
│   │       ├── 📄 health.py            # GET /api/health
│   │       ├── 📄 vacancies.py         # GET /api/vacancies
│   │       ├── 📄 profiles.py          # GET /api/profiles
│   │       ├── 📄 clusters.py          # GET /api/clusters
│   │       ├── 📄 taxonomy.py          # GET /api/taxonomy
│   │       ├── 📄 trends.py            # GET /api/trends
│   │       ├── 📄 market.py            # GET /api/market
│   │       ├── 📄 pipeline.py          # POST /api/pipeline
│   │       ├── 📄 results.py           # GET /api/results
│   │       └── 📄 admin.py             # POST /api/admin
│
│   # ML
│   ├── 📁 predictors/
│   │   ├── 📄 recommendation_engine.py # Движок рекомендаций
│   │   ├── 📄 ltr_recommendation_engine.py  # XGBoost + SHAP
│   │   ├── 📄 skill_forecast.py        # Прогноз трендов
│   │   ├── 📄 base.py                  # Базовый предиктор
│   │   ├── 📄 factory.py               # Фабрика
│   │   └── 📄 models.py                # Pydantic-модели
│
│   # ML-эксперименты
│   ├── 📁 ml/
│   │   ├── 📄 clusters.py              # Vacancy clustering
│   │   ├── 📄 tracker.py               # Трекинг ML-экспериментов
│   │   └── 📄 registry.py              # Реестр моделей
│   │
│   # Модели
│   ├── 📁 models/
│   │   ├── 📄 vacancy.py               # Vacancy, KeySkill, Salary
│   │   ├── 📄 student.py               # StudentProfile
│   │   ├── 📄 competency.py            # Competency, CompetencyMatrix
│   │   ├── 📄 comparison.py            # ComparisonReport
│   │   ├── 📄 data_contracts.py        # PipelineContext
│   │   ├── 📄 enums.py                 # Уровни, приоритеты
│   │   ├── 📄 hh_responses.py          # Ответы hh.ru
│   │   ├── 📄 market_metrics.py        # SkillMetrics
│   │   └── 📄 api_responses.py         # API-ответы
│
│   # Мониторинг
│   ├── 📁 monitoring/
│   │   └── 📄 metrics.py                # Prometheus: Histogram, Counter, Gauge
│   │                                      # Pipeline, API, LTR-метрики
│   │
│   # Доменные порты
│   ├── 📁 domain/
│   │   └── 📄 ports.py                   # Абстракции доменной модели
│   │
│   # Инфраструктура
│   ├── 📁 infrastructure/
│   │   ├── 📄 hh_provider.py             # Провайдер hh.ru
│   │   └── 📄 file_provider.py           # Файловый провайдер
│   │
│   # Загрузчики РПД
│   ├── 📁 loaders/
│   │   ├── 📄 rpd_loader.py              # Загрузка РПД
│   │   └── 📄 rpd_skill_cleaner.py       # Очистка навыков РПД
│   │
│   # CLI-утилиты
│   ├── 📁 cli/
│   │   ├── 📄 __main__.py                # python -m src.cli <command>
│   │   ├── 📄 seed_db.py                 # Наполнение БД
│   │   ├── 📄 backup_db.py               # Бэкап/восстановление БД
│   │   ├── 📄 create_user.py             # Создание пользователя
│   │   ├── 📄 embeddings.py              # Управление эмбеддингами
│   │   ├── 📄 import_students.py         # Импорт студентов из CSV
│   │   ├── 📄 export_json.py             # Экспорт в JSON
│   │   ├── 📄 export_results.py          # Экспорт результатов в БД
│   │   ├── 📄 rebuild.py                 # Пересборка данных
│   │   ├── 📄 extend_skills.py           # Расширение it_skills
│   │   ├── 📄 teacher_analysis.py        # Teacher analysis
│   │   ├── 📄 fix_rpd_data.py            # Исправление данных РПД
│   │   ├── 📄 dedup_disciplines.py       # Дедупликация дисциплин
│   │   ├── 📄 compute_competency_trends.py
│   │   ├── 📄 compute_competency_vectors.py # Векторные эмбеддинги
│   │   ├── 📄 export_vacancies.py        # Экспорт вакансий
│   │   ├── 📄 map_ksa_to_skills.py       # KSA → навыки
│   │   ├── 📄 populate_parsed_skills.py  # Парсинг навыков в БД
│   │   └── 📄 snapshot_professions.py    # Снимки профессий
│   │
│   # Оценка качества
│   ├── 📁 evaluation/
│   │   ├── 📄 metrics.py                 # Метрики оценки
│   │   ├── 📄 base.py                    # Базовый класс
│   │   └── 📄 report.py                  # Отчёты
│   │
│   # Верификация
│   ├── 📁 ground_truth/
│   │   └── 📄 hh_proxy.py                # HHGroundTruth
│   │
│   # Оценка качества вакансий
│   ├── 📁 scoring/
│   │   └── 📄 vacancy_quality_scorer.py  # Спам-фильтр (9 критериев)
│   ├── 📁 loaders_student/
│   │   └── 📄 student_loader.py          # Загрузка профилей
│   ├── 📁 n8n/
│   │   ├── 📄 auth.py
│   │   ├── 📄 webhooks.py
│   │   └── 📁 workflows/
│   │       ├── 📄 nightly_pipeline.json
│   │       ├── 📄 student_onboarding.json
│   │       ├── 📄 trend_alert.json
│   │       └── 📄 weekly_report.json
│   └── 📁 visualization/
│       ├── 📄 _config.py
│       ├── 📄 _utils.py
│       ├── 📄 coverage.py              # Графики покрытия
│       ├── 📄 radar.py                 # Радарные диаграммы
│       ├── 📄 importance.py            # Важность навыков
│       ├── 📄 correlation.py           # Тепловая карта
│       ├── 📄 clusters.py              # Кластеры
│       └── 📄 orchestration.py         # Сохранение графиков
│
├── 📁 tests/                           # pytest (~74% coverage)
│   ├── 📄 conftest.py
│   ├── 📁 analyzers/
│   ├── 📁 api/
│   ├── 📁 integration/
│   ├── 📁 loaders/
│   ├── 📁 models/
│   ├── 📁 parsing/
│   ├── 📁 pipeline/
│   ├── 📁 predictors/
│   ├── 📁 scoring/
│   ├── 📁 scripts/
│   ├── 📁 visualization/
│   ├── 📁 snapshots/
│   ├── 📄 test_artifacts.py
│   ├── 📄 test_logging_config.py
│   ├── 📄 test_result.py
│   └── 📄 test_utils.py
│
├── 📄 main.py                          # Точка входа (CLI)
├── 📄 MakeFile                         # make test/lint/train/rebuild
├── 📄 pyproject.toml                   # ruff, mypy, pytest, bandit
├── 📄 requirements.txt                 # Зависимости
├── 📄 requirements-dev.txt             # dev-зависимости
├── 📄 .env.example                     # Переменные окружения
├── 📄 .pre-commit-config.yaml          # pre-commit хуки
├── 📄 .dockerignore                    # Игнор для Docker
├── 📄 seed_users.py                    # Наполнение пользователей из users.json
├── 📄 users.json                       # Пользователи по умолчанию (admin/teacher/student)
├── 📄 backup-volumes.sh                # Бэкап Docker volumes
├── 📄 update-deployment.sh             # Полный деплой + бэкап + rollback
├── 📄 start-ollama.sh                  # Запуск Ollama
└── 📄 README.md
```

## Быстрый старт

```bash
# 1. Установка
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2. Полный цикл: сбор → обучение → gap-анализ (одна команда)
python main.py --it-sector --excel

# 3. Или пошагово:
python main.py --train-model                           # LTR-модель
python main.py --skip-collection                       # gap-анализ без сбора
python main.py --teacher-analysis                      # teacher analysis
python -m src.cli rebuild                              # пересборка данных

# 4. API
uvicorn src.api_pkg:app --host 0.0.0.0 --port 8000 --reload

# 5. Фронтенд (отдельный терминал)
cd frontend && npm install && npx vite
```

## Запуск через Docker

### Предварительные требования

- Установленный [Docker](https://docs.docker.com/get-docker/) и Docker Compose
- Файл `.env` — скопировать из `.env.example` и заполнить:
  ```bash
  cp .env.example .env
  ```
  Минимально для работы требуется указать `HH_CLIENT_ID` и `HH_CLIENT_SECRET` (создать на https://dev.hh.ru/admin).

### Полный запуск (все сервисы)

```bash
# Сборка образов
docker compose build

# Запуск
docker compose up -d

# Просмотр логов
docker compose logs -f
```

### Выборочный запуск (без тяжёлых сервисов)

Если не нужны n8n и open-webui:

```bash
docker compose up -d backend frontend prometheus grafana postgres-exporter node-exporter cadvisor
```

### Состав сервисов

| Сервис | Назначение | Порт |
|--------|-----------|------|
| `backend` | FastAPI (метрики, pipeline, API) | `:8000` |
| `frontend` | React SPA (вкладка "Мониторинг") | `:8080` |
| `competency-postgres` | Основная БД (pgvector) | `:5432` |
| `open-webui` | Веб-чат с внешней Ollama (`qwen3.6:latest`) | `:3000` |
| `n8n` | Автоматизация воркфлоу | `:5678` |
| `n8n-postgres` | БД n8n | — |
| `prometheus` | Сбор метрик (30 дней хранения) | `:9090` |
| `grafana` | Визуализация метрик (admin / 2JQeA2nD7Ndsj1kr) | `:3001` |
| `postgres-exporter` | Метрики PostgreSQL | `:9187` |
| `node-exporter` | Метрики хоста | `:9100` |
| `cadvisor` | Метрики Docker-контейнеров | `:8081` |

### Доступ к сервисам

| Ссылка | Описание |
|--------|----------|
| http://localhost:8080 | Frontend (вкладка "Мониторинг" для admin) |
| http://localhost:8000/docs | Swagger-документация API |
| http://localhost:8000/metrics | Prometheus-метрики (raw) |
| http://localhost:8000/api/admin/monitoring | JSON-дашборд мониторинга |
| http://localhost:3001 | Grafana (admin / 2JQeA2nD7Ndsj1kr) — дашборды: Server, Application, LLM & AI Monitoring |
| http://localhost:9090 | Prometheus Web UI |
| http://localhost:5678 | n8n |
| http://localhost:3000 | Open WebUI (чат с Qwen 3.6) |

### Остановка и управление

```bash
# Остановить все сервисы
docker compose down

# Остановить с удалением томов БД
docker compose down -v

# Перезапустить конкретный сервис
docker compose restart backend

# Логи конкретного сервиса
docker compose logs -f backend

# Пересобрать без кэша
docker compose build --no-cache
```

### Возможные проблемы

1. **Open WebUI не видит модель** — проверьте, что внешняя Ollama доступна: `docker exec openwebui curl -s http://ollama8.r61.net:11434/api/tags`
2. **Frontend пустая страница** — убедитесь, что `frontend/dist/` существует. Если нет — соберите вручную:
   ```bash
   cd frontend && npm install && npm run build
   ```
3. **Backend не стартует** — проверьте `.env` и что PostgreSQL доступен (он стартует дольше всех).
4. **Метрики пустые** — выполните pipeline через API (POST `/api/pipeline/run`) или CLI (`python main.py --it-sector --excel`).
5. **cAdvisor не видит контейнеры** — проверьте `docker logs cadvisor`. Требуются монтирования `/var/lib/docker/` и `/sys/`.

## Зависимости

**Python:** fastapi, uvicorn, requests, aiohttp, pandas, numpy, scikit-learn, xgboost, shap, sentence-transformers, matplotlib, seaborn, pydantic, structlog, pymorphy3, rapidfuzz

**Frontend:** React 18, TypeScript, Vite 6.3, shadcn/ui (60+), recharts, motion, react-router, lucide-react

## Тестирование

```bash
pytest --cov=src --cov-report=term --ignore=tests/test_api.py
```

- 1900+ тестов, 74% покрытие (0 failed), 79 skipped
- E2E-тест пайплайна, тесты мониторинга, SHAP, векторизации
- Ключевые: vacany_quality_scorer (100%), engines (94%), data_source (93%)

## Примеры

```bash
python main.py --query "Data Scientist" --area-id 2 --max-pages 5 --excel
python main.py --queries-file queries.txt --regions 1,2 --excel
python main.py --interactive
```

## Мониторинг и инфраструктура

Все сервисы мониторинга разворачиваются через Docker — см. раздел [Запуск через Docker](#запуск-через-docker).

- **Prometheus** — сбор метрик с backend, postgres-exporter, node-exporter, cAdvisor
- **Grafana** — визуализация метрик, дашборды: Server Monitoring, Application Monitoring, LLM & AI Monitoring
- **LLM метрики** — `llm_requests_total` (status: ok/error, model), `llm_request_duration_seconds` (гистограмма)
- **cAdvisor** — метрики Docker-контейнеров (CPU, RAM, Network, Disk для каждого контейнера)
- **n8n** — автоматизация: nightly pipeline, student onboarding, trend alerts, weekly report
- **Ollama** — внешняя LLM (сервер университета, `http://ollama8.r61.net:11434`, модель `qwen3.6:latest`, 36B)
- **PostgreSQL / pgvector** — основная БД с поддержкой векторного поиска

## Скрипты (корень проекта)

| Файл | Назначение |
|------|------------|
| `seed_users.py` | Наполнение пользователей из `users.json` в PostgreSQL через `asyncpg`. Запуск: `python seed_users.py` |
| `users.json` | Пользователи по умолчанию: admin, teacher, student |
| `backup-volumes.sh` | Бэкап Docker volumes + дамп PostgreSQL в `./backups/` |
| `update-deployment.sh` | Полный деплой: бэкап → git pull → compose down → build → up → health check → rollback при ошибке |
| `start-ollama.sh` | Запуск Ollama (для локальной разработки) |
| `.dockerignore` | Игнор для Docker сборки (исключает node_modules, .git, \_\_pycache\_\_) |

## Документация

- `docs/ARCHITECTURE.md` — архитектура системы
- `docs/user_manual.md` — полное руководство пользователя
- `src/n8n/n8n_guide.md` — интеграция с n8n (workflows, credentials, деплой)
