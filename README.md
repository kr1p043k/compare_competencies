# Competency Gap Analyzer

Анализ соответствия учебных компетенций студентов требованиям IT-рынка (hh.ru).

Собирает вакансии, нормализует навыки, выполняет gap-анализ и формирует персонализированные рекомендации через ML (XGBoost + SHAP).

## Возможности

- **Сбор вакансий** — hh.ru API, синхронный и асинхронный клиенты, 2000 вакансий на запрос
- **Нормализация навыков** — синонимы, fuzzy-матчинг, BM25 + SentenceTransformer + PCA
- **Таксономия** — 19 категорий навыков + профессии с привязкой к доменам
- **Gap-анализ** — дефициты по уровням junior/middle/senior, приоритеты, готовность
- **ML-ранжирование** — XGBoost LTR + SHAP, предсказание важности навыков (0-100%)
- **Кластеризация** — KMeans/HDBSCAN + авто k по silhouette, человекочитаемые имена
- **Тренды** — динамика спроса по историческим снимкам, временные ряды топ-10
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
├── 📁 scripts/
│   ├── 📄 check_clusters.py            # Проверка кластеров
│   ├── 📄 extend_it_skills.py          # Расширение белого списка навыков
│   ├── 📄 full_rebuild.py              # Пересборка проекта
│   └── 📄 train_clusters.py            # Обучение кластеров
│
├── 📁 src/
│   │   # Корень
│   ├── 📄 config.py                    # Pydantic Settings (пути, API, модели)
│   ├── 📄 logging_config.py            # structlog
│   ├── 📄 api.py                       # FastAPI (legacy)
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
│   # Остальное
│   ├── 📁 scoring/
│   │   └── 📄 vacancy_quality_scorer.py    # Спам-фильтр (9 критериев)
│   ├── 📁 loaders_student/
│   │   └── 📄 student_loader.py        # Загрузка профилей
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
├── 📁 tests/                           # pytest (86% coverage)
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
python scripts/train_clusters.py --level all           # кластеризация
python main.py --train-model                           # LTR-модель
python main.py --skip-collection                       # gap-анализ без сбора

# 4. API
uvicorn src.api_pkg:app --host 0.0.0.0 --port 8000 --reload

# 5. Фронтенд (отдельный терминал)
cd frontend && npm install && npx vite
```

## Зависимости

**Python:** fastapi, uvicorn, requests, aiohttp, pandas, numpy, scikit-learn, xgboost, shap, sentence-transformers, matplotlib, seaborn, pydantic, structlog, pymorphy3, rapidfuzz

**Frontend:** React 18, TypeScript, Vite 6.3, shadcn/ui (60+), recharts, motion, react-router, lucide-react

## Тестирование

```bash
pytest --cov=src --cov-report=term --ignore=tests/test_api.py
```

- 35+ тестовых файлов, 86% покрытие
- Ключевые: vacany_quality_scorer (100%), engines (98%), data_source (97%)

## Примеры

```bash
python main.py --query "Data Scientist" --area-id 2 --max-pages 5 --excel
python main.py --queries-file queries.txt --regions 1,2 --excel
python main.py --interactive
```

## Документация

- `docs/ARCHITECTURE.md` — архитектура системы
- `docs/user_manual.md` — полное руководство пользователя
- `src/n8n/n8n_guide.md` — интеграция с n8n (workflows, credentials, деплой)
