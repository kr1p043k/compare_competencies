# Сравнение компетенций учебной программы с требованиями рынка труда (hh.ru)

Проект предназначен для анализа соответствия учебных компетенций студентов реальным требованиям IT-рынка, извлекаемым из вакансий hh.ru.
Система собирает вакансии, нормализует навыки, выполняет gap-анализ и формирует персонализированные рекомендации с использованием машинного обучения.

## 🎯 Основные возможности

- **Сбор вакансий с hh.ru** — поиск по профессиям, регионам, пакетный и интерактивный режим
- **Нормализация навыков** — очистка, синонимия, BM25 + SentenceTransformer + PCA
- **Таксономия навыков и профессий** — 19 категорий, привязка к доменам и КРМ-компетенциям
- **Gap-анализ (junior/middle/senior)** — дефициты, приоритеты, готовность к уровню
- **Покрытие целевой профессии** — средневзвешенное по доменам
- **ML-ранжирование (LTR + XGBoost + SHAP)** — предсказание важности навыков
- **Кластеризация вакансий** — KMeans/HDBSCAN + автоопределение k
- **Визуализация** — радары, тепловые карты, сравнение профилей (300 DPI)
- **Анализ трендов** — динамика спроса, временные ряды
- **REST API (api_pkg)** — 10 FastAPI-роутеров, n8n, startup/hooks
- **React-фронтенд** — Vite, shadcn/ui, TypeScript, Vite proxy

## 📁 Структура проекта

```plaintext
└── 📁 compare_competencies/
    │
    ├── 📁 data/                            # Данные: кэш, справочники, результаты
    │   ├── 📁 cache/                       # Единый кэш быстрых данных
    │   ├── 📁 history/                     # Исторические снимки частот навыков
    │   ├── 📁 last_uploaded/               # Последняя загруженная матрица компетенций
    │   ├── 📁 models/                      # Обученные ML-модели (joblib)
    │   ├── 📁 processed/                   # Обработанные данные после парсинга
    │   ├── 📁 raw/                         # Сырые входные данные
    │   ├── 📁 reference/                   # Эталонные справочники
    │   ├── 📁 result/                      # Результаты: отчёты, графики (300 DPI)
    │   └── 📁 students/                    # Профили студентов
    │
    ├── 📁 docs/                            # Документация
    │   ├── 📄 ARCHITECTURE.md              # Архитектура и возможности системы
    │   ├── 📄 PROJECT_OVERVIEW.md           # Обзор проекта
    │   └── 📄 user_manual.md               # Руководство пользователя
    │
    ├── 📁 frontend/                        # React + Vite + TypeScript
    │   ├── 📁 src/
    │   │   ├── 📁 app/
    │   │   │   ├── 📁 components/
    │   │   │   │   ├── 📁 figma/
    │   │   │   │   │   └── 📄 ImageWithFallback.tsx
    │   │   │   │   ├── 📁 ui/              # shadcn/ui (60+ компонентов)
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
    │   │   ├── 📁 imports/                 # Импортированные скрипты и данные
    │   │   │   ├── 📁 pasted_text/
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
    │   ├── 📄 pipeline.bat
    │   └── 📄 pnpm-workspace.yaml
    │
    ├── 📁 htmlcov/                         # Отчёты о покрытии тестами
    │
    ├── 📁 logs/                            # Логи приложения
    │
    ├── 📁 notebooks/                       # Jupyter ноутбуки
    │
    ├── 📁 scripts/
    │   ├── 📄 check_clusters.py            # Проверка обученных кластеров
    │   ├── 📄 extend_it_skills.py          # Расширение белого списка навыков
    │   ├── 📄 full_rebuild.py              # Очистка кэшей и пересборка
    │   └── 📄 train_clusters.py            # Обучение кластеров по уровням
    │
    ├── 📁 src/
    │   ├── 📁 analyzers/
    │   │   ├── 📁 clustering/
    │   │   │   └── 📄 vacancy_clustering.py
    │   │   ├── 📁 comparison/
    │   │   │   ├── 📄 comparator.py
    │   │   │   ├── 📄 domain_analyzer.py
    │   │   │   ├── 📄 embedding_comparator.py
    │   │   │   └── 📄 engines.py
    │   │   ├── 📁 gap/
    │   │   │   ├── 📄 gap_analyzer.py
    │   │   │   └── 📄 profile_evaluator.py
    │   │   └── 📁 skills/
    │   │       ├── 📄 profession_taxonomy.py
    │   │       ├── 📄 skill_correlation.py
    │   │       ├── 📄 skill_filter.py
    │   │       ├── 📄 skill_level_analyzer.py
    │   │       ├── 📄 skill_taxonomy.py
    │   │       └── 📄 trends.py
    │   │
    │   ├── 📁 api_pkg/                     # FastAPI-пакет (10 роутеров)
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 deps.py
    │   │   ├── 📄 n8n.py
    │   │   ├── 📄 startup.py
    │   │   └── 📁 routers/
    │   │       ├── 📄 __init__.py
    │   │       ├── 📄 admin.py
    │   │       ├── 📄 clusters.py
    │   │       ├── 📄 health.py
    │   │       ├── 📄 market.py
    │   │       ├── 📄 pipeline.py
    │   │       ├── 📄 profiles.py
    │   │       ├── 📄 results.py
    │   │       ├── 📄 taxonomy.py
    │   │       ├── 📄 trends.py
    │   │       └── 📄 vacancies.py
    │   │
    │   ├── 📁 loaders_student/
    │   │   ├── 📄 __init__.py
    │   │   └── 📄 student_loader.py
    │   │
    │   ├── 📁 models/                      # Pydantic-модели
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 api_responses.py
    │   │   ├── 📄 comparison.py
    │   │   ├── 📄 competency.py
    │   │   ├── 📄 data_contracts.py
    │   │   ├── 📄 enums.py
    │   │   ├── 📄 hh_responses.py
    │   │   ├── 📄 market_metrics.py
    │   │   ├── 📄 student.py
    │   │   └── 📄 vacancy.py
    │   │
    │   ├── 📁 parsing/
    │   │   ├── 📁 api/
    │   │   │   ├── 📄 __init__.py
    │   │   │   ├── 📄 hh_api.py
    │   │   │   ├── 📄 hh_api_async.py
    │   │   │   └── 📄 embedding_loader.py
    │   │   ├── 📁 skills/
    │   │   │   ├── 📄 __init__.py
    │   │   │   ├── 📄 bm25_ranker.py
    │   │   │   ├── 📄 hybrid_weight_calculator.py
    │   │   │   ├── 📄 skill_embedding_cache.py
    │   │   │   ├── 📄 skill_normalizer.py
    │   │   │   ├── 📄 skill_parser.py
    │   │   │   ├── 📄 skill_validator.py
    │   │   │   └── 📄 vacancy_parser.py
    │   │   └── 📄 utils.py
    │   │
    │   ├── 📁 pipeline/
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 data_source.py
    │   │   ├── 📄 gap_runner.py
    │   │   ├── 📄 helpers.py
    │   │   ├── 📄 level_builder.py
    │   │   ├── 📄 metric_computer.py
    │   │   ├── 📄 recommendation_runner.py
    │   │   ├── 📄 skill_extractor.py
    │   │   └── 📄 weight_cleaner.py
    │   │
    │   ├── 📁 predictors/
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 ltr_recommendation_engine.py
    │   │   ├── 📄 recommendation_engine.py
    │   │   └── 📄 skill_forecast.py
    │   │
    │   ├── 📁 scoring/
    │   │   ├── 📄 __init__.py
    │   │   └── 📄 vacancy_quality_scorer.py
    │   │
    │   ├── 📁 visualization/
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 _config.py
    │   │   ├── 📄 _utils.py
    │   │   ├── 📄 clusters.py
    │   │   ├── 📄 correlation.py
    │   │   ├── 📄 coverage.py
    │   │   ├── 📄 importance.py
    │   │   ├── 📄 orchestration.py
    │   │   └── 📄 radar.py
    │   │
    │   ├── 📄 __init__.py
    │   ├── 📄 api.py
    │   ├── 📄 artifacts.py
    │   ├── 📄 config.py
    │   ├── 📄 logging_config.py
    │   ├── 📄 result.py
    │   └── 📄 utils.py
    │
    ├── 📁 tests/                           # pytest (35 файлов, ~86% coverage)
    │   ├── 📁 analyzers/
    │   │   ├── 📁 comparison/
    │   │   ├── 📄 test_analyzers.py … test_vacancy_clustering.py
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
    │   ├── 📄 conftest.py
    │   ├── 📄 test_artifacts.py
    │   ├── 📄 test_logging_config.py
    │   └── 📄 test_utils.py
    │
    ├── 📄 main.py                          # Точка входа: полный пайплайн
    ├── 📄 pyproject.toml                   # Конфигурация проекта и инструментов
    ├── 📄 requirements.txt                 # Prod-зависимости
    ├── 📄 requirements-dev.txt             # dev-зависимости
    ├── 📄 MakeFile
    └── 📄 README.md
```

## 🚀 Быстрый старт

1. **Установите зависимости**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt   # тесты, ноутбуки

2. **Соберите вакансии**
   ```bash
   python main.py --it-sector --excel

3. **Обучите кластеры**
   ```bash
   python scripts/train_clusters.py --level all

4. **Обучите ML‑модель ранжирования**
   ```bash
   python main.py --train-model

5. **Запустите gap‑анализ и получите рекомендации**
   ```bash
   python main.py --skip-collection --run-gap-analysis

6. **Графики и отчёты**
   Пайплайн генерирует графики в `data/result/`: радарные диаграммы, важность навыков, тепловые карты, покрытие доменов.

7. **Запустите backend**
   ```bash
   uvicorn src.api_pkg.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   API будет доступен на `http://localhost:8000`, документация: `http://localhost:8000/docs`.

8. **Запустите фронтенд (опционально)**
   ```bash
   cd frontend
   npm install
   npx vite
   ```
   Приложение на `http://localhost:3000` (Vite proxy → backend :8000).

## 📦 Зависимости

**Python (requirements.txt) — актуальные установленные версии:**
- `fastapi==0.116.1`, `uvicorn==0.35.0` — REST API
- `requests==2.32.4`, `aiohttp==3.12.14` — работа с API hh.ru
- `pandas==2.3.1`, `numpy==2.2.6` — обработка данных
- `scikit-learn==1.8.0`, `xgboost==3.0.3`, `shap==0.51.0` — ML
- `sentence-transformers==5.3.0`, `torch>=2.5.0` — эмбеддинги
- `matplotlib==3.10.3`, `seaborn==0.13.2` — визуализация
- `pydantic==2.13.4` — модели данных
- `structlog==25.5.0` — логирование
- `lightgbm==4.6.0`, `faiss-cpu==1.13.2` (опционально)
- `pydantic-settings==2.14.1` — конфигурация
- `aioresponses` — моки для тестов API

**Frontend (frontend/package.json):**
- `React 18` + `TypeScript` + `Vite 6.3`
- `shadcn/ui` (Radix UI, Tailwind CSS) — 60+ компонентов
- `recharts==2.15.2` — графики
- `lucide-react==0.487.0` — иконки
- `motion==12.23` — анимации
- `react-router==7.13` — маршрутизация
- `cmdk`, `sonner`, `vaul`, `date-fns`, `react-hook-form` — UI-утилиты

## 🧪 Тестирование

```bash
pytest --cov=src --cov-report=term --ignore=tests/test_api.py
```
- **35 тестовых файлов**, 36 модулей src под покрытием
- **Общее покрытие: 86%** (отчёт: `htmlcov/index.html`)
- Ключевые модули: спам-фильтрация (100%), движки сравнения (98%), data_source (97%)
- API-тесты (hh.ru) выключены — требуют токена, тестируются интеграционно

## 📝 Примеры использования

### Поиск по конкретному запросу
```bash
python main.py --query "Data Scientist" --area-id 2 --max-pages 5 --excel

### Загрузка запросов из файла
python main.py --queries-file queries.txt --regions 1,2 --excel

### Интерактивный режим
python main.py --interactive

### Обучение модели на свежих данных
python main.py --it-sector          # сбор вакансий
python scripts/train_clusters.py --level all
python main.py --train-model        # обучение LTR
python main.py --skip-collection --run-gap-analysis  # генерация рекомендаций

## 📖 Подробнее

Детальное руководство пользователя находится в файле user_manual.md.
Вопросы и предложения можно оставлять в Issues репозитория.

---
*Проект создан для образовательных и исследовательских целей.*
