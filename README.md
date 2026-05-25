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
└── 📁 competency_comparison
    │
    ├── 📁 data/                            # Данные: кэш, справочники, результаты
    │   ├── 📁 cache/                       # Единый кэш быстрых данных
    │   ├── 📁 history/                     # Исторические снимки частот навыков
    │   ├── 📁 last_uploaded/               # Последняя загруженная матрица компетенций
    │   ├── 📁 models/                      # Обученные ML-модели
    │   ├── 📁 processed/                   # Обработанные данные после парсинга
    │   ├── 📁 raw/                         # Сырые входные данные
    │   ├── 📁 reference/                   # Эталонные справочники
    │   ├── 📁 result/                      # Результаты: отчёты, графики
    │   └── 📁 students/                    # Профили студентов
    │
    ├── 📁 frontend/                        # React + Vite + TypeScript
    │   └── 📁 src/
    │       ├── 📁 app/
    │       │   ├── 📁 components/
    │       │   │   ├── 📁 figma/
    │       │   │   │   └── 📄 ImageWithFallback.tsx
    │       │   │   ├── 📁 ui/              # shadcn/ui (button, card, dialog, tooltip…)
    │       │   │   ├── 📄 AnalysisTab.tsx   # Вкладка анализа, принимает состояние через пропсы
    │       │   │   ├── 📄 DataViewer.tsx    # Табличный просмотр данных
    │       │   │   ├── 📄 Footer.tsx        # Подвал
    │       │   │   ├── 📄 GapAnalysisVisualizer.tsx  # Визуализация gap-анализа
    │       │   │   ├── 📄 LoadingSpinner.tsx         # Индикатор загрузки
    │       │   │   ├── 📄 MetricsExplanation.tsx     # Объяснение метрик
    │       │   │   ├── 📄 PipelineProgress.tsx       # Прогресс пайплайна
    │       │   │   ├── 📄 RecommendationsReport.tsx  # Рекомендации + gap-объяснения с тултипами
    │       │   │   ├── 📄 RegionCombobox.tsx         # Выбор региона
    │       │   │   ├── 📄 StatsCards.tsx             # Карточки статистики
    │       │   │   ├── 📄 VacanciesList.tsx          # Список вакансий
    │       │   │   └── 📄 VacancyCard.tsx            # Карточка вакансии
    │       │   └── 📄 App.tsx               # Корневой компонент, состояние наверх
    │       ├── 📁 lib/
    │       │   └── 📄 logger.ts             # Клиентское логирование
    │       ├── 📁 styles/                   # CSS: tailwind, globals, theme
    │       │   ├── 📄 fonts.css             # Шрифты
    │       │   ├── 📄 globals.css           # Глобальные стили
    │       │   ├── 📄 index.css             # Входная точка стилей
    │       │   ├── 📄 tailwind.css          # Tailwind directives
    │       │   └── 📄 theme.css             # Тема
    │       └── 📄 main.tsx                  # Точка входа React
    │
    ├── 📁 notebooks/                        # Jupyter ноутбуки
    │
    ├── 📁 scripts/
    │   ├── 📄 check_clusters.py             # Проверка обученных кластеров вакансий
    │   ├── 📄 extend_it_skills.py           # Расширение белого списка навыков из вакансий
    │   ├── 📄 full_rebuild.py               # Очистка кэшей и полная пересборка проекта
    │   └── 📄 train_clusters.py             # Обучение кластеров вакансий по уровням
    │
    ├── 📁 src/
    │   ├── 📁 analyzers/                    # Аналитика и сравнение
    │   │   ├── 📁 clustering/
    │   │   │   └── 📄 vacancy_clustering.py # Кластеризация вакансий KMeans/HDBSCAN
    │   │   ├── 📁 comparison/
    │   │   │   ├── 📄 comparator.py         # Сравнение навыков студента с рынком (TF-IDF / эмбеддинги)
    │   │   │   ├── 📄 domain_analyzer.py    # Анализ покрытия по 15 доменам
    │   │   │   ├── 📄 embedding_comparator.py # Семантическое сравнение через эмбеддинги + FAISS
    │   │   │   └── 📄 engines.py            # Движки сравнения профилей
    │   │   ├── 📁 gap/
    │   │   │   ├── 📄 gap_analyzer.py       # Gap-анализ: разница между навыками студента и рынка
    │   │   │   └── 📄 profile_evaluator.py  # Оценка профиля: метрики покрытия, readiness
    │   │   └── 📁 skills/
    │   │       ├── 📄 profession_taxonomy.py # Таксономия профессий: домены, КРМ-коды
    │   │       ├── 📄 skill_correlation.py  # Анализ совместной встречаемости навыков (Jaccard)
    │   │       ├── 📄 skill_filter.py       # Фильтрация навыков, удаление мусора
    │   │       ├── 📄 skill_level_analyzer.py # Распределение навыков по уровням
    │   │       ├── 📄 skill_taxonomy.py     # Таксономия навыков по 19 категориям (Singleton)
    │   │       └── 📄 trends.py             # Анализ трендов: сравнение исторических снимков
    │   │
    │   ├── 📁 api_pkg/                      # FastAPI-пакет
    │   │   ├── 📄 __init__.py               # Инициализация пакета
    │   │   ├── 📄 deps.py                   # Depends-зависимости (DI)
    │   │   ├── 📄 n8n.py                    # Интеграция с n8n
    │   │   ├── 📄 startup.py                # Startup/hooks приложения
    │   │   └── 📁 routers/
    │   │       ├── 📄 __init__.py            # Сборка роутеров
    │   │       ├── 📄 admin.py              # Админ-эндпоинты
    │   │       ├── 📄 clusters.py           # Эндпоинты кластеров
    │   │       ├── 📄 health.py             # Health check (/api/health)
    │   │       ├── 📄 market.py             # Эндпоинты рыночных данных
    │   │       ├── 📄 pipeline.py           # Управление пайплайном
    │   │       ├── 📄 profiles.py           # Эндпоинты профилей студентов
    │   │       ├── 📄 results.py            # Результаты анализа
    │   │       ├── 📄 taxonomy.py           # Таксономия навыков
    │   │       ├── 📄 trends.py             # Тренды
    │   │       └── 📄 vacancies.py          # Вакансии
    │   │
    │   ├── 📁 loaders_student/              # Загрузка профилей студентов
    │   │   ├── 📄 __init__.py
    │   │   └── 📄 student_loader.py         # Загрузка из JSON/CSV
    │   │
    │   ├── 📁 models/                       # Pydantic-модели
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 api_responses.py          # Pydantic-схемы ответов FastAPI (30+ моделей)
    │   │   ├── 📄 comparison.py             # Модели сравнения профилей
    │   │   ├── 📄 competency.py             # Модель компетенции
    │   │   ├── 📄 data_contracts.py         # Контракты между этапами пайплайна
    │   │   ├── 📄 enums.py                  # Enum'ы (уровни, приоритеты, тренды)
    │   │   ├── 📄 hh_responses.py           # Модели ответов API hh.ru
    │   │   ├── 📄 market_metrics.py         # Метрики рынка
    │   │   ├── 📄 student.py                # Модель студента
    │   │   └── 📄 vacancy.py                # Модель вакансии
    │   │
    │   ├── 📁 parsing/                      # Парсинг вакансий hh.ru
    │   │   ├── 📁 api/
    │   │   │   ├── 📄 __init__.py
    │   │   │   ├── 📄 hh_api.py             # Синхронный клиент API hh.ru
    │   │   │   ├── 📄 hh_api_async.py       # Асинхронный клиент (пакетная загрузка)
    │   │   │   └── 📄 embedding_loader.py   # Загрузка SentenceTransformer (Singleton)
    │   │   ├── 📁 skills/
    │   │   │   ├── 📄 __init__.py
    │   │   │   ├── 📄 bm25_ranker.py        # BM25-ранжер с предфильтрацией
    │   │   │   ├── 📄 hybrid_weight_calculator.py # Гибридные веса BM25 + Embeddings + PCA
    │   │   │   ├── 📄 skill_embedding_cache.py # JSON-кэш эмбеддингов навыков
    │   │   │   ├── 📄 skill_normalizer.py   # Нормализация: синонимы, fuzzy-матчи
    │   │   │   ├── 📄 skill_parser.py       # Извлечение навыков из текста
    │   │   │   ├── 📄 skill_validator.py    # Валидация: чёрный/белый списки
    │   │   │   └── 📄 vacancy_parser.py     # Фасад парсера
    │   │   └── 📄 utils.py                  # Утилиты: whitelist, интерактивный режим
    │   │
    │   ├── 📁 pipeline/                     # Конвейер обработки
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 data_source.py            # Загрузка вакансий из кэша или hh.ru
    │   │   ├── 📄 gap_runner.py             # Оркестрация gap-анализа и рекомендаций
    │   │   ├── 📄 helpers.py                # Общие функции: загрузка, сохранение
    │   │   ├── 📄 level_builder.py          # Подготовка level_vacancies_data
    │   │   ├── 📄 metric_computer.py        # Оценка профилей (ProfileEvaluator)
    │   │   ├── 📄 recommendation_runner.py  # Генерация рекомендаций
    │   │   ├── 📄 skill_extractor.py        # Извлечение навыков из вакансий
    │   │   └── 📄 weight_cleaner.py         # Фильтрация hybrid_weights через справочники
    │   │
    │   ├── 📁 predictors/                   # ML-модели
    │   │   ├── 📄 __init__.py
    │   │   ├── 📄 ltr_recommendation_engine.py  # LTR на XGBoost (обучение, SHAP)
    │   │   ├── 📄 recommendation_engine.py      # Ансамбль: скоры, тренды, домены
    │   │   └── 📄 skill_forecast.py             # Прогнозирование трендов навыков
    │   │
    │   ├── 📁 scoring/                      # Скоринг
    │   │   ├── 📄 __init__.py
    │   │   └── 📄 vacancy_quality_scorer.py # Оценка качества вакансий
    │   │
    │   ├── 📁 visualization/                # Графики и отчёты
    │   │   ├── 📄 __init__.py               # Публичный API визуализации
    │   │   ├── 📄 _config.py                # Настройки стилей и эмодзи
    │   │   ├── 📄 _utils.py                 # Загрузка данных для графиков
    │   │   ├── 📄 clusters.py               # Ближайшие кластеры вакансий
    │   │   ├── 📄 correlation.py            # Тепловая карта Jaccard-матрицы
    │   │   ├── 📄 coverage.py               # Графики покрытия рынка
    │   │   ├── 📄 importance.py             # Важность навыков (ML + веса)
    │   │   ├── 📄 orchestration.py          # Сохранение графиков, запуск ноутбуков
    │   │   └── 📄 radar.py                  # Радарная диаграмма навыков
    │   │
    │   ├── 📄 __init__.py
    │   ├── 📄 api.py                        # FastAPI-сервер для доступа к данным
    │   ├── 📄 artifacts.py                  # Манифест артефактов (версия, хеш)
    │   ├── 📄 config.py                     # Pydantic Settings
    │   ├── 📄 logging_config.py             # structlog + маскирование секретов
    │   ├── 📄 result.py                     # Result[T, E] / Either pattern
    │   └── 📄 utils.py                      # Утилиты: atomic_write_json, safe_read_json
    │
    ├── 📁 tests/                            # pytest + mypy --strict
    │   ├── 📁 analyzers/
    │   ├── 📁 integration/
    │   ├── 📁 loaders/
    │   ├── 📁 models/
    │   ├── 📁 parsing/
    │   ├── 📁 predictors/
    │   ├── 📁 scripts/
    │   ├── 📁 visualization/
    │   └── 📄 conftest.py
    │
    ├── 📄 main.py                           # Точка входа: полный пайплайн
    ├── 📄 requirements.txt                  # Prod-зависимости
    ├── 📄 requirements-dev.txt              # dev-зависимости
    ├── 📄 MakeFile
    ├── 📄 user_manual.md                    # Руководство пользователя
    └── 📄 README.md                         # Документация проекта
```

## 🚀 Быстрый старт

1. **Установите зависимости**
   ```bash
   pip install -r requirements.txt
   # Для разработки (тесты, ноутбуки):
   pip install -r requirements-dev.txt

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

7. **Запустите фронтенд (опционально)**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Приложение на `http://localhost:5173`. Vite proxy отправляет `/api/*` на backend (порт 8000).

## 📦 Зависимости

**Python (requirements.txt):**
- `fastapi`, `uvicorn` – REST API
- `requests`, `aiohttp` – работа с API hh.ru
- `pandas`, `numpy` – обработка данных
- `scikit‑learn`, `xgboost`, `shap` – машинное обучение
- `sentence‑transformers` – эмбеддинги
- `matplotlib`, `seaborn` – визуализация
- `pydantic` – модели данных
- `structlog` – логирование
- `lightgbm`, `faiss‑cpu` (опционально)

**Frontend (frontend/package.json):**
- `React 18`, `TypeScript`, `Vite`
- `shadcn/ui` (Radix UI, Tailwind CSS)
- `recharts` – графики
- `lucide-react` – иконки

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
