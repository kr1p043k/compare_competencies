# Архитектура проекта

## Обзор

Competency Gap Analyzer — система анализа соответствия учебных компетенций студентов требованиям IT-рынка. Собирает вакансии с hh.ru, нормализует навыки, выполняет gap-анализ и формирует персонализированные рекомендации.

## Pipeline

```
Stage 1: Data Source          → data_source.py
  Сбор/загрузка вакансий из hh.ru или кэша
  ↓

Stage 2: Skill Extraction     → skill_extractor.py
  Извлечение навыков → BM25 + эмбеддинги → гибридные веса
  ↓

Stage 3: Weight Cleaning      → weight_cleaner.py
  Фильтрация через справочники, удаление мусора
  ↓

Stage 4: Level Builder        → level_builder.py
  Распределение по уровням (junior/middle/senior)
  ↓

Stage 5: Gap Analysis         → gap_runner.py
  ProfileEvaluator → метрики покрытия → рекомендации
  ↓

(Stage 2, 4, 5 инструментированы Prometheus-метриками:
 длительность стадии, количество ошибок, собранные вакансии)
```

## Similarity Engines (`src/analyzers/comparison/`)

```
CompetencyComparator
  ├── TF-IDF mode (legacy) — TfidfVectorizer + cosine_similarity
  └── Embedding mode
        └── EmbeddingComparator
              ├── cosine_similarity (sklearn) — основной движок
              │     SentenceTransformer → эмбеддинги 768d
              │     Атомарный кэш через joblib + ArtifactManifest
              │
              └── Extra engines (опционально, через Protocol)
                    ├── JaccardEngine
                    │     rapidfuzz.token_sort_ratio
                    │     Не требует модели — чисто строки
                    └── EnsembleEngine
                          Взвешенная композиция движков
                          {cosine: 0.7, jaccard: 0.3}

SimilarityEngine (Protocol)
  │
  ├── JaccardEngine        — fuzzy string, без модели
  ├── EnsembleEngine       — взвешенная композиция
  └── (любой кастомный движок, реализующий Protocol)
```

## Возможности системы

### Сбор вакансий с hh.ru
Гибкий поиск по профессиям, регионам и отраслям. Поддержка одиночных запросов, пакетного режима и интерактивного меню. Автоматическое разбиение периода на интервалы для обхода ограничения в 2000 вакансий.

### Извлечение и нормализация навыков
Очистка, приведение к каноническому виду, фильтрация мусора, синонимия, учёт фразовых конструкций. Гибридные веса BM25 + эмбеддинги (SentenceTransformer) с PCA-оптимизацией для больших словарей.

### Таксономия навыков и профессий
Иерархическая классификация по 19 категориям (языки программирования, базы данных, DevOps, Data Science, безопасность и др.) + таксономия профессий с привязкой к доменам. Используется для осмысленных имён кластеров, категоризации рекомендаций и визуализации покрытия.

### Gap-анализ по уровням (junior/middle/senior)
Оценка покрытия рынка с учётом востребованности навыков, выявление дефицитов с приоритетами, расчёт готовности студента к целевому уровню. Доменный анализ по 15 направлениям с весом доминирующего домена.

### Покрытие целевой профессии
Оценка соответствия навыков студента доменам целевой профессии. Средневзвешенное покрытие по всем доменам профессии (Backend, Data Science, DevOps и т.д.) с детализацией по каждому домену.

### ML-ранжирование навыков (LTR + XGBoost)
Обучение на собранных вакансиях, предсказание важности недостающих навыков (0–100%), генерация объяснений с помощью SHAP (всегда включён, fallback на частотный/уровневый анализ при ошибке SHAP). Персонализированные признаки: семантическая близость к профилю, категория навыка, уровень. Кросс-доменные объяснения: если навык относится к неосновному для студента домену (например, Docker для Data Science), в why_important добавляется пояснение про частоту запроса навыка в другом домене.

### Кластеризация вакансий
KMeans/HDBSCAN с автоопределением числа кластеров по silhouette score. Определение профессиональных ролей с человекочитаемыми именами на основе таксономии. Ближайшие роли показываются вместе с семантической близостью и покрытием навыков.

### Визуализация и отчёты
Столбчатые диаграммы сравнения профилей, радарные диаграммы навыков, тепловые карты покрытия и совместной встречаемости навыков (Jaccard-матрица). Все графики в высоком разрешении (DPI 300).

### Анализ трендов
Отслеживание динамики спроса на навыки по историческим снимкам рынка. Выявление растущих, падающих и стабильных навыков. Временные ряды для топ-10 навыков.

### Типизированные контракты PipelineContext
Строгие Pydantic-модели для данных между этапами пайплайна вместо сырых dict. Явные поля, валидация на лету, поддержка IDE-автодополнения.

### Result[T, E] pattern
Явная обработка ошибок без исключений. Ok(value) / Err(error) с методами unwrap, unwrap_or, ok, err — замена try/except в бизнес-логике.

### monitoring/metrics.py — Prometheus-метрики
Histogram для длительности стадий/запросов, Counter для ошибок/рекомендаций, Gauge для состояния пайплайна. Экспорт через `/metrics` (text/plain) и `/api/admin/monitoring` (JSON).

### vector_search/faiss_index.py — FAISS-индексы
Обёртка над FAISS. FlatIP (точное косинусное сходство) и HNSW (приближённое для >10k векторов). Сохранение/загрузка через `.faiss` + `.pkl` (метаданные).

### REST API (api_pkg)
Модульный FastAPI-пакет `src/api_pkg/` с 10 роутерами: health, vacancies, profiles, clusters, taxonomy, trends, market, pipeline, results, admin. Встроенный startup/hooks, зависимости через Depends.

### n8n интеграция
Внешний оркестратор n8n запускает воркфлоу по расписанию и через webhook'и. Документация: `src/n8n/n8n_guide.md`. Доступные воркфлоу:
- `nightly_pipeline.json` — еженочный пайплайн + Telegram
- `trend_alert.json` — алёрт при скачке тренда
- `student_onboarding.json` — приём студента
- `weekly_report.json` — еженедельный отчёт (LLM + Telegram + Email + Postgres)

### Pydantic API response models
FastAPI-эндпоинты возвращают строго типизированные Pydantic-схемы (30+ моделей). Документация /docs и /openapi.json генерируется автоматически.

### Mypy strict checking
Весь проект проверяется mypy --strict. Типизация Generics, Literal, Union — отлов ошибок на этапе CI.

### Мониторинг (Prometheus + `/metrics`)
Модуль `src/monitoring/metrics.py` — Prometheus-метрики (Histogram, Counter, Gauge) для:
- Pipeline: количество запусков, ошибок, длительность по стадиям
- API: количество запросов, длительность, статусы
- LTR-модель: количество рекомендаций, ошибки SHAP

FastAPI-эндпоинт `GET /metrics` для сбора Prometheus-сервером.
Структурированный JSON-эндпоинт `GET /api/admin/monitoring` для фронтенда.
Instrumentation пайплайна: `DataCollectionStage`, `ModelTrainingStage`, `GapAnalysisStage` обёрнуты в `@track_pipeline_stage`.

### Векторный поиск (FAISS)
Модуль `src/vector_search/faiss_index.py` — обёртка над FAISS (FlatIP / HNSW). `build`/`search`/`save`/`load`. Экспортируется через `create_faiss_index()`. Готов к интеграции в `EmbeddingComparator` / `BM25Ranker` при росте объёма векторов (>1000).

### Вспомогательные скрипты
Расширение белого списка навыков (`scripts/extend_it_skills.py`). Проверка кластеров (`scripts/check_clusters.py`). Обучение кластеров (`scripts/train_clusters.py`). Полная пересборка проекта (`scripts/full_rebuild.py`).

### React-фронтенд (Vite + shadcn/ui)
SPA с корневым состоянием в `App.tsx`, 12+ компонентов, TypeScript, Tailwind CSS. DataViewer для табличного просмотра, AnalysisTab с поднятым состоянием, RecommendationsReport с gap-объяснениями и тултипами. Vite proxy для /api → backend. MonitoringTab для администратора: карточки метрик (запуски пайплайна, ошибки, рекомендации, API-запросы), средняя длительность стадий, метрики LTR, разбивка ошибок по стадиям.

## Ключевые модули

### Пайплайн (`src/pipeline/`)
- **orchestrator.py** — PipelineOrchestrator: запуск стадий, прогресс, event bus
- **stages.py** — 8 конкретных этапов (DataCollection, ClusterTraining, ModelTraining, GapAnalysis и др.)
- **progress.py** — SSE-прогресс через progress.json
- **clean.py** — очистка чистых отчётов и промежуточных данных
- Инструментирование: стадии DataCollection, ModelTraining, GapAnalysis обёрнуты в `@track_pipeline_stage`

### Парсинг (`src/parsing/`)
- **hh_api.py** — синхронный клиент hh.ru с кэшированием токена
- **bm25_ranker.py** — BM25Okapi с лемматизацией и minmax-нормализацией
- **hybrid_weight_calculator.py** — BM25 + SentenceTransformer + PCA
- **skill_normalizer.py** — синонимы, fuzzy-матчинг, приведение к канону

### Анализ (`src/analyzers/`)
- **profile_evaluator.py** — центральный модуль: считает покрытие, готовность, gap
- **profession_taxonomy.py** — таксономия профессий: домены, покрытие
- **skill_taxonomy.py** — 19 категорий навыков (Singleton)
- **vacancy_clustering.py** — KMeans/HDBSCAN с автоопределением k

### Рекомендации (`src/predictors/`)
- **recommendation_engine.py** — смешивание evaluator + LTR + тренды + домены
- **ltr_recommendation_engine.py** — XGBoost + SHAP

### API (`src/api_pkg/`)
FastAPI-сервер: middleware (request_id, body limit, CORS), rate limiting, роутеры для рекомендаций, pipeline, статуса. Эндпоинт `GET /metrics` для Prometheus, `GET /api/admin/monitoring` для фронтенда.

### Мониторинг (`src/monitoring/`)
Prometheus-метрики: pipeline_stage_duration_seconds, pipeline_stage_errors_total, api_request_duration_seconds, ltr_recommendations_generated_total, ltr_model_shap_errors_total. Декоратор `@track_pipeline_stage` для инструментирования стадий пайплайна.

### Векторный поиск (`src/vector_search/`)
FAISS-обёртка: `FaissIndex` с поддержкой FlatIP и HNSW. Методы `build`, `search`, `save`, `load`. Экспортируется через `create_faiss_index()`. Готов к интеграции в EmbeddingComparator/BM25Ranker.

### Скоринг и фильтрация спама (`src/scoring/`)
**`VacancyQualityScorer`** оценивает качество вакансий и отфильтровывает спам перед агрегацией в Excel.

**Критерии спама (каждый снижает score):**
| Флаг | Условие | Вычет |
|------|---------|-------|
| `NO_DESCRIPTION` | Описание отсутствует | –0.25 |
| `TOO_SHORT_DESCRIPTION` | Описание < 100 символов | –0.15 |
| `NO_SKILLS` | Нет key_skills | –0.20 |
| `TOO_FEW_SKILLS` | Меньше 2 key_skills | –0.10 |
| `SUSPICIOUS_EMPLOYER` | Работодатель — кадровое агентство / рекрутинг / аутсорс | –0.30 |
| `GENERIC_NAME` | Название общее ("вакансия", "работа", "водитель") | –0.30 |
| `PROMO_DESCRIPTION` | Рекламный текст ("самая высокая зарплата") | –0.20 |
| `EXCESSIVE_URLS` | > 3 ссылок в описании | –0.15 |
| `SALARY_ANOMALY` | Зарплата > 1 млн | –0.20 |

Итоговый `score = max(0, 1 - сумма_вычетов)`. Если `score < 0.5` — вакансия помечается спамом.

**Интеграция:** результат `filter_vacancies()` передаётся в `VacancyParser.aggregate_to_dataframe()`, который добавляет в Excel колонки «Спам» и «Причина спама».

**Пример вывода отчёта:**
```
==================================================
  VACANCY QUALITY REPORT
==================================================
  Total vacancies:  150
  Clean:            138
  Spam:             12 (8.0%)
  Avg score:        0.723

  Spam reasons:
    * SUSPICIOUS_EMPLOYER: 5
    * GENERIC_NAME: 4
    * NO_DESCRIPTION: 3

  Spam vacancies:
    [0.35] Менеджер по продажам @ Кадровое агентство
         Reasons: SUSPICIOUS_EMPLOYER; GENERIC_NAME
    [0.40] Вакансия @ HR-рекрутинг
         Reasons: GENERIC_NAME; NO_DESCRIPTION
==================================================
```

## Data Flow

```
Вакансии hh.ru → JSON → SkillExtractor → skill_freq + hybrid_weights
                                              ↓
Студенты → JSON → ProfileEvaluator ← WeightCleaner
                       ↓
              EmbeddingComparator
                ├── cosine_similarity (sklearn)
                └── extra_engines (Jaccard/Ensemble)

              Метрики + рекомендации → data/result/
```

## Профили студентов

- **base** — базовый набор компетенций (Junior Data Scientist)
- **dc** — дополнительный набор (Middle Data Scientist)
- **top_dc** — углублённый набор (Senior Data Scientist)

## Справочники (`data/reference/`)

| Файл | Назначение |
|------|------------|
| `profession_taxonomy.json` | Профессии → домены |
| `skill_taxonomy.json` | 19 категорий навыков |
| `domain_map.json` | 15 доменов → навыки |
| `it_skills.json` | Белый список IT-навыков |

## Технологии

FastAPI, Pydantic v2, XGBoost, SHAP,
SentenceTransformers, rank_bm25, pymorphy3, scikit-learn, matplotlib, structlog,
rapidfuzz (JaccardEngine), joblib (сериализация вместо pickle)

## Планируемые улучшения

*(список закрыт — приоритеты пересмотрены)*
