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
```

## Ключевые модули

### Парсинг (`src/parsing/`)
- **hh_api.py** — синхронный клиент hh.ru с кэшированием токена
- **bm25_ranker.py** — BM25Okapi с лемматизацией и minmax-нормализацией
- **hybrid_weight_calculator.py** — BM25 + SentenceTransformer + PCA
- **skill_normalizer.py** — синонимы, fuzzy-матчинг, приведение к канону

### Анализ (`src/analyzers/`)
- **profile_evaluator.py** — центральный модуль: считает покрытие, готовность, gap
- **profession_taxonomy.py** — таксономия профессий: домены, КРМ-коды, покрытие
- **skill_taxonomy.py** — 19 категорий навыков (Singleton)
- **vacancy_clustering.py** — KMeans/HDBSCAN с автоопределением k

### Рекомендации (`src/predictors/`)
- **recommendation_engine.py** — смешивание evaluator + LTR + тренды + домены
- **ltr_recommendation_engine.py** — XGBoost + SHAP

### API (`src/api.py`)
FastAPI-сервер: endpoints для рекомендаций, pipeline, статуса

## Data Flow

```
Вакансии hh.ru → JSON → SkillExtractor → skill_freq + hybrid_weights
                                              ↓
Студенты → JSON → ProfileEvaluator ← WeightCleaner
                       ↓
              Метрики + рекомендации → data/result/
```

## Профили студентов

- **base** — базовый набор компетенций (Junior Data Scientist)
- **dc** — дополнительный набор (Middle Data Scientist)
- **top_dc** — углублённый набор (Senior Data Scientist)

## Справочники (`data/reference/`)

| Файл | Назначение |
|------|------------|
| `profession_taxonomy.json` | Профессии → домены + КРМ-коды |
| `krm_competency_mapping.json` | КРМ-коды → навыки |
| `skill_taxonomy.json` | 19 категорий навыков |
| `domain_map.json` | 15 доменов → навыки |
| `it_skills.json` | Белый список IT-навыков |

## Технологии

FastAPI, Pydantic v2, SQLAlchemy (не используется), XGBoost, SHAP,
SentenceTransformers, rank_bm25, pymorphy3, scikit-learn, matplotlib, structlog

## Планируемые улучшения

Полный список — в `ROADMAP.md`.
