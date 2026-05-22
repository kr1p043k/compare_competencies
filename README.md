# Сравнение компетенций учебной программы с требованиями рынка труда (hh.ru)

Проект предназначен для анализа соответствия учебных компетенций студентов реальным требованиям IT-рынка, извлекаемым из вакансий hh.ru.
Система собирает вакансии, нормализует навыки, выполняет gap-анализ и формирует персонализированные рекомендации с использованием машинного обучения.

## 🎯 Основные возможности

- **Сбор вакансий с hh.ru**
  Гибкий поиск по профессиям, регионам и отраслям. Поддержка одиночных запросов, пакетного режима и интерактивного меню. Автоматическое разбиение периода на интервалы для обхода ограничения в 2000 вакансий.

- **Извлечение и нормализация навыков**
  Очистка, приведение к каноническому виду, фильтрация мусора, синонимия, учёт фразовых конструкций. Гибридные веса BM25 + эмбеддинги (SentenceTransformer) с PCA-оптимизацией для больших словарей.

- **Таксономия навыков и профессий**
   Иерархическая классификация по 19 категориям (языки программирования, базы данных, DevOps, Data Science, безопасность и др.) + таксономия профессий с привязкой к доменам и КРМ-компетенциям. Используется для осмысленных имён кластеров, категоризации рекомендаций и визуализации покрытия.

- **Gap-анализ по уровням (junior/middle/senior)**
   Оценка покрытия рынка с учётом востребованности навыков, выявление дефицитов с приоритетами, расчёт готовности студента к целевому уровню. Доменный анализ по 15 направлениям с весом доминирующего домена.

- **Покрытие целевой профессии**
   Оценка соответствия навыков студента доменам целевой профессии. Средневзвешенное покрытие по всем доменам профессии (Backend, Data Science, DevOps и т.д.) с детализацией по каждому домену.

- **ML-ранжирование навыков (LTR + XGBoost)**
  Обучение на собранных вакансиях, предсказание важности недостающих навыков (0–100%), генерация объяснений с помощью SHAP. Персонализированные признаки: семантическая близость к профилю, категория навыка, уровень.

- **Кластеризация вакансий**
  KMeans/HDBSCAN с автоопределением числа кластеров по silhouette score. Определение профессиональных ролей с человекочитаемыми именами на основе таксономии. Ближайшие роли показываются вместе с семантической близостью и покрытием навыков.

- **Визуализация и отчёты**
  Столбчатые диаграммы сравнения профилей, радарные диаграммы навыков, тепловые карты покрытия и совместной встречаемости навыков (Jaccard-матрица). Все графики в высоком разрешении (DPI 300).

- **Анализ трендов**
  Отслеживание динамики спроса на навыки по историческим снимкам рынка. Выявление растущих, падающих и стабильных навыков. Временные ряды для топ-10 навыков.

- **Вспомогательные скрипты**
  Расширение белого списка навыков (`extend_it_skills.py`) с интерактивным подтверждением и интеграцией с таксономией. Проверка кластеров (`check_clusters.py`). Полная пересборка проекта (`full_rebuild.py`).

## 📁 Структура проекта

```plaintext
└── 📁 competency_comparison
    │
    ├── 📁 data/
    │   ├── 📁 cache/                            # Единый кэш быстрых данных
    │   │   ├── 📄 parsed_skills.pkl             # Кэш извлечённых навыков
    │   │   ├── 📁 embeddings/                   # Рыночные эмбеддинги
    │   │   │   ├── 📄 market_embeddings_junior.pkl
    │   │   │   ├── 📄 market_embeddings_middle.pkl
    │   │   │   └── 📄 market_embeddings_senior.pkl
    │   │   ├── 📁 clusters/                     # Модели кластеров вакансий
    │   │   │   ├── 📄 vacancy_clusters_junior.pkl
    │   │   │   ├── 📄 vacancy_clusters_middle.pkl
    │   │   │   └── 📄 vacancy_clusters_senior.pkl
    │   │   └── 📁 students/                     # Кэш эмбеддингов студентов
    │   │       ├── 📄 base_embedding.json
    │   │       ├── 📄 dc_embedding.json
    │   │       └── 📄 top_dc_embedding.json
    │   │
    │   ├── 📁 history/                          # Исторические снимки частот навыков
    │   │   └── 📄 freq_2025-04-01-120000.json
    │   │
    │   ├── 📁 last_uploaded/                    # Последняя загруженная матрица компетенций
    │   │   └── 📄 competency_matrix.csv
    │   │
    │   ├── 📁 models/                           # Обученные ML-модели
    │   │   ├── 📄 ltr_ranker_xgb_regressor.joblib
    │   │   └── 📄ltr_feature_importance.png
    │   │
    │   ├── 📁 processed/                        # Обработанные данные после парсинга
    │   │   ├── 📄 competency_frequency.json        # Частоты навыков на рынке
    │   │   ├── 📄 competency_frequency_mapped.json # Частоты, сопоставленные с учебными компетенциями
    │   │   ├── 📄 competency_mapping.json          # Маппинг: коды компетенций → рыночные навыки
    │   │   ├── 📄 skill_weights.json               # Очищенные веса навыков для gap-анализа
    │   │   └── 📄 hh_vacancies_detailed.json       # Детальные вакансии (с key_skills и описанием)
    │   │
    │   ├── 📁raw/                               # Сырые входные данные
    │   │   ├── 📄 hh_vacancies_basic.json        # Базовые вакансии (результат поиска)
    │   │   └── 📄competency_matrix.csv          # Исходная матрица компетенций
    │   │
    │   ├── 📁reference/                         # Эталонные справочники (не генерируются)
    │   │   ├── 📄it_skills.json                 # Белый список IT-навыков
    │   │   ├── 📄skill_taxonomy.json            # Таксономия навыков по категориям
    │   │   ├── 📄domain_map.json                # Доменная карта (15 доменов)
    │   │   ├── 📄 hard_skills.json               # Список "жёстких" навыков для рекомендаций
    │   │   ├── 📄 trend_hot_skills.json          # Горячие навыки для бонусов трендов
    │   │   ├── 📄 timeframe_groups.json          # Группы для оценки времени изучения
    │   │   ├── 📄skill_blacklist.json           # Чёрный список навыков
    │   │   ├── 📄 generic_words.json             # Общие слова, исключаемые из навыков
    │   │   ├── 📄 filler_words.json              # Слова-паразиты
    │   │   ├── 📄 profession_taxonomy.json       # Таксономия профессий → домены + КРМ-коды
    │   │   ├── 📄 krm_competency_mapping.json    # КРМ-компетенции → наборы навыков
    │   │   └── 📄 stop_lemmas.json               # Стоп-леммы для фильтрации BM25
    │   │
    │   ├── 📁 result/                            # Результаты анализа (отчёты, графики)
    │   │   ├── 📁 base/                          # Профиль base (junior)
    │   │   │   ├── 📄 full_recommendations_base.json
    │   │   │   ├── 📄 ltr_recommendations_base.json
    │   │   │   ├── 📄 radar_base.png
    │   │   │   ├── 📄 weights_base.png
    │   │   │   ├── 📄 ml_importance_base.png
    │   │   │   ├── 📄 deficits_base.png
    │   │   │   └── 📄 cluster_insights_base.png
    │   │   ├── 📁 dc/                            # Профиль dc (middle)
    │   │   │   ├── 📄 full_recommendations_dc.json
    │   │   │   ├── 📄 ltr_recommendations_dc.json
    │   │   │   ├── 📄 radar_dc.png
    │   │   │   ├── 📄 weights_dc.png
    │   │   │   ├── 📄 ml_importance_dc.png
    │   │   │   ├── 📄 deficits_dc.png
    │   │   │   └── 📄 cluster_insights_dc.png
    │   │   ├── 📁top_dc/                        # Профиль top_dc (senior)
    │   │   │   ├── 📄 full_recommendations_top_dc.json
    │   │   │   ├── 📄 ltr_recommendations_top_dc.json
    │   │   │   ├── 📄 radar_top_dc.png
    │   │   │   ├── 📄 weights_top_dc.png
    │   │   │   ├── 📄 ml_importance_top_dc.png
    │   │   │   ├── 📄 deficits_top_dc.png
    │   │   │   └── 📄 cluster_insights_top_dc.png
    │   │   ├── 📁trends/                        # Графики трендов
    │   │   │   ├── 📄 trending_skills.png
    │   │   │   └── 📄 skill_timeline.png
    │   │   ├── 📄 coverage_comparison.png
    │   │   ├── 📄 skills_heatmap.png
    │   │   ├── 📄 skill_correlation_heatmap.png
    │   │   ├── 📄 profession_coverage.png     # Покрытие доменов целевой профессии
    │   │   ├── 📄 domain_skill_gaps.png
    │   │   └── 📄 profiles_comparison_summary.json
    │   │
    │   └── 📁students/                          # Профили студентов (учебные компетенции)
    │       ├── 📄 base_competency.json
    │       ├── 📄 dc_competency.json
    │       ├── 📄 top_dc_competency.json
    │       └── 📄 description_of_competency.txt
    │
    ├── 📁 frontend/
    │   └── 📄 app.py                    # приложение для визуализации
    │
    ├── 📁 notebooks/                    # Jupyter ноутбуки для интерактивного анализа
    │
    ├── 📁 scripts/
    │   ├── 📄 check_clusters.py         # Проверка обученных кластеров вакансий
    │   ├── 📄 extend_it_skills.py       # Расширение белого списка навыков из вакансий
    │   ├── 📄 full_rebuild.py           # Полная очистка кэшей и пересборка проекта
    │   └── 📄 train_clusters.py         # Обучение кластеров вакансий по уровням
    │
    ├── 📁 src/
    │   ├── 📁 analyzers/
    │   │   ├── 📁 comparison/
    │   │   │   ├── 📄 comparator.py         # Сравнение навыков студента с рынком (TF-IDF / эмбеддинги)
    │   │   │   ├── 📄 embedding_comparator.py # Семантическое сравнение через эмбеддинги + FAISS
    │   │   │   └── 📄 domain_analyzer.py    # Анализ покрытия по 15 доменам (Backend, DS, Frontend…)
    │   │   ├── 📁 gap/
    │   │   │   ├── 📄 gap_analyzer.py       # Gap-анализ: разница между навыками студента и рынка
    │   │   │   └── 📄 profile_evaluator.py  # Оценка профиля: метрики покрытия, readiness, кластерный контекст
    │   │   ├── 📁 skills/
    │   │   │   ├── 📄 skill_filter.py       # Фильтрация навыков: удаление мусора, нормализация весов
    │   │   │   ├── 📄 skill_level_analyzer.py # Распределение навыков по уровням (junior/middle/senior)
    │   │   │   ├── 📄 skill_taxonomy.py     # Таксономия навыков: категоризация по 19 категориям (Singleton)
    │   │   │   ├── 📄 skill_correlation.py  # Анализ совместной встречаемости навыков (матрица Jaccard)
    │   │   │   ├── 📄 profession_taxonomy.py # Таксономия профессий: домены, КРМ-коды, покрытие
    │   │   │   └── 📄 trends.py             # Анализ трендов: сравнение исторических снимков рынка
    │   │   └── 📁 clustering/
    │   │       └── 📄 vacancy_clustering.py # Кластеризация вакансий (KMeans/HDBSCAN + автоопределение k)
    │   │
    │   ├── 📁 loaders/
    │   │   └── 📄 student_loader.py     # Загрузка профилей студентов из JSON/CSV
    │   │
    │   ├── 📁 models/
    │   │   ├── 📄 comparison.py         # Модели для сравнения профилей (ComparisonReport, GapResult)
    │   │   ├── 📄 competency.py         # Модель компетенции
    │   │   ├── 📄 data_contracts.py     # Строгие Pydantic-контракты между этапами пайплайна
    │   │   ├── 📄 enums.py              # Централизованные Enum'ы (уровни, приоритеты, тренды)
    │   │   ├── 📄 hh_responses.py       # Pydantic-модели для ответов API hh.ru
    │   │   ├── 📄 market_metrics.py     # Метрики рынка: SkillMetrics, DomainMetrics
    │   │   ├── 📄 student.py            # Модель студента (StudentProfile, ExperienceLevel)
    │   │   └── 📄 vacancy.py            # Модель вакансии (Vacancy, KeySkill, Salary…)
    │   │
    │   ├── 📁 parsing/
    │   │   ├── 📁 api/
    │   │   │   ├── 📄 hh_api.py             # Синхронный клиент для API hh.ru (авторизация, поиск)
    │   │   │   ├── 📄 hh_api_async.py       # Асинхронный клиент (пакетная загрузка вакансий)
    │   │   │   └── 📄 embedding_loader.py   # Загрузка модели эмбеддингов (SentenceTransformer, Singleton)
    │   │   ├── 📁 skills/
    │   │   │   ├── 📄 skill_normalizer.py   # Нормализация навыков: синонимы, версии, fuzzy-матчи
    │   │   │   ├── 📄 skill_parser.py       # Извлечение навыков из текста вакансий
    │   │   │   ├── 📄 skill_validator.py    # Валидация навыков: чёрный/белый списки, фильтрация мусора
    │   │   │   ├── 📄 vacancy_parser.py     # Фасад парсера: BM25, гибридные веса, эмбеддинги, PCA
    │   │   │   ├── 📄 bm25_ranker.py        # BM25-ранжер с предфильтрацией и кэшированием
    │   │   │   ├── 📄 hybrid_weight_calculator.py # Гибридные веса (BM25 + Embeddings + PCA)
    │   │   │   └── 📄 skill_embedding_cache.py  # Атомарный JSON-кэш эмбеддингов навыков
    │   │   └── 📄 utils.py                  # Утилиты парсинга: загрузка whitelist, интерактивный режим
    │   │
    │   ├── 📁 pipeline/
    │   │   ├── 📄 helpers.py                # Общие функции: загрузка деталей, режим асинхронности, сохранение
    │   │   ├── 📄 data_source.py            # Загрузка вакансий из кэша или через hh.ru
    │   │   ├── 📄 skill_extractor.py        # Извлечение навыков из вакансий или из кэша
    │   │   ├── 📄 weight_cleaner.py         # Фильтрация hybrid_weights через справочники
    │   │   ├── 📄 level_builder.py          # Подготовка level_vacancies_data и vacancies_skills
    │   │   ├── 📄 gap_runner.py             # Оркестрация gap‑анализа и генерации рекомендаций
    │   │   ├── 📄 metric_computer.py        # Оценка профилей (использует ProfileEvaluator)
    │   │   └── 📄 recommendation_runner.py  # Генерация персональных рекомендаций (движок)
    │   │
    │   ├── 📁 predictors/
    │   │   ├── 📄 ltr_recommendation_engine.py  # LTR-модель на XGBoost (обучение, предсказание, SHAP)
    │   │   ├── 📄 recommendation_engine.py      # Движок рекомендаций: смешивание скоров, тренды, домены, объяснения
    │   │   └── 📄 skill_forecast.py             # Прогнозирование трендов навыков (заглушка)
    │   │
    │   ├── 📁 visualization/
    │   │   ├── 📄 __init__.py              # Публичный API пакета визуализации
    │   │   ├── 📄 _config.py               # Настройки стилей и эмодзи → текст
    │   │   ├── 📄 _utils.py                # Загрузка skill_weights, ML-рекомендаций
    │   │   ├── 📄 coverage.py              # Графики покрытия рынка и тепловая карта
    │   │   ├── 📄 radar.py                 # Радарная диаграмма навыков
    │   │   ├── 📄 importance.py            # Важность навыков (ML + распределение весов)
    │   │   ├── 📄 correlation.py           # Тепловая карта совместной встречаемости
    │   │   ├── 📄 clusters.py              # Ближайшие кластеры вакансий
    │   │   └── 📄 orchestration.py         # Сохранение всех графиков, запуск ноутбуков, контекстная информация
    │   │
    │   ├── 📄 config.py                # Централизованная конфигурация (Pydantic Settings)
    │   ├── 📄 logging_config.py        # Настройка structlog + маскирование секретов
    │   ├── 📄 artifacts.py             # Манифест артефактов (версия, хеш, метрики)
    │   ├── 📄 api.py                   # FastAPI-сервер для доступа к рекомендациям
    │   └── 📄 utils.py                 # Общие утилиты: atomic_write_json, safe_read_json, валидация путей
    ├── 📁 tests                            # Тесты
    │   ├── 📁 analyzers                    # Тесты анализаторов
    │   ├── 📁 integration                  # Интеграционные тесты
    │   ├── 📁 loaders                      # Тесты загрузчиков
    │   ├── 📁 models                       # Тесты моделей
    │   ├── 📁 parsing                      # Тесты парсинга
    │   ├── 📁 predictors                   # Тесты предикторов
    │   ├── 📁 scripts                      # Тесты скриптов
    │   ├── 📁 visualization                # Тесты графиков
    │   └── 📄 conftest.py                  # Фикстуры для pytest
    ├── 📄 main.py                         # Точка входа: полный пайплайн сбора, анализа и рекомендаций
    ├── 📄 README.md                       # Документация проекта
    ├── 📄 requirements.txt                # Prod-зависимости
    ├── 📄 requirements-dev.txt            # dev-зависимости
    ├── 📄 MakeFile
    └── 📄 user_manual.md                  # Руководство пользователя
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

6. **Визуализация**
   `charts.py` строит графики покрытия, важности навыков, радарные диаграммы и сохраняет их в `data/result/`.

## 📦 Зависимости

Основные пакеты перечислены в `requirements.txt`:
- `requests`, `aiohttp` – работа с API
- `pandas`, `numpy` – обработка данных
- `scikit‑learn`, `xgboost`, `shap` – машинное обучение
- `sentence‑transformers` – эмбеддинги
- `matplotlib`, `seaborn` – визуализация
- `pydantic` – модели данных
- `lightgbm` (опционально), `faiss‑cpu` (опционально)

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
