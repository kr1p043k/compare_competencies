# Сравнение компетенций учебной программы с требованиями рынка труда (hh.ru)

Проект предназначен для анализа соответствия учебных компетенций студентов реальным требованиям IT-рынка, извлекаемым из вакансий hh.ru.
Система собирает вакансии, нормализует навыки, выполняет gap-анализ и формирует персонализированные рекомендации с использованием машинного обучения.

## 🎯 Основные возможности

- **Сбор вакансий с hh.ru**
  Гибкий поиск по профессиям, регионам и отраслям. Поддержка одиночных запросов, пакетного режима и интерактивного меню. Автоматическое разбиение периода на интервалы для обхода ограничения в 2000 вакансий.

- **Извлечение и нормализация навыков**
  Очистка, приведение к каноническому виду, фильтрация мусора, синонимия, учёт фразовых конструкций. Гибридные веса BM25 + эмбеддинги (SentenceTransformer) с PCA-оптимизацией для больших словарей.

- **Таксономия навыков**
  Иерархическая классификация по 19 категориям (языки программирования, базы данных, DevOps, Data Science, безопасность и др.). Используется для осмысленных имён кластеров, категоризации рекомендаций и визуализации покрытия.

- **Gap-анализ по уровням (junior/middle/senior)**
  Оценка покрытия рынка с учётом востребованности навыков, выявление дефицитов с приоритетами, расчёт готовности студента к целевому уровню. Доменный анализ по 15 направлениям с весом доминирующего домена.

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
└── 📁 competency_comparison                 # Корень проекта: сравнение компетенций с рынком
    ├── 📁 data                              # Все данные проекта
    │   ├── 📁 embeddings                    # Кэш эмбеддингов навыков и вакансий
    │   │   └── 📁 cache                     # Кэшированные эмбеддинги (npz + index.json)
    │   ├── 📁 history                       # Исторические снимки частот навыков для анализа трендов
    │   ├── 📁 last_uploaded                 # Последняя загруженная матрица компетенций (CSV)
    │   │   └── 📄 competency_matrix.csv     # Матрица: дисциплины × компетенции
    │   ├── 📁 models                        # Обученные ML-модели
    │   │   └── 📄 ltr_ranker_xgb_regressor.joblib  # LTR-модель ранжирования навыков (XGBoost)
    │   ├── 📁 processed                     # Обработанные данные после парсинга и анализа
    │   │   ├── 📄 competency_frequency.json        # Частоты навыков на рынке
    │   │   ├── 📄 competency_frequency_mapped.json # Частоты, сопоставленные с учебными компетенциями
    │   │   ├── 📄 competency_mapping.json          # Маппинг: коды компетенций → рыночные навыки
    │   │   ├── 📄 skill_weights.json               # Очищенные веса навыков для gap-анализа
    │   │   ├── 📄 profiles_comparison_summary.json # Сводка метрик по всем профилям
    │   │   ├── 📄 cluster_training_report.json     # Отчёт об обучении кластеров
    │   │   ├── 📄 parsed_skills.pkl                # Кэш результатов парсинга навыков
    │   │   ├── 📄 vacancy_clusters_junior.pkl      # Модель кластеров для junior-вакансий
    │   │   ├── 📄 vacancy_clusters_middle.pkl      # Модель кластеров для middle-вакансий
    │   │   └── 📄 vacancy_clusters_senior.pkl      # Модель кластеров для senior-вакансий
    │   ├── 📁 raw                           # Сырые данные (входные)
    │   │   ├── 📄 competency_matrix.csv            # Исходная матрица компетенций
    │   │   ├── 📄 hh_vacancies_basic.json          # Базовые данные вакансий (без деталей)
    │   │   └── 📄 hh_vacancies_detailed.json       # Детальные данные вакансий (с key_skills и описанием)
    │   ├── 📁 result                        # Результаты анализа: отчёты и графики
    │   │   ├── 📁 base                            # Результаты для профиля base (junior)
    │   │   │   ├── 📄 full_recommendations_base.json  # Полные рекомендации с объяснениями
    │   │   │   ├── 📄 ltr_recommendations_base.json   # LTR-рекомендации (топ-10)
    │   │   │   ├── 📄 radar_base.png                  # Радарная диаграмма навыков
    │   │   │   ├── 📄 ml_importance_base.png          # Важность навыков по ML-модели
    │   │   │   ├── 📄 cluster_insights_base.png       # Ближайшие кластеры вакансий
    │   │   │   └── 📄 deficits_base.png               # Топ дефицитов высокого спроса
    │   │   ├── 📁 dc                              # Результаты для профиля dc (middle)
    │   │   │   ├── 📄 full_recommendations_dc.json
    │   │   │   ├── 📄 ltr_recommendations_dc.json
    │   │   │   ├── 📄 radar_dc.png
    │   │   │   ├── 📄 ml_importance_dc.png
    │   │   │   ├── 📄 cluster_insights_dc.png
    │   │   │   └── 📄 deficits_dc.png
    │   │   ├── 📁 top_dc                          # Результаты для профиля top_dc (senior)
    │   │   │   ├── 📄 full_recommendations_top_dc.json
    │   │   │   ├── 📄 ltr_recommendations_top_dc.json
    │   │   │   ├── 📄 radar_top_dc.png
    │   │   │   ├── 📄 ml_importance_top_dc.png
    │   │   │   ├── 📄 cluster_insights_top_dc.png
    │   │   │   └── 📄 deficits_top_dc.png
    │   │   ├── 📁 trends                          # Графики анализа трендов
    │   │   │   ├── 📄 trending_skills.png              # Растущие и падающие навыки
    │   │   │   └── 📄 skill_timeline.png               # Временные ряды топ-навыков
    │   │   ├── 📄 coverage_comparison.png              # Сравнение покрытия трёх профилей
    │   │   ├── 📄 skills_heatmap.png                   # Тепловая карта покрытия навыков
    │   │   ├── 📄 skill_correlation_heatmap.png        # Совместная встречаемость навыков
    │   │   └── 📄 hh_vacancies_detailed.json          # Детальные вакансии (кэш для --skip-collection)
    │   ├── 📁 students                      # Профили студентов (учебные компетенции)
    │   │   ├── 📄 base_competency.json            # Навыки профиля base (junior)
    │   │   ├── 📄 dc_competency.json              # Навыки профиля dc (middle)
    │   │   ├── 📄 top_dc_competency.json           # Навыки профиля top_dc (senior)
    │   │   └── 📄 description_of_competency.txt   # Описания всех компетенций
    │   ├── 📄 it_skills.json               # Белый список IT-навыков (~900+)
    │   └── 📄 skill_taxonomy.json           # Таксономия навыков по 19 категориям
    ├── 📁 frontend                         # Веб-интерфейс (Streamlit)
    │   └── 📄 app.py                       # Streamlit-приложение для визуализации
    ├── 📁 notebooks                        # Jupyter ноутбуки для интерактивного анализа
    │   ├── 📄 01_hh_analysis.ipynb         # Анализ частот навыков и распределений
    │   ├── 📄 02_competency_matching.ipynb # Gap-анализ и сравнение профилей
    │   └── 📄 03_prediction_model.ipynb    # Обучение и оценка ML-модели
    ├── 📁 scripts                          # Вспомогательные скрипты (запуск из командной строки)
    │   ├── 📄 check_clusters.py            # Проверка обученных кластеров вакансий
    │   ├── 📄 extend_it_skills.py          # Расширение белого списка навыков из вакансий
    │   ├── 📄 full_rebuild.py              # Полная очистка кэшей и пересборка проекта
    │   └── 📄 train_clusters.py            # Обучение кластеров вакансий по уровням
    ├── 📁 src                              # Исходный код проекта
    │   ├── 📁 analyzers                    # Модули анализа: сравнение, gap-анализ, кластеры
    │   │   ├── 📄 comparator.py            # Сравнение навыков студента с рынком (TF-IDF / эмбеддинги)
    │   │   ├── 📄 domain_analyzer.py       # Анализ покрытия по 15 доменам (Backend, Frontend, DS...)
    │   │   ├── 📄 embedding_comparator.py  # Семантическое сравнение через эмбеддинги + FAISS
    │   │   ├── 📄 gap_analyzer.py          # Gap-анализ: разница между навыками студента и рынка
    │   │   ├── 📄 profile_evaluator.py     # Оценка профиля студента: метрики, кластерный контекст
    │   │   ├── 📄 skill_correlation.py     # Анализ совместной встречаемости навыков (матрица Jaccard)
    │   │   ├── 📄 skill_filter.py          # Фильтрация навыков: удаление мусора, нормализация весов
    │   │   ├── 📄 skill_level_analyzer.py  # Распределение навыков по уровням (junior/middle/senior)
    │   │   ├── 📄 skill_taxonomy.py        # Таксономия навыков: категоризация по 19 категориям (Singleton)
    │   │   ├── 📄 trends.py                # Анализ трендов: сравнение исторических снимков рынка
    │   │   └── 📄 vacancy_clustering.py    # Кластеризация вакансий (KMeans/HDBSCAN + автоопределение k)
    │   ├── 📁 loaders                      # Загрузчики данных
    │   │   └── 📄 student_loader.py        # Загрузка профилей студентов из JSON/CSV
    │   ├── 📁 models                       # Pydantic/Dataclass модели данных
    │   │   ├── 📄 comparison.py            # Модели для сравнения профилей (ComparisonReport, GapResult)
    │   │   ├── 📄 competency.py            # Модель компетенции
    │   │   ├── 📄 market_metrics.py        # Метрики рынка: SkillMetrics, DomainMetrics
    │   │   ├── 📄 student.py               # Модель студента (StudentProfile, ExperienceLevel)
    │   │   └── 📄 vacancy.py               # Модель вакансии (Vacancy, KeySkill, Salary...)
    │   ├── 📁 parsing                      # Сбор и обработка вакансий
    │   │   ├── 📄 embedding_loader.py      # Загрузка модели эмбеддингов (SentenceTransformer, Singleton)
    │   │   ├── 📄 hh_api.py                # Синхронный клиент для API hh.ru (авторизация, поиск)
    │   │   ├── 📄 hh_api_async.py          # Асинхронный клиент для API hh.ru (пакетная загрузка)
    │   │   ├── 📄 skill_normalizer.py      # Нормализация навыков: синонимы, версии, fuzzy-матчи
    │   │   ├── 📄 skill_parser.py          # Извлечение навыков из текста вакансий
    │   │   ├── 📄 skill_validator.py       # Валидация навыков: чёрный/белый списки, фильтрация мусора
    │   │   ├── 📄 utils.py                 # Утилиты парсинга: загрузка whitelist, интерактивный режим
    │   │   └── 📄 vacancy_parser.py        # Главный парсер: BM25, hybrid-веса, эмбеддинги, PCA
    │   ├── 📁 predictors                   # ML-движки рекомендаций
    │   │   ├── 📄 ltr_recommendation_engine.py  # LTR-модель (XGBoost): обучение и предсказание важности
    │   │   ├── 📄 recommendation_engine.py      # Движок рекомендаций: объединяет gap-анализ и LTR
    │   │   └── 📄 skill_forecast.py             # Прогнозирование трендов навыков
    │   ├── 📁 visualization               # Графики и визуализация
    │   │   └── 📄 charts.py               # Все графики: покрытие, радар, heatmap, кластеры, корреляции
    │   ├── 📄 config.py                   # Конфигурация: пути, API-ключи, параметры моделей
    │   ├── 📄 logging_config.py           # Конфигурация: логи
    │   └── 📄 utils.py                    # Общие утилиты: atomic_write_json/npz, логирование
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
