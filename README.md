# Сравнение компетенций учебной программы с требованиями рынка труда (hh.ru)

Проект предназначен для анализа соответствия учебных компетенций студентов реальным требованиям IT-рынка, извлекаемым из вакансий hh.ru.  
Система собирает вакансии, нормализует навыки, выполняет gap-анализ и формирует персонализированные рекомендации с использованием машинного обучения.

## 🎯 Основные возможности

- **Сбор вакансий с hh.ru**  
  Гибкий поиск по профессиям, регионам и отраслям. Поддержка одиночных запросов, пакетного режима и интерактивного меню.

- **Извлечение и нормализация навыков**  
  Очистка, приведение к каноническому виду, фильтрация мусора, синонимия, учёт фразовых конструкций.

- **Семантическое сравнение (эмбеддинги)**  
  Модели Sentence‑Transformers оценивают смысловую близость навыков, повышая точность сопоставления.

- **Gap‑анализ по уровням (junior/middle/senior)**  
  Оценка покрытия рынка, выявление дефицитов с приоритетами, расчёт готовности студента к целевому уровню.

- **ML‑ранжирование навыков (LTR + XGBoost)**  
  Обучение на собранных вакансиях, предсказание важности недостающих навыков (0–100%), генерация объяснений с помощью SHAP.

- **Визуализация и отчёты**  
  Столбчатые диаграммы, радарные графики, тепловые карты, сохранение результатов в JSON и Excel.

- **Jupyter ноутбуки**  
  Интерактивный анализ данных, построение графиков, эксперименты с ML‑моделями.

## 📁 Структура проекта

```plaintext
└── 📁 competency_comparison
    ├── 📁 data
    │   ├── 📁 embeddings
    │   ├── 📁 last_uploaded
    │   │   └── 📄 competency_matrix.csv
    │   ├── 📁 processed
    │   │   ├── 📄 competency_frequency.json
    │   │   ├── 📄 competency_mapping.json
    │   │   └── 📄 market_competencies.csv
    │   ├── 📁 raw
    │   │   ├── 📄 competency_matrix.csv
    │   │   └── 📄 hh_vacancies.json
    │   ├── 📁 result
    │   │   ├── 📁 base
    │   │   │   ├── 📄 comparison_report_base.json
    │   │   │   └── 📄 recommendations_base.json
    │   │   ├── 📁 dc
    │   │   │   ├── 📄 comparison_report_dc.json
    │   │   │   └── 📄 recommendations_dc.json
    │   │   └── 📁 top_dc
    │   │       ├── 📄 comparison_report_top_dc.json
    │   │       └── 📄 recommendations_top_dc.json
    │   ├── 📁 students
    │   │   ├── 📄 base_competency.json
    │   │   ├── 📄 dc_competency.json
    │   │   ├── 📄 descriptiom_of_competency.txt
    │   │   └── 📄 top_dc_competency.json
    │   └── 📄 it_skills.json
    ├── 📁 frontend
    │   └── 📄 app.py
    ├── 📁 notebooks
    │   ├── 📄 01_hh_analysis.ipynb
    │   ├── 📄 02_competency_matching.ipynb
    │   └── 📄 03_prediction_model.ipynb
    ├── 📁 scripts
    │   └── 📄 train_clusters.py
    ├── 📁 src
    │   ├── 📁 analyzers
    │   │   ├── 📄 comparator.py
    │   │   ├── 📄 domain_analyzer.py
    │   │   ├── 📄 embedding_comparator.py
    │   │   ├── 📄 gap_analyzer.py
    │   │   ├── 📄 init.py
    │   │   ├── 📄 profile_evaluator.py
    │   │   ├── 📄 skill_filter.py
    │   │   ├── 📄 skill_level_analyzer.py
    │   │   └── 📄 trends.py
    │   ├── 📁 loaders
    │   │   ├── 📄 init.py
    │   │   └── 📄 student_loader.py
    │   ├── 📁 models
    │   │   ├── 📄 comparison.py
    │   │   ├── 📄 competency.py
    │   │   ├── 📄 init.py
    │   │   ├── 📄 market_metrics.py
    │   │   ├── 📄 student.py
    │   │   └── 📄 vacancy.py
    │   ├── 📁 parsing
    │   │   ├── 📄 hh_api_async.py
    │   │   ├── 📄 hh_api.py
    │   │   ├── 📄 init.py
    │   │   ├── 📄 skill_normalizer.py
    │   │   ├── 📄 skill_parser.py
    │   │   ├── 📄 skill_validator.py
    │   │   ├── 📄 utils.py
    │   │   └── 📄 vacancy_parser.py
    │   ├── 📁 predictors
    │   │   ├── 📄 init.py
    │   │   ├── 📄 ltr_recommendation_engine.py
    │   │   ├── 📄 recommendation_engine.py
    │   │   └── 📄 skill_forecast.py
    │   ├── 📁 visualization
    │   │   ├── 📄 charts.py
    │   │   └── 📄 init.py
    │   ├── 📄 config.py
    │   ├── 📄 init.py
    │   └── 📄 utils.py
    ├── 📁 tests
    │   ├── 📁 analyzers
    │   │   ├── 📄 test_analyzers.py
    │   │   ├── 📄 test_comparator.py
    │   │   └── 📄 test_gap_analyzers.py
    │   ├── 📁 integration
    │   │   └── 📄 test_full_pipeline.py
    │   ├── 📁 loaders
    │   │   └── 📄 test_loaders.py
    │   ├── 📁 models
    │   │   └── 📄 test_models.py
    │   ├── 📁 parsing
    │   │   └── 📄 test_parsers.py
    │   ├── 📁 predictors
    │   │   ├── 📄 test_forecast.py
    │   │   └── 📄 test_ml_recommendation.py
    │   ├── 📄 conftest.py
    │   └── 📄 init.py
    ├── 📄 main.py
    ├── 📄 README.md
    ├── 📄 requirements.txt
    └── 📄 user_manual.md
```

## 🚀 Быстрый старт

1. **Установите зависимости**  
   ```bash
   pip install -r requirements.txt

2. **Соберите вакансии**  
   ```bash
   python main.py --it-sector --excel

3. **Обучите ML‑модель ранжирования**  
   ```bash
   python main.py --train-model

4. **Запустите gap‑анализ и получите рекомендации**  
   ```bash
   python main.py --run-gap-analysis

5. **Визуализация**  
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
python main.py --train-model        # обучение LTR
python main.py --run-gap-analysis   # генерация рекомендаций

## 📖 Подробнее

Детальное руководство пользователя находится в файле user_manual.md.
Вопросы и предложения можно оставлять в Issues репозитория.

---
*Проект создан для образовательных и исследовательских целей.*