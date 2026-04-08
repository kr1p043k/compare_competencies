# compare_competencies
Сравнение компетенций учебной программы и с job-сайтами
Ориентировочное древо проекта:
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
    ├── 📁 src
    │   ├── 📁 analyzers
    │   │   ├── 📄 comparator.py
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
    │   │   ├── 📄 ml_recommendation_engine.py
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
    └── 📄 user_manual.md└── 📁 competency_comparison  # Корень проекта сравнения компетенций учеников с требованиями рынка труда (hh.ru)
    ├── 📁 data  # Централизованное хранилище всех данных: входных, промежуточных, выходных
    │   ├── 📁 embeddings  # Предвычисленные векторные представления навыков (для семантического поиска и сравнения)
    │   ├── 📁 last_uploaded  # Последняя успешно загруженная версия компетенций учеников (резервная копия)
    │   │   └── 📄 competency_matrix.csv  # Матрица компетенций (ученик × навык) из последней выгрузки
    │   ├── 📁 processed  # Обработанные и агрегированные данные рынка труда (после парсинга и нормализации)
    │   │   ├── 📄 competency_frequency.json  # Словарь {навык: частота_упоминаний} для быстрого доступа
    │   │   ├── 📄 competency_mapping.json  # Правила нормализации: {сырой_навык → каноническое_название}
    │   │   └── 📄 market_competencies.csv  # Таблица рыночных навыков с частотами и категориями (для Pandas)
    │   ├── 📁 raw  # Сырые данные, полученные напрямую от источников (API hh.ru)
    │   │   ├── 📄 competency_matrix.csv  # Исходная матрица компетенций учеников (может использоваться загрузчиком)
    │   │   └── 📄 hh_vacancies.json  # Полный дамп вакансий с HH.ru, включая поле `key_skills`
    │   ├── 📁 result  # Результаты сравнения и персональные рекомендации для каждого профиля ученика
    │   │   ├── 📁 base  # Результаты для базового ученика "base"
    │   │   │   ├── 📄 comparison_report_base.json  # Детальный отчёт сравнения: общие навыки, дефициты, метрики покрытия
    │   │   │   └── 📄 recommendations_base.json  # Индивидуальные рекомендации по развитию (курсы, технологии)
    │   │   ├── 📁 dc  # Результаты для ученика "dc" (Data Scientist?)
    │   │   │   ├── 📄 comparison_report_dc.json
    │   │   │   └── 📄 recommendations_dc.json
    │   │   └── 📁 top_dc  # Результаты для ученика "top_dc" (продвинутый Data Scientist)
    │   │       ├── 📄 comparison_report_top_dc.json
    │   │       └── 📄 recommendations_top_dc.json
    │   ├── 📁 students  # Входные JSON-файлы с перечнями навыков конкретных учеников
    │   │   ├── 📄 base_competency.json  # Список навыков ученика "base"
    │   │   ├── 📄 dc_competency.json  # Список навыков ученика "dc"
    │   │   ├── 📄 descriptiom_of_competency.txt  # Текстовое описание компетенций (возможно, инструкция)
    │   │   └── 📄 top_dc_competency.json  # Список навыков ученика "top_dc"
    │   └── 📄 it_skills.json  # Белый список / эталонный словарь IT-навыков для валидации и нормализации (костыль на доработке)
    ├── 📁 frontend  # Прототип веб-интерфейса для демонстрации результатов
    │   └── 📄 app.py  # Простое Flask / Streamlit приложение для визуализации отчётов и рекомендаций
    ├── 📁 notebooks  # Jupyter-ноутбуки для исследовательского анализа данных и прототипирования моделей
    │   ├── 📄 01_hh_analysis.ipynb  # Анализ сырых данных hh.ru: частоты навыков, распределения, первичные графики
    │   ├── 📄 02_competency_matching.ipynb  # Сопоставление профилей учеников с рынком, gap-анализ, визуализация дефицитов
    │   └── 📄 03_prediction_model.ipynb  # Эксперименты с ML-моделями для прогнозирования востребованности навыков
    ├── 📁 src  # Основной исходный код проекта (Python-пакет)
    │   ├── 📁 analyzers  # Модули сравнения профилей и анализа дефицитов компетенций
    │   │   ├── 📄 comparator.py  # Функции сравнения навыков ученика с рыночным эталоном (пересечения, расстояние, метрики)
    │   │   ├── 📄 embedding_comparator.py  # Сравнение навыков с использованием эмбеддингов (семантическая близость)
    │   │   ├── 📄 gap_analyzer.py  # Выявление и приоритизация недостающих навыков (gap-анализ), группировка по категориям
    │   │   ├── 📄 init.py  # Маркер пакета Python, может содержать импорты для удобства
    │   │   ├── 📄 profile_evaluator.py  # Комплексная оценка профиля ученика (уровень, полнота, соответствие рынку)
    │   │   ├── 📄 skill_filter.py  # Фильтрация и очистка списка навыков (удаление дубликатов, стоп-слов, нормализация)
    │   │   ├── 📄 skill_level_analyzer.py  # Определение уровня владения навыком (junior/middle/senior) по косвенным признакам
    │   │   └── 📄 trends.py  # Анализ временных трендов востребованности навыков (если есть исторические срезы данных)
    │   ├── 📁 loaders  # Загрузка и валидация входных данных (ученики)
    │   │   ├── 📄 init.py
    │   │   └── 📄 student_loader.py  # Чтение JSON-файлов учеников, валидация через Pydantic-модели
    │   ├── 📁 models  # Pydantic-модели для строгой типизации данных (валидация, сериализация)
    │   │   ├── 📄 comparison.py  # Модели результатов сравнения: дефициты, метрики схожести, структура отчёта
    │   │   ├── 📄 competency.py  # Модель рыночной компетенции (название, частота, категория, синонимы)
    │   │   ├── 📄 init.py
    │   │   ├── 📄 student.py  # Модель ученика (id, имя, список навыков, целевая профессия/роль)
    │   │   └── 📄 vacancy.py  # Модель вакансии (id, название, требуемые навыки, регион, зарплата)
    │   ├── 📁 parsing  # Модули взаимодействия с API hh.ru и обработки полученных данных
    │   │   ├── 📄 hh_api_async.py  # Асинхронный клиент для ускоренного сбора вакансий (asyncio / aiohttp)
    │   │   ├── 📄 hh_api.py  # Синхронный клиент API hh.ru: поиск вакансий, обработка пагинации, экспорт в JSON
    │   │   ├── 📄 init.py
    │   │   ├── 📄 skill_normalizer.py  # Приведение названий навыков к единому виду (лемматизация, lowercase, удаление спецсимволов)
    │   │   ├── 📄 skill_parser.py  # Извлечение ключевых навыков из текста вакансии / поля `key_skills`
    │   │   ├── 📄 skill_validator.py  # Проверка, является ли строка корректным IT-навыком (по белому списку / эвристикам)
    │   │   ├── 📄 utils.py  # Вспомогательные функции для парсеров: очистка текста, ротация User-Agent, sleep с jitter
    │   │   └── 📄 vacancy_parser.py  # Основной парсер вакансий: извлечение навыков, подсчёт частот, агрегация
    │   ├── 📁 predictors  # Модули прогнозирования и генерации рекомендаций
    │   │   ├── 📄 init.py
    │   │   ├── 📄 ml_recommendation_engine.py  # Рекомендательная система на основе ML (коллаборативная фильтрация, подобие профилей)
    │   │   ├── 📄 recommendation_engine.py  # Подбор образовательных курсов и материалов под выявленные дефициты
    │   │   └── 📄 skill_forecast.py  # Прогнозирование будущей востребованности навыков (временные ряды, регрессия)
    │   ├── 📁 visualization  # Построение графиков и диаграмм для отчётов
    │   │   ├── 📄 charts.py  # Функции отрисовки: столбчатые диаграммы, радарные графики, heatmap дефицитов
    │   │   └── 📄 init.py
    │   ├── 📄 config.py  # Конфигурационные параметры: пути к данным, ключи API, пороговые значения, категории
    │   ├── 📄 init.py  # Маркер корневого пакета `src`
    │   └── 📄 utils.py  # Общие утилиты: логирование, безопасное чтение/запись JSON, создание директорий
    ├── 📁 tests  # Модульные и интеграционные тесты для всех компонентов системы
    │   ├── 📁 analyzers  # Тесты для модулей сравнения и анализа
    │   │   ├── 📄 test_analyzers.py  # Общие тесты анализаторов
    │   │   ├── 📄 test_comparator.py  # Тесты функций сравнения списков навыков
    │   │   └── 📄 test_gap_analyzers.py  # Тесты выявления и приоритизации дефицитов
    │   ├── 📁 integration  # Интеграционные тесты (сквозная проверка пайплайна)
    │   │   └── 📄 test_full_pipeline.py  # Проверка полного цикла: загрузка → парсинг → сравнение → рекомендации
    │   ├── 📁 loaders  # Тесты загрузчиков данных
    │   │   └── 📄 test_loaders.py  # Тесты валидации JSON учеников, корректной работы student_loader
    │   ├── 📁 models  # Тесты Pydantic-моделей
    │   │   └── 📄 test_models.py  # Валидация схем данных (корректные/некорректные входы)
    │   ├── 📁 parsing  # Тесты парсеров и взаимодействия с API (с использованием моков)
    │   │   └── 📄 test_parsers.py  # Тесты нормализации навыков, парсинга ответа API, обработки ошибок
    │   ├── 📁 predictors  # Тесты рекомендательных движков и прогнозаторов
    │   │   ├── 📄 test_forecast.py  # Тесты моделей прогнозирования востребованности
    │   │   └── 📄 test_ml_recommendation.py  # Тесты ML-рекомендаций (обучение на фиктивных данных)
    │   ├── 📄 conftest.py  # Фикстуры pytest (например, моковые данные учеников и вакансий)
    │   └── 📄 init.py
    ├── 📄 main.py  # Точка входа CLI: запуск полного пайплайна (парсинг → сравнение → отчёты) с аргументами командной строки
    ├── 📄 README.md  # Описание проекта, инструкции по установке и использованию, примеры команд
    ├── 📄 requirements.txt  # Список зависимостей Python (requests, pandas, pydantic, matplotlib, pytest, etc.)
    └── 📄 user_manual.md  # Подробное руководство пользователя с примерами сценариев работы
```
