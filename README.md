# compare_competencies
Сравнение компетенций учебной программы и с job-сайтами
Ориентировочное древо проекта:
competency-comparison/
│
├── data/                              # Хранение всех данных
│   ├── raw/                           # Сырые данные с hh.ru (ответы API, html)
│   │   └── hh_vacancies.json
│   ├── processed/                      # Очищенные и нормализованные данные
│   │   ├── market_competencies.csv
│   │   └── competency_frequency.json
│   ├── students/                       # JSON-файлы с компетенциями учеников
│   │   ├── student_1.json
│   │   └── student_2.json
│   └── results/                         # Результаты сравнения и прогнозы
│       ├── comparison_report_student_1.json
│       └── recommendations_student_1.json
│
├── notebooks/                          # Jupyter notebooks для экспериментов и визуализации
│   ├── 01_hh_analysis.ipynb
│   ├── 02_competency_matching.ipynb
│   └── 03_prediction_model.ipynb
│
├── src/                                 # Исходный код
│   ├── __init__.py
│   │
│   ├── parsers/                          # Модули для сбора данных с hh.ru
│   │   ├── __init__.py
│   │   ├── hh_api.py                     # Работа с API HeadHunter (получение вакансий, навыков)
│   │   ├── vacancy_parser.py              # Извлечение ключевых навыков из описаний
│   │   └── utils.py                       # Вспомогательные функции (очистка, нормализация)
│   │
│   ├── models/                            # Модели данных (Pydantic / dataclasses)
│   │   ├── __init__.py
│   │   ├── student.py                      # Схема данных ученика
│   │   ├── competency.py                   # Схема компетенции
│   │   └── vacancy.py                       # Схема вакансии
│   │
│   ├── loaders/                            # Загрузка данных из JSON и других источников
│   │   ├── __init__.py
│   │   └── student_loader.py                # Чтение JSON учеников
│   │
│   ├── analyzers/                          # Логика сравнения и анализа
│   │   ├── __init__.py
│   │   ├── comparator.py                    # Сравнение компетенций ученика с рыночными
│   │   ├── gap_analyzer.py                   # Определение дефицитов (gap-анализ)
│   │   └── trends.py                         # Анализ востребованности навыков (частоты)
│   │
│   ├── predictors/                          # Модули прогнозирования и рекомендаций
│   │   ├── __init__.py
│   │   ├── recommendation_engine.py          # Формирование рекомендаций по развитию
│   │   └── skill_forecast.py                  # Предсказание будущих трендов (если используется ML)
│   │
│   ├── visualization/                       # Построение графиков и отчётов
│   │   ├── __init__.py
│   │   └── charts.py                          # Визуализация сравнения (например, radar chart)
│   │
│   ├── config.py                             # Настройки проекта (пути, параметры API, ключи)
│   └── utils.py                               # Общие утилиты (логирование, работа с файлами)
│
├── tests/                                 # Модульные тесты
│   ├── __init__.py
│   ├── test_parsers.py
│   ├── test_comparator.py
│   └── test_loaders.py
│
├── requirements.txt                       # Зависимости Python
├── README.md                              # Описание проекта, установка, запуск
├── .gitignore                             # Исключения для git
└── main.py                                # Точка входа для запуска полного пайплайна