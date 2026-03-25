# tests/test_parsers.py
"""
Модульные тесты для парсеров вакансий.
Тестирует:
- VacancyParser: очистка, нормализация, извлечение навыков, агрегация.
- HeadHunterAPI: поиск вакансий и получение деталей (с моками).

Также содержит точку входа для ручного поиска вакансий из командной строки.
"""

import json
import sys
import argparse
import re
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest
import requests

# Добавляем корень проекта в sys.path для корректных импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.vacancy_parser import VacancyParser
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.utils import setup_logging
from src import config

# ----------------------------------------------------------------------
# Фикстуры
# ----------------------------------------------------------------------

@pytest.fixture
def sample_vacancies():
    """Возвращает список сырых вакансий (как из API hh.ru) для тестов."""
    return [
        {
            "id": "123456",
            "name": "Python Developer",
            "employer": {"name": "IT Company"},
            "snippet": {
                "requirement": "Знание Python, Django, опыт работы с PostgreSQL",
                "responsibility": "Разработка backend на Python"
            },
            "description": "Требуется Python разработчик. Ключевые навыки: Python, Django, REST API.",
            "key_skills": [
                {"name": "Python"},
                {"name": "Django"},
                {"name": "PostgreSQL"}
            ],
            "salary": {"from": 100000, "to": 150000, "currency": "RUR"},
            "area": {"id": "1", "name": "Москва"},
            "published_at": "2023-01-01T00:00:00+0300",
            "alternate_url": "https://hh.ru/vacancy/123456"
        },
        {
            "id": "789012",
            "name": "Data Scientist",
            "employer": {"name": "Data Lab"},
            "snippet": {
                "requirement": "Машинное обучение, Python, Pandas, SQL",
                "responsibility": "Анализ данных, построение моделей"
            },
            "description": "Ищем Data Scientist. Необходимо знание <highlighttext>Python</highlighttext>, <highlighttext>scikit-learn</highlighttext>.",
            "key_skills": [
                {"name": "Python"},
                {"name": "scikit-learn"},
                {"name": "SQL"}
            ],
            "salary": {"from": 120000, "to": 180000, "currency": "RUR"},
            "area": {"id": "1", "name": "Москва"},
            "published_at": "2023-01-02T00:00:00+0300",
            "alternate_url": "https://hh.ru/vacancy/789012"
        }
    ]

@pytest.fixture
def sample_vacancy_with_highlight(sample_vacancies):
    """Вакансия с тегами <highlighttext> для проверки очистки."""
    vac = sample_vacancies[1].copy()
    vac["description"] = "Требуется <highlighttext>Python</highlighttext> и <highlighttext>scikit-learn</highlighttext>"
    vac["snippet"]["requirement"] = "Знание <highlighttext>Python</highlighttext> и SQL"
    return vac

# ----------------------------------------------------------------------
# Тесты VacancyParser
# ----------------------------------------------------------------------

class TestVacancyParser:
    """Тестирование методов класса VacancyParser."""

    def test_clean_highlighttext(self):
        """Проверка удаления тегов <highlighttext>."""
        parser = VacancyParser()
        text = "Навыки: <highlighttext>Python</highlighttext> и <highlighttext>Django</highlighttext>"
        expected = "Навыки: Python и Django"
        assert parser.clean_highlighttext(text) == expected

        # Пустой текст
        assert parser.clean_highlighttext(None) == ""
        assert parser.clean_highlighttext("") == ""

        # Текст без тегов
        clean = "Обычный текст"
        assert parser.clean_highlighttext(clean) == clean

    def test_normalize_skill(self):
        """Проверка нормализации навыков (удаление мусора, синонимы)."""
        parser = VacancyParser()
        # Простая нормализация
        assert parser.normalize_skill("Python") == "python"
        assert parser.normalize_skill("Python 3") == "python3"
        assert parser.normalize_skill("JavaScript") == "js"  # синоним
        assert parser.normalize_skill("Машинное обучение") == "ml"  # синоним
        # Удаление префиксов
        assert parser.normalize_skill("опыт работы с Python") == "python"
        assert parser.normalize_skill("знание SQL") == "sql"
        # Удаление суффиксов
        assert parser.normalize_skill("Python плюсом") == "python"
        # Фильтрация символов
        assert parser.normalize_skill("C++") == "cpp"  # синоним
        assert parser.normalize_skill("Kubernetes") == "k8s"
        # Пустое значение
        assert parser.normalize_skill("") == ""
        assert parser.normalize_skill(None) == ""

    def test_extract_skills(self, sample_vacancies):
        """Извлечение навыков из поля key_skills."""
        parser = VacancyParser()
        skills = parser.extract_skills(sample_vacancies)
        # Ожидаем 3+3 = 6 навыков
        assert len(skills) == 6
        assert "Python" in skills
        assert "Django" in skills
        assert "scikit-learn" in skills

    def test_extract_skills_with_highlight(self, sample_vacancy_with_highlight):
        """Извлечение из key_skills с тегами (теги должны удаляться)."""
        parser = VacancyParser()
        # В key_skills нет тегов, но метод всё равно их чистит
        skills = parser.extract_skills([sample_vacancy_with_highlight])
        assert len(skills) == 3
        assert "Python" in skills
        assert "scikit-learn" in skills

    def test_extract_skills_from_text(self, sample_vacancies):
        """Извлечение навыков из текстовых полей (snippet, description)."""
        parser = VacancyParser()
        skills = parser.extract_skills_from_text(sample_vacancies)
        # Должны быть найдены Python, Django, PostgreSQL, SQL, scikit-learn, etc.
        assert len(skills) > 0
        # Проверяем наличие нормализованных названий (с учётом синонимов)
        # Реальные извлечённые навыки могут быть разными, проверяем ключевые
        skill_set = set(skills)
        # Ожидаем, что Python и SQL будут найдены
        assert "python" in skill_set or "Python" in skill_set
        # В тексте есть "SQL" – должен быть найден
        assert "sql" in skill_set or "SQL" in skill_set

    def test_count_skills(self):
        """Подсчёт частот с фильтрацией мусора."""
        parser = VacancyParser()
        skills_list = ["Python", "Python", "Java", "SQL", "английского языка", "опыт работы", "Python"]
        counts = parser.count_skills(skills_list)
        # Ожидаем: Python:3, Java:1, SQL:1; мусор отфильтрован
        assert counts == {"python": 3, "java": 1, "sql": 1}
        assert len(counts) == 3

    def test_aggregate_to_dataframe(self, sample_vacancies, sample_vacancy_with_highlight):
        """Преобразование вакансий в DataFrame с очисткой highlighttext."""
        parser = VacancyParser()
        df = parser.aggregate_to_dataframe(sample_vacancies + [sample_vacancy_with_highlight])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # Проверка очистки highlighttext
        # В sample_vacancy_with_highlight description содержит теги – они должны быть удалены
        desc = df.loc[df['id'] == '789012', 'description'].values[0]
        assert "<highlighttext>" not in desc
        # Проверяем, что ключевые навыки также очищены
        key_skills = df.loc[df['id'] == '789012', 'key_skills'].values[0]
        assert "<highlighttext>" not in key_skills
        # Проверяем основные колонки
        required_cols = ['id', 'name', 'employer_name', 'salary_from', 'salary_to', 'key_skills']
        for col in required_cols:
            assert col in df.columns

    def test_save_raw_vacancies(self, sample_vacancies, tmp_path):
        """Сохранение сырых вакансий в JSON."""
        # Временно подменяем config.DATA_RAW_DIR на временную директорию
        original_dir = config.DATA_RAW_DIR
        config.DATA_RAW_DIR = tmp_path
        parser = VacancyParser()
        filename = "test_raw.json"
        parser.save_raw_vacancies(sample_vacancies, filename)
        filepath = tmp_path / filename
        assert filepath.exists()
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert len(data) == len(sample_vacancies)
        config.DATA_RAW_DIR = original_dir

    def test_save_processed_frequencies(self, sample_vacancies, tmp_path, monkeypatch):
        """Сохранение частот с опциональной фильтрацией."""
        # Подменяем пути
        monkeypatch.setattr(config, 'DATA_PROCESSED_DIR', tmp_path)
        parser = VacancyParser()
        freqs = {"python": 10, "java": 5, "sql": 3}
        # Без фильтрации
        parser.save_processed_frequencies(freqs, filename="test_freq.json", apply_filter=False)
        filepath = tmp_path / "test_freq.json"
        assert filepath.exists()
        with open(filepath, 'r') as f:
            saved = json.load(f)
        assert saved == freqs
        # С фильтрацией – нужно замокать load_it_skills и filter_skills_by_whitelist
        # Здесь просто проверяем, что метод вызывается, фильтрацию протестируем отдельно
        # Для простоты оставим без фильтрации

# ----------------------------------------------------------------------
# Тесты HeadHunterAPI (с моками)
# ----------------------------------------------------------------------

class TestHeadHunterAPI:
    """Тестирование методов HeadHunterAPI с подменой сетевых запросов."""

    @patch('src.parsing.hh_api.requests.get')
    def test_search_vacancies_success(self, mock_get):
        """Успешный поиск вакансий."""
        # Мокаем ответ API
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {"id": "1", "name": "Python Developer"},
                {"id": "2", "name": "Data Scientist"}
            ],
            "pages": 1
        }
        mock_get.return_value = mock_response

        api = HeadHunterAPI()
        result = api.search_vacancies(text="Python", area=1, period_days=30, max_pages=1, per_page=100)

        assert len(result) == 2
        assert result[0]["id"] == "1"
        # Проверяем, что запрос был с правильными параметрами
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "text=Python" in kwargs["params"]["text"]
        assert kwargs["params"]["area"] == 1
        assert kwargs["params"]["per_page"] == 100

    @patch('src.parsing.hh_api.requests.get')
    def test_search_vacancies_http_error(self, mock_get):
        """Обработка ошибок HTTP."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not Found")
        mock_get.return_value = mock_response

        api = HeadHunterAPI()
        result = api.search_vacancies(text="Python", area=1)
        # Должен вернуть пустой список при ошибке
        assert result == []

    @patch('src.parsing.hh_api.requests.get')
    def test_get_vacancy_details_success(self, mock_get):
        """Получение детальной информации о вакансии."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "123",
            "name": "Python Developer",
            "key_skills": [{"name": "Python"}, {"name": "Django"}]
        }
        mock_get.return_value = mock_response

        api = HeadHunterAPI()
        details = api.get_vacancy_details("123")
        assert details is not None
        assert details["id"] == "123"
        assert len(details["key_skills"]) == 2

    @patch('src.parsing.hh_api.requests.get')
    def test_get_vacancy_details_failure(self, mock_get):
        """Неудачное получение деталей (ошибка сети)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        mock_get.return_value = mock_response

        api = HeadHunterAPI()
        details = api.get_vacancy_details("123")
        assert details is None

# ----------------------------------------------------------------------
# Интеграционный тест (опционально) на основе реального JSON-файла
# ----------------------------------------------------------------------

def test_parse_from_real_file(tmp_path):
    """Тест парсинга вакансий из сохранённого JSON-файла (например, из data/raw/hh_vacancies.json)."""
    # Создаём фиктивный JSON-файл с тестовыми данными
    test_data = [
        {
            "id": "1",
            "name": "Test",
            "key_skills": [{"name": "Python"}],
            "snippet": {"requirement": "Знание Python"},
            "description": "Требуется Python"
        }
    ]
    filepath = tmp_path / "test_vacancies.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)

    with open(filepath, 'r', encoding='utf-8') as f:
        vacancies = json.load(f)

    parser = VacancyParser()
    skills = parser.extract_skills(vacancies)
    assert len(skills) == 1
    assert skills[0] == "Python"

    # Проверяем извлечение из текста
    text_skills = parser.extract_skills_from_text(vacancies)
    # Должен быть найден "python" (после нормализации)
    assert "python" in set(text_skills) or "Python" in set(text_skills)


# ----------------------------------------------------------------------
# Точка входа для ручного поиска вакансий из командной строки
# ----------------------------------------------------------------------

def safe_print(text: str) -> None:
    """Безопасный вывод в консоль (игнорирует ошибки кодировки)."""
    try:
        print(text)
    except UnicodeEncodeError:
        clean_text = re.sub(r'[^\x00-\x7F]+', '', text)
        print(clean_text)

def run_search():
    """Запуск поиска вакансий с параметрами командной строки."""
    parser = argparse.ArgumentParser(
        description="Поиск вакансий на hh.ru и извлечение навыков (тестовый режим)."
    )
    parser.add_argument('--query', '-q', type=str, default="Data Scientist",
                        help="Поисковый запрос (например, 'Data Scientist')")
    parser.add_argument('--area-id', '-a', type=int, default=1,
                        help="ID региона (по умолчанию 1 - Москва)")
    parser.add_argument('--max-pages', '-p', type=int, default=20,
                        help="Максимальное количество страниц (по умолчанию 20)")
    parser.add_argument('--period', '-d', type=int, default=30,
                        help="Период поиска в днях (по умолчанию 30)")
    parser.add_argument('--show-vacancies', '-v', action='store_true',
                        help="Показать список найденных вакансий в консоли")
    parser.add_argument('--skip-details', '-s', action='store_true',
                        help="Пропустить загрузку деталей (только базовый поиск)")
    parser.add_argument('--excel', '-e', action='store_true',
                        help="Сохранить результаты в Excel")
    parser.add_argument('--no-filter', '-nf', action='store_true',
                        help="Отключить фильтрацию навыков по белому списку")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("test_search")

    logger.info("=" * 60)
    logger.info("ТЕСТОВЫЙ ПОИСК ВАКАНСИЙ С HH.RU")
    logger.info("=" * 60)
    logger.info(f"Запрос: '{args.query}'")
    logger.info(f"Регион ID: {args.area_id}, период: {args.period} дней, макс. страниц: {args.max_pages}")

    hh_api = HeadHunterAPI()
    vac_parser = VacancyParser()

    # 1. Поиск вакансий
    logger.info("Поиск вакансий...")
    basic_vacancies = hh_api.search_vacancies(
        text=args.query,
        area=args.area_id,
        period_days=args.period,
        max_pages=args.max_pages,
        per_page=100,
        search_fields=['name', 'company_name', 'description']
    )

    if not basic_vacancies:
        logger.error("Не найдено ни одной вакансии. Завершение.")
        return

    logger.info(f"Найдено вакансий: {len(basic_vacancies)}")

    # 2. Получение деталей (если требуется)
    if args.skip_details:
        logger.info("Пропускаем загрузку деталей.")
        vacancies_to_process = basic_vacancies
    else:
        logger.info("Загрузка детальной информации...")
        detailed_vacancies = []
        for idx, vac in enumerate(basic_vacancies, 1):
            if idx % 20 == 0:
                logger.info(f"Прогресс: {idx}/{len(basic_vacancies)}")
            details = hh_api.get_vacancy_details(vac['id'])
            if details:
                detailed_vacancies.append(details)
            time.sleep(config.REQUEST_DELAY)  # задержка между запросами
        logger.info(f"Загружено деталей: {len(detailed_vacancies)}")
        vacancies_to_process = detailed_vacancies

    # 3. Вывод списка (по желанию)
    if args.show_vacancies:
        vac_parser.print_vacancies_list(vacancies_to_process)

    # 4. Извлечение навыков
    logger.info("Извлечение навыков из вакансий...")
    all_skills = vac_parser.extract_skills(vacancies_to_process)
    if not all_skills:
        logger.info("key_skills не найдены, пробуем извлечь из текста...")
        all_skills = vac_parser.extract_skills_from_text(vacancies_to_process)

    if not all_skills:
        logger.error("Не удалось извлечь навыки ни из одного источника.")
        return

    logger.info(f"Извлечено сырых навыков: {len(all_skills)}")
    skill_freq = vac_parser.count_skills(all_skills)
    logger.info(f"Уникальных навыков после нормализации: {len(skill_freq)}")

    # Сохраняем частоты (с фильтрацией или без)
    vac_parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)

    # Топ-20
    top_skills = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    safe_print("\n" + "=" * 60)
    safe_print("ТОП-20 НАВЫКОВ ПО ЧАСТОТЕ УПОМИНАНИЙ")
    safe_print("=" * 60)
    for i, (skill, count) in enumerate(top_skills, 1):
        safe_print(f"{i:2}. {skill:<50} {count:>4}")

    # 5. Сохранение в Excel (по желанию)
    if args.excel:
        logger.info("Сохранение результатов в Excel...")
        df = vac_parser.aggregate_to_dataframe(vacancies_to_process)
        if not df.empty:
            filename = f"test_vacancies_{args.query}_{args.area_id}.xlsx".replace(' ', '_')
            vac_parser.save_to_excel(df, filename)
            logger.info(f"Excel сохранён: {filename}")

    logger.info("Тестовый поиск завершён.")

if __name__ == "__main__":
    # Если запущен напрямую, но без аргументов, можно показать справку
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']):
        # Если нет аргументов, показываем справку и выходим
        run_search.__defaults__ = (None,)  # чтобы парсер отработал с -h
        run_search()
    else:
        run_search()