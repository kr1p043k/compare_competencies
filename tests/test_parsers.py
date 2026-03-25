# tests/test_parsers.py
"""
Модульные тесты для парсеров вакансий.
Интерактивная утилита для сбора вакансий с hh.ru.
При выборе пункта 11 используется список позиций + ограничение 500 вакансий.
"""

import json
import sys
import argparse
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import pytest
import requests
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.vacancy_parser import VacancyParser
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.utils import setup_logging
from src import config

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Фикстуры и тесты
# ----------------------------------------------------------------------

@pytest.fixture
def sample_vacancies():
    return [{"id": "123", "name": "Test Vacancy"}]


class TestVacancyParser:
    def test_clean_highlighttext(self):
        parser = VacancyParser()
        assert parser.clean_highlighttext("Python <highlighttext>test</highlighttext>") == "Python test"


class TestHeadHunterAPI:
    @patch('src.parsing.hh_api.requests.get')
    def test_search_vacancies_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [{"id": "1"}], "pages": 1}
        mock_get.return_value = mock_response
        api = HeadHunterAPI()
        result = api.search_vacancies(text="Python", area=1)
        assert len(result) == 1


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------

def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(re.sub(r'[^\x00-\x7F]+', '', text))


def input_int(prompt: str, default: int = 30, min_val: int = 1, max_val: int = 30) -> int:
    while True:
        val = input(prompt).strip()
        if not val:
            return default
        try:
            num = int(val)
            if min_val <= num <= max_val:
                return num
            print(f"Введите число от {min_val} до {max_val}")
        except ValueError:
            print("Введите целое число")


def input_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = " (y/n, по умолчанию y)" if default else " (y/n, по умолчанию n)"
    ans = input(prompt + default_text).strip().lower()
    if not ans:
        return default
    return ans in ('y', 'yes', 'да')


def select_from_list(items: List[str], prompt: str) -> str:
    print(prompt)
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    while True:
        try:
            idx = int(input("> ").strip())
            if 1 <= idx <= len(items):
                return items[idx-1]
        except:
            print("Некорректный ввод")


# ----------------------------------------------------------------------
# Интерактивная конфигурация
# ----------------------------------------------------------------------

def interactive_config() -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("ИНТЕРАКТИВНЫЙ СБОР ВАКАНСИЙ С HH.RU")
    print("=" * 90)

    mode_options = [
        "1. Data Scientist",
        "2. Python Developer",
        "3. Java Developer",
        "4. Frontend Developer",
        "5. Backend Developer",
        "6. DevOps Engineer",
        "7. Machine Learning Engineer",
        "8. QA Engineer",
        "9. Системный аналитик",
        "10. Другое (ввести свой запрос)",
        "11. Поиск по всему IT-сектору (industry=7)"
    ]

    selected_mode = select_from_list(mode_options, "\nВыберите вариант поиска:")

    if selected_mode == "11. Поиск по всему IT-сектору (industry=7)":
        print("\nРежим: Поиск по всему IT-сектору")
        positions = [
            "Data Scientist", "Data Analyst", "Machine Learning Engineer",
            "Python Developer", "Java Developer", "Frontend Developer",
            "Backend Developer", "Fullstack Developer", "DevOps Engineer",
            "QA Engineer", "Системный аналитик", "Product Manager",
            "Mobile Developer", "iOS Developer", "Android Developer"
        ]
        print("Позиции для поиска:")
        for p in positions:
            print(f"  • {p}")
        queries = positions
        industry = 7
        is_it_sector = True
        query = "IT_Sector_Multiple"
    elif selected_mode == "10. Другое (ввести свой запрос)":
        query = input("\nВведите поисковый запрос: ").strip() or "Data Scientist"
        industry = None
        is_it_sector = False
        queries = [query]
    else:
        query = selected_mode.split('. ', 1)[1]
        industry = None
        is_it_sector = False
        queries = [query]

    # Регионы
    region_options = [
        ("Москва", 1), ("Санкт-Петербург", 2), ("Екатеринбург", 3),
        ("Новосибирск", 4), ("Казань", 88), ("Нижний Новгород", 66),
        ("Ростов-на-Дону", 76), ("Вся Россия", 0)
    ]
    region_names = [f"{name} (ID {rid})" for name, rid in region_options]
    print("\nВыберите регионы (можно несколько, введите номера через пробел):")
    try:
        indices = list(map(int, input("> ").split()))
        selected_regions = [region_names[i-1] for i in indices if 1 <= i <= len(region_names)]
    except:
        selected_regions = [region_names[0]]

    area_ids = [int(re.search(r'ID (\d+)', s).group(1)) for s in selected_regions if re.search(r'ID (\d+)', s)]
    if not area_ids:
        area_ids = [1]

    # Параметры
    if is_it_sector:
        period = 30
        max_pages = 50
        skip_details = False
        show_list = False
        print("\nОграничение: 500 вакансий на одну позицию")
        apply_filter = input_yes_no("Применять фильтрацию по белому списку?", default=False)
    else:
        period = input_int("\nПериод поиска в днях (по умолчанию 30): ", default=30)
        max_pages = input_int("Максимальное количество страниц (по умолчанию 20): ", default=20, max_val=20)
        skip_details = not input_yes_no("Загружать полную информацию по каждой вакансии?", default=True)
        show_list = input_yes_no("Показывать список найденных вакансий?", default=False)
        apply_filter = input_yes_no("Применять фильтрацию по белому списку?", default=True)

    save_excel = input_yes_no("Сохранить результаты в Excel?", default=True)

    return {
        "query": query,
        "queries": queries,
        "area_ids": area_ids,
        "industry": industry,
        "period": period,
        "max_pages": max_pages,
        "skip_details": skip_details,
        "show_vacancies": show_list,
        "excel": save_excel,
        "no_filter": not apply_filter,
        "is_it_sector": is_it_sector
    }


# ----------------------------------------------------------------------
# Сбор вакансий
# ----------------------------------------------------------------------

def collect_vacancies(
    hh_api: HeadHunterAPI,
    queries: List[str],
    area_ids: List[int],
    period_days: int,
    max_pages: int,
    industry: Optional[int] = None,
    max_vacancies_per_query: int = 1000
) -> List[Dict[str, Any]]:
    all_vacancies = []
    seen_ids: set[str] = set()
    logger = logging.getLogger("collector")

    for query in queries:
        query_vacancies = []
        for area_id in area_ids:
            logger.info(f"Поиск: '{query}', регион ID {area_id}")
            vacs = hh_api.search_vacancies(
                text=query,
                area=area_id,
                period_days=period_days,
                max_pages=max_pages,
                per_page=100,
                industry=industry
            )
            for vac in vacs:
                vid = vac.get('id')
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    query_vacancies.append(vac)
                    if len(query_vacancies) >= max_vacancies_per_query:
                        break
            if len(query_vacancies) >= max_vacancies_per_query:
                break
            time.sleep(config.REQUEST_DELAY)

        all_vacancies.extend(query_vacancies[:max_vacancies_per_query])
        logger.info(f"Для запроса '{query}' собрано {len(query_vacancies[:max_vacancies_per_query])} вакансий")

    logger.info(f"Всего собрано уникальных вакансий: {len(all_vacancies)}")
    return all_vacancies


# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------

def run_search(args: argparse.Namespace = None, interactive: bool = False):
    if interactive or args is None or len(sys.argv) == 1:
        params = interactive_config()
        args = argparse.Namespace(**params)

    setup_logging()
    logger = logging.getLogger("hh_collector")

    logger.info("=" * 90)
    logger.info("ЗАПУСК СБОРА ВАКАНСИЙ С HH.RU")
    logger.info("=" * 90)

    queries = getattr(args, 'queries', [args.query])
    area_ids = args.area_ids
    sector_suffix = "_it_sector" if getattr(args, 'is_it_sector', False) else ""
    query_part = re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁ_-]', '_', args.query)[:40]

    max_vacancies_limit = 500 if getattr(args, 'is_it_sector', False) else 1000

    original_raw = config.DATA_RAW_DIR
    original_proc = config.DATA_PROCESSED_DIR
    config.DATA_RAW_DIR = config.DATA_PROCESSED_DIR = TEST_OUTPUT_DIR

    try:
        hh_api = HeadHunterAPI()
        parser = VacancyParser()

        logger.info(f"Запрос(ы): {queries} | Регионы: {area_ids}")
        if getattr(args, 'is_it_sector', False):
            logger.info(f"IT-сектор: лимит {max_vacancies_limit} вакансий на позицию")

        basic_vacancies = collect_vacancies(
            hh_api=hh_api,
            queries=queries,
            area_ids=area_ids,
            period_days=args.period,
            max_pages=args.max_pages,
            industry=getattr(args, 'industry', None),
            max_vacancies_per_query=max_vacancies_limit
        )

        if not basic_vacancies:
            logger.error("Вакансий не найдено.")
            return

        vacancies_to_process = basic_vacancies
        if not args.skip_details:
            logger.info("Загрузка детальной информации...")
            detailed = []
            for i, vac in enumerate(basic_vacancies, 1):
                if i % 15 == 0:
                    logger.info(f"Прогресс деталей: {i}/{len(basic_vacancies)}")
                det = hh_api.get_vacancy_details(vac['id'])
                if det:
                    detailed.append(det)
                time.sleep(config.REQUEST_DELAY)
            vacancies_to_process = detailed

        if args.show_vacancies:
            parser.print_vacancies_list(vacancies_to_process)

        all_skills = parser.extract_skills(vacancies_to_process) or parser.extract_skills_from_text(vacancies_to_process)
        skill_freq = parser.count_skills(all_skills)

        raw_file = f"raw_vacancies{sector_suffix}_{query_part}.json"
        parser.save_raw_vacancies(vacancies_to_process, raw_file)

        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_raw.json", apply_filter=False)
        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_filtered.json", apply_filter=not args.no_filter)

        top_skills = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\nТОП-20 НАВЫКОВ:")
        for i, (skill, count) in enumerate(top_skills, 1):
            print(f"{i:2}. {skill:<45} {count:>5}")

        if args.excel:
            df = parser.aggregate_to_dataframe(vacancies_to_process)
            excel_name = f"vacancies{sector_suffix}_{query_part}.xlsx"
            parser.save_to_excel(df, excel_name)
            logger.info(f"Excel сохранён: {excel_name}")

        logger.info(f"Все файлы сохранены в: {TEST_OUTPUT_DIR}")

    finally:
        config.DATA_RAW_DIR = original_raw
        config.DATA_PROCESSED_DIR = original_proc


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_search(interactive=True)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--interactive', action='store_true')
        args = parser.parse_args()
        run_search(args, interactive=args.interactive)