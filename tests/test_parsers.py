# tests/test_parsers.py
"""
Модульные тесты для парсеров вакансий.
Интерактивная утилита для сбора вакансий с hh.ru.
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
from src.parsing.utils import (
    setup_logging,
    collect_vacancies_multiple,
    interactive_config,
    safe_print,
    extract_and_count_skills,
    print_top_skills
)
from src import config

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Фикстуры и тесты (остаются без изменений)
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

        basic_vacancies = collect_vacancies_multiple(
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

        skill_freq = extract_and_count_skills(vacancies_to_process, parser)
        if not skill_freq:
            return

        raw_file = f"raw_vacancies{sector_suffix}_{query_part}.json"
        parser.save_raw_vacancies(vacancies_to_process, raw_file)

        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_raw.json", apply_filter=False)
        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_filtered.json", apply_filter=not args.no_filter)

        print_top_skills(skill_freq)

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