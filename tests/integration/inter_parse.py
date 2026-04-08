import logging
import json
import sys
import argparse
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pytest
import requests
from unittest.mock import Mock, patch
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.skill_validator import SkillValidator, ValidationReason
from src.parsing.skill_normalizer import SkillNormalizer 
from src.parsing.utils import (
    setup_logging,
    collect_vacancies_multiple,
    interactive_config,
    safe_print,
    extract_and_count_skills,
    print_top_skills
)
from src import config
logger = logging.getLogger(__name__)

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

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

        # Загружаем детальную информацию
        logger.info(f"Загружаем детальную информацию для {len(basic_vacancies)} вакансий...")
        vacancies_to_process = []
        for i, vac in enumerate(basic_vacancies, 1):
            if i % 50 == 0:
                logger.info(f"Прогресс деталей: {i}/{len(basic_vacancies)}")
            det = hh_api.get_vacancy_details(vac['id'])
            if det:
                vacancies_to_process.append(det)
            time.sleep(config.REQUEST_DELAY)

        if not vacancies_to_process:
            logger.error("Не удалось загрузить ни одной детальной вакансии")
            return

        if args.show_vacancies:
            parser.print_vacancies_list(vacancies_to_process)

        # === ИСПРАВЛЕННЫЙ ВЫЗОВ ===
        result = extract_and_count_skills(vacancies_to_process, parser)
        skill_freq = result["frequencies"]          # ← только частоты
        tfidf_weights = result.get("tfidf_weights", {})

        if not skill_freq:
            logger.error("Не удалось извлечь навыки.")
            return

        # Сохранение
        raw_file = f"raw_vacancies{sector_suffix}_{query_part}.json"
        parser.save_raw_vacancies(vacancies_to_process, raw_file)

        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_raw.json", apply_filter=False)
        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_filtered.json", apply_filter=not args.no_filter)

        # Вывод
        print_top_skills(skill_freq)

        # Новый вывод TF-IDF весов
        if tfidf_weights:
            print("\n" + "=" * 60)
            print("ТОП-15 НАВЫКОВ ПО TF-IDF ВЕСУ")
            print("=" * 60)
            top_weights = sorted(tfidf_weights.items(), key=lambda x: x[1], reverse=True)[:15]
            for i, (skill, weight) in enumerate(top_weights, 1):
                print(f"{i:2}. {skill:<40} {weight:.4f}")

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