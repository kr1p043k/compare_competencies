#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.hh_api import HeadHunterAPI
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.utils import setup_logging
from src import config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Сбор и анализ вакансий с hh.ru по региону")
    parser.add_argument('--query', '-q', type=str, default="аналитик данных",
                        help="Поисковый запрос (например, 'аналитик данных')")
    parser.add_argument('--area-id', '-a', type=int, default=config.DEFAULT_AREA,
                        help=f"ID региона (по умолчанию {config.DEFAULT_AREA} - Ростовская область)")
    parser.add_argument('--max-pages', '-p', type=int, default=config.DEFAULT_MAX_PAGES,
                        help=f"Максимальное количество страниц (по умолчанию {config.DEFAULT_MAX_PAGES})")
    parser.add_argument('--period', '-d', type=int, default=config.DEFAULT_PERIOD_DAYS,
                        help=f"Период поиска в днях (по умолчанию {config.DEFAULT_PERIOD_DAYS})")
    parser.add_argument('--show-vacancies', '-v', action='store_true',
                        help="Показать список найденных вакансий в консоли")
    return parser.parse_args()

def main():
    setup_logging()
    logger = logging.getLogger("main")

    args = parse_arguments()
    logger.info("=" * 60)
    logger.info("СБОР ВАКАНСИЙ С HH.RU ПО РЕГИОНУ")
    logger.info("=" * 60)
    logger.info(f"Запрос: '{args.query}'")
    logger.info(f"Регион ID: {args.area_id}, период: {args.period} дней, макс. страниц: {args.max_pages}")

    hh_api = HeadHunterAPI()
    parser = VacancyParser()

    # ------------------------------------------------------------------
    # ЭТАП 1: ПОИСК ВАКАНСИЙ (теперь с поиском по названию, компании и описанию)
    # ------------------------------------------------------------------
    logger.info("ЭТАП 1: Поиск вакансий")
    vacancies = hh_api.search_vacancies(
        text=args.query,
        area=args.area_id,
        period_days=args.period,
        max_pages=args.max_pages,
        search_fields=['name', 'company_name', 'description']  # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ!
    )

    if not vacancies:
        logger.error("Не получено ни одной вакансии. Завершение работы.")
        return

    # Сохраняем сырые данные
    parser.save_raw_vacancies(vacancies)

    # Если запрошен вывод вакансий — показываем
    if args.show_vacancies:
        parser.print_vacancies_list(vacancies)

    # ------------------------------------------------------------------
    # ЭТАП 2: ИЗВЛЕЧЕНИЕ И АНАЛИЗ НАВЫКОВ
    # ------------------------------------------------------------------
    logger.info("ЭТАП 2: Извлечение и анализ навыков")
    all_skills = parser.extract_skills(vacancies)

    if not all_skills:
        logger.warning("Навыки (key_skills) не найдены в вакансиях.")
        # Здесь можно добавить альтернативный парсинг из текста, но пока просто предупреждение
    else:
        skill_frequencies = parser.count_skills(all_skills)
        parser.save_processed_frequencies(skill_frequencies)

        # Вывод топ-10
        top_skills = sorted(skill_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n📊 Топ-10 навыков (из key_skills):")
        for i, (skill, count) in enumerate(top_skills, 1):
            print(f"  {i:2}. {skill}: {count}")

    logger.info("Пайплайн сбора и анализа завершён.")

if __name__ == "__main__":
    main()