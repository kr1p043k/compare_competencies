#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.hh_api import HeadHunterAPI
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.utils import setup_logging
from src import config
# Импортируем функцию для обновления профилей учеников
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.config import DATA_RAW_DIR, STUDENTS_DIR, LAST_UPLOADED_DIR

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
    parser.add_argument('--skip-details', '-s', action='store_true',
                        help="Пропустить загрузку деталей (только базовый поиск)")
    parser.add_argument('--excel', '-e', action='store_true',
                        help="Сохранить результаты в Excel")
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
    logger.info(f"Загружать детали: {not args.skip_details}")

    # ------------------------------------------------------------------
    # ЭТАП 0: ОБНОВЛЕНИЕ ПРОФИЛЕЙ УЧЕНИКОВ ИЗ CSV (если доступен)
    # ------------------------------------------------------------------
    try:
        logger.info("Проверка наличия CSV-матрицы компетенций...")
        # Используем путь по умолчанию: data/raw/competency_matrix.csv
        generate_profiles_from_csv()
        logger.info("✅ Профили учеников успешно обновлены из CSV.")
    except FileNotFoundError:
        logger.warning("⚠️ CSV-файл не найден, используются существующие профили в data/students/.")
    except Exception as e:
        logger.exception(f"❌ Ошибка при обновлении профилей: {e}")

    hh_api = HeadHunterAPI()
    parser = VacancyParser()

    # ------------------------------------------------------------------
    # ЭТАП 1: ПОИСК ВАКАНСИЙ
    # ------------------------------------------------------------------
    logger.info("ЭТАП 1: Поиск вакансий")

    # Параметры поиска из документации
    search_fields = ['name', 'company_name', 'description']

    basic_vacancies = hh_api.search_vacancies(
        text=args.query,
        area=args.area_id,
        period_days=args.period,
        max_pages=args.max_pages,
        per_page=100,
        search_fields=search_fields
    )

    if not basic_vacancies:
        logger.error("Не получено ни одной вакансии. Завершение работы.")
        return

    # ------------------------------------------------------------------
    # ЭТАП 2: ПОЛУЧЕНИЕ ДЕТАЛЬНОЙ ИНФОРМАЦИИ (если не пропущено)
    # ------------------------------------------------------------------
    if args.skip_details:
        logger.info("Пропускаем загрузку деталей по запросу")
        vacancies_to_process = basic_vacancies
        parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json")
    else:
        logger.info(f"ЭТАП 2: Загрузка детальной информации для {len(basic_vacancies)} вакансий")
        detailed_vacancies = []
        success_count = 0

        for idx, vac in enumerate(basic_vacancies, 1):
            vac_id = vac['id']
            logger.debug(f"Загрузка деталей вакансии {idx}/{len(basic_vacancies)}: {vac_id}")

            details = hh_api.get_vacancy_details(vac_id)

            if details:
                detailed_vacancies.append(details)
                success_count += 1
                # Проверяем наличие ключевых навыков для отладки
                if details.get('key_skills'):
                    logger.debug(f"✅ Вакансия {vac_id} содержит {len(details['key_skills'])} ключевых навыков")
            else:
                logger.warning(f"⚠️ Не удалось загрузить детали для вакансии {vac_id}, использую базовую версию")
                detailed_vacancies.append(vac)

            # Обязательная пауза между запросами
            time.sleep(config.REQUEST_DELAY)

        logger.info(f"Загружена детальная информация для {success_count}/{len(basic_vacancies)} вакансий")
        vacancies_to_process = detailed_vacancies

        # Сохраняем подробные данные
        parser.save_raw_vacancies(detailed_vacancies, filename="hh_vacancies_detailed.json")

    # ------------------------------------------------------------------
    # ЭТАП 3: ВЫВОД В КОНСОЛЬ (если запрошено)
    # ------------------------------------------------------------------
    if args.show_vacancies:
        parser.print_vacancies_list(vacancies_to_process)

    # ------------------------------------------------------------------
    # ЭТАП 4: ИЗВЛЕЧЕНИЕ И АНАЛИЗ НАВЫКОВ
    # ------------------------------------------------------------------
    logger.info("ЭТАП 4: Извлечение и анализ навыков")

    # Сначала пробуем извлечь из key_skills
    all_skills = parser.extract_skills(vacancies_to_process)

    # Если не нашли, пробуем из текста
    if not all_skills:
        logger.info("key_skills не найдены, пробуем извлечь из текста вакансий...")
        all_skills = parser.extract_skills_from_text(vacancies_to_process)

    if not all_skills:
        logger.warning("❌ Не удалось извлечь навыки ни из key_skills, ни из текста.")
    else:
        skill_frequencies = parser.count_skills(all_skills)
        parser.save_processed_frequencies(skill_frequencies)

        # Вывод топ-20 навыков
        top_skills = sorted(skill_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\n📊 Топ-20 наиболее востребованных навыков:")
        print("-" * 40)
        for i, (skill, count) in enumerate(top_skills, 1):
            print(f"{i:2}. {skill:30} {count:3} упоминаний")

    # ------------------------------------------------------------------
    # ЭТАП 5: СОХРАНЕНИЕ В EXCEL (если запрошено)
    # ------------------------------------------------------------------
    if args.excel:
        logger.info("ЭТАП 5: Сохранение результатов в Excel")
        df = parser.aggregate_to_dataframe(vacancies_to_process)
        if not df.empty:
            filename = f"vacancies_{args.query}_{args.area_id}.xlsx".replace(' ', '_')
            parser.save_to_excel(df, filename)

    logger.info("=" * 60)
    logger.info("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()