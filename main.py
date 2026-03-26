#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
from pathlib import Path
import json

# Устанавливаем кодировку для вывода в консоль Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.hh_api import HeadHunterAPI
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.utils import (
    setup_logging,
    collect_vacancies_multiple,
    load_queries_from_file,
    interactive_config,
    safe_print,
    extract_and_count_skills,
    map_to_competencies,
    print_top_skills,
    print_top_competencies
)
from src import config
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.utils import load_competency_mapping


def parse_arguments():
    parser = argparse.ArgumentParser(description="Сбор и анализ вакансий с hh.ru")
    # Базовые параметры
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
    # Расширенные параметры
    parser.add_argument('--queries-file', '-qf', type=str,
                        help="Файл со списком запросов (по одному на строку)")
    parser.add_argument('--regions', '-r', type=str,
                        help="Список ID регионов через запятую (например, '1,2,3')")
    parser.add_argument('--industry', '-i', type=int,
                        help="Код профессиональной области (например, 7 - IT)")
    parser.add_argument('--interactive', action='store_true',
                        help="Запустить в интерактивном режиме (с меню)")
    parser.add_argument('--max-vacancies-per-query', type=int, default=1000,
                        help="Максимальное количество вакансий на один запрос (по умолчанию 1000)")
    parser.add_argument('--it-sector', action='store_true',
                        help="Поиск по всему IT-сектору (industry=7) с предопределённым списком позиций")
    return parser.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("СБОР ВАКАНСИЙ С HH.RU")
    logger.info("=" * 60)

    # Определяем, используем ли расширенный режим
    use_multiple = (
        args.interactive or
        args.queries_file is not None or
        args.regions is not None or
        args.industry is not None or
        args.it_sector
    )

    # ------------------------------------------------------------------
    # Расширенный режим (множественные запросы/регионы/интерактив)
    # ------------------------------------------------------------------
    if use_multiple:
        if args.interactive:
            params = interactive_config()
            # Подменяем аргументы на полученные из интерактива
            args.query = params["query"]
            args.queries = params["queries"]
            args.area_ids = params["area_ids"]
            args.industry = params["industry"]
            args.period = params["period"]
            args.max_pages = params["max_pages"]
            args.skip_details = params["skip_details"]
            args.show_vacancies = params["show_vacancies"]
            args.excel = params["excel"]
            args.no_filter = params["no_filter"]
            args.max_vacancies_per_query = params["max_vacancies_per_query"]
        else:
            # Формируем список запросов
            if args.it_sector:
                args.queries = [
                    "Data Scientist", "Data Analyst", "Machine Learning Engineer",
                    "Python Developer", "Java Developer", "Frontend Developer",
                    "Backend Developer", "Fullstack Developer", "DevOps Engineer",
                    "QA Engineer", "Системный аналитик", "Product Manager",
                    "Mobile Developer", "iOS Developer", "Android Developer"
                ]
                args.industry = 7
                args.max_vacancies_per_query = 500
                logger.info("Режим: поиск по всему IT-сектору (предопределённые позиции)")
            elif args.queries_file:
                args.queries = load_queries_from_file(Path(args.queries_file))
                if not args.queries:
                    logger.error("Не удалось загрузить запросы из файла.")
                    return
            else:
                args.queries = [args.query]

            # Формируем список регионов
            if args.regions:
                args.area_ids = [int(x.strip()) for x in args.regions.split(',')]
            else:
                args.area_ids = [args.area_id]

            # Если не указан industry, но указан it-sector, он уже установлен выше
            if args.it_sector and args.industry is None:
                args.industry = 7

        logger.info(f"Режим: расширенный поиск")
        logger.info(f"Запросы: {args.queries}")
        logger.info(f"Регионы: {args.area_ids}")
        logger.info(f"Период: {args.period} дней, макс. страниц: {args.max_pages}")
        if args.industry:
            logger.info(f"Профобласть: {args.industry}")
        logger.info(f"Лимит вакансий на запрос: {args.max_vacancies_per_query}")

        hh_api = HeadHunterAPI()
        parser = VacancyParser()

        logger.info("Сбор вакансий...")
        basic_vacancies = collect_vacancies_multiple(
            hh_api=hh_api,
            queries=args.queries,
            area_ids=args.area_ids,
            period_days=args.period,
            max_pages=args.max_pages,
            industry=args.industry,
            max_vacancies_per_query=args.max_vacancies_per_query
        )

        if not basic_vacancies:
            logger.error("Не найдено ни одной вакансии.")
            return

        logger.info(f"Собрано вакансий: {len(basic_vacancies)}")

        # Детальная информация
        if args.skip_details:
            logger.info("Пропускаем загрузку деталей.")
            vacancies_to_process = basic_vacancies
        else:
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
            logger.info(f"Загружено деталей: {len(detailed)}")

        # Вывод списка
        if args.show_vacancies:
            parser.print_vacancies_list(vacancies_to_process)

        # Извлечение и подсчёт навыков
        skill_freq = extract_and_count_skills(vacancies_to_process, parser)
        if not skill_freq:
            return

        # Сохранение частот
        parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)

        # Вывод топ-20 навыков
        print_top_skills(skill_freq)

        # Преобразование в компетенции
        try:
            mapping = load_competency_mapping()
            if mapping:
                logger.info("Преобразование рыночных навыков в учебные компетенции...")
                comp_counter = map_to_competencies(skill_freq, mapping)
                if comp_counter:
                    comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency_mapped.json"
                    with open(comp_freq_path, 'w', encoding='utf-8') as f:
                        json.dump(dict(comp_counter.most_common()), f, ensure_ascii=False, indent=2)
                    logger.info(f"✅ Частоты компетенций сохранены в {comp_freq_path}")
                    print_top_competencies(comp_counter)
                else:
                    print("\n⚠️ Нет совпадений между рыночными навыками и учебными компетенциями")
            else:
                logger.warning("Маппинг не загружен, пропущено преобразование в компетенции.")
        except Exception as e:
            logger.exception(f"Ошибка при преобразовании компетенций: {e}")

        # Сохранение в Excel
        if args.excel:
            logger.info("Сохранение результатов в Excel...")
            df = parser.aggregate_to_dataframe(vacancies_to_process)
            if not df.empty:
                if len(args.queries) == 1:
                    query_part = args.queries[0].replace(' ', '_')
                else:
                    query_part = "multiple_queries"
                if args.it_sector:
                    query_part = "it_sector"
                regions_part = '_'.join(map(str, args.area_ids))
                filename = f"vacancies_{query_part}_{regions_part}.xlsx"
                parser.save_to_excel(df, filename)
                logger.info(f"✅ Excel файл сохранён: {filename}")

    # ------------------------------------------------------------------
    # Стандартный режим (один запрос, один регион)
    # ------------------------------------------------------------------
    else:
        logger.info(f"Запрос: '{args.query}'")
        logger.info(f"Регион ID: {args.area_id}, период: {args.period} дней, макс. страниц: {args.max_pages}")
        logger.info(f"Загружать детали: {not args.skip_details}")
        logger.info(f"Фильтрация навыков: {'ВЫКЛЮЧЕНА' if args.no_filter else 'ВКЛЮЧЕНА'}")

        # Загрузка профилей учеников (как было)
        try:
            logger.info("Проверка наличия CSV-матрицы компетенций...")
            csv_path = config.DATA_RAW_DIR / "competency_matrix.csv"
            if not csv_path.exists():
                csv_path = config.DATA_DIR / "last_uploaded" / "competency_matrix.csv"
            if csv_path.exists():
                profiles = generate_profiles_from_csv(csv_path)
                logger.info(f"✅ Профили учеников успешно обновлены из CSV: {len(profiles)} профилей")
            else:
                logger.warning("⚠️ CSV-файл не найден, используются существующие профили.")
        except Exception as e:
            logger.exception(f"❌ Ошибка при обновлении профилей: {e}")

        hh_api = HeadHunterAPI()
        parser = VacancyParser()

        logger.info("ЭТАП 1: Поиск вакансий")
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

        logger.info(f"Найдено {len(basic_vacancies)} вакансий")

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
                if idx % 20 == 0:
                    logger.info(f"Прогресс: {idx}/{len(basic_vacancies)} вакансий")
                details = hh_api.get_vacancy_details(vac_id)
                if details:
                    detailed_vacancies.append(details)
                    success_count += 1
                else:
                    logger.warning(f"⚠️ Не удалось загрузить детали для вакансии {vac_id}")
                time.sleep(config.REQUEST_DELAY)
            logger.info(f"Загружена детальная информация для {success_count}/{len(basic_vacancies)} вакансий")
            vacancies_to_process = detailed_vacancies
            parser.save_raw_vacancies(detailed_vacancies, filename="hh_vacancies_detailed.json")

        if args.show_vacancies:
            parser.print_vacancies_list(vacancies_to_process)

        # Извлечение и подсчёт навыков
        skill_freq = extract_and_count_skills(vacancies_to_process, parser)
        if not skill_freq:
            return

        parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)

        print_top_skills(skill_freq)

        # Преобразование в компетенции
        logger.info("Преобразование рыночных навыков в учебные компетенции...")
        try:
            mapping = load_competency_mapping()
            if mapping:
                comp_counter = map_to_competencies(skill_freq, mapping)
                if comp_counter:
                    comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency_mapped.json"
                    with open(comp_freq_path, 'w', encoding='utf-8') as f:
                        json.dump(dict(comp_counter.most_common()), f, ensure_ascii=False, indent=2)
                    logger.info(f"✅ Частоты компетенций сохранены в {comp_freq_path}")
                    print_top_competencies(comp_counter)
                else:
                    print("\n⚠️ Нет совпадений между рыночными навыками и учебными компетенциями")
            else:
                logger.warning("Маппинг не загружен, пропущено преобразование в компетенции.")
                print("\n⚠️ Файл competency_mapping.json не найден или пуст")
        except Exception as e:
            logger.exception(f"Ошибка при преобразовании компетенций: {e}")
            print(f"\n❌ Ошибка при преобразовании компетенций: {e}")

        if args.excel:
            logger.info("ЭТАП 5: Сохранение результатов в Excel")
            df = parser.aggregate_to_dataframe(vacancies_to_process)
            if not df.empty:
                filename = f"vacancies_{args.query}_{args.area_id}.xlsx".replace(' ', '_')
                parser.save_to_excel(df, filename)
                logger.info(f"✅ Excel файл сохранен: {filename}")

    logger.info("=" * 60)
    logger.info("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()