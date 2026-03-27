#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
from pathlib import Path
import json

# Windows UTF-8 fix
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
    print_top_skills,
    print_top_competencies,
    extract_and_count_skills,
    map_to_competencies
)
from src import config
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.utils import load_competency_mapping

# === Gap-анализ и рекомендации ===
from src.models.student import StudentProfile
from src.analyzers.gap_analyzer import GapAnalyzer
from src.predictors.recommendation_engine import RecommendationEngine


def parse_arguments():
    parser = argparse.ArgumentParser(description="Полный пайплайн: сбор вакансий + gap-анализ + рекомендации")
    
    parser.add_argument('--query', '-q', type=str, default="Frontend Developer")
    parser.add_argument('--area-id', '-a', type=int, default=76)
    parser.add_argument('--max-pages', '-p', type=int, default=20)
    parser.add_argument('--period', '-d', type=int, default=30)
    parser.add_argument('--show-vacancies', '-v', action='store_true')
    parser.add_argument('--skip-details', '-s', action='store_true')
    parser.add_argument('--excel', '-e', action='store_true')
    parser.add_argument('--no-filter', '-nf', action='store_true')
    parser.add_argument('--queries-file', '-qf', type=str)
    parser.add_argument('--regions', '-r', type=str)
    parser.add_argument('--industry', '-i', type=int)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--max-vacancies-per-query', type=int, default=1000)
    parser.add_argument('--it-sector', action='store_true')
    
    parser.add_argument('--run-gap-analysis', action='store_true', default=True,
                        help="Выполнить gap-анализ и генерацию рекомендаций (по умолчанию True)")

    return parser.parse_args()


def load_student_competencies(profile_name: str):
    """Загружает компетенции студента"""
    path = config.DATA_DIR / "students" / f"{profile_name}_competency.json"
    if not path.exists():
        path = config.DATA_DIR / "students" / f"{profile_name}.json"
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data.get("компетенции") or data.get("навыки") or data.get("codes") or []
    except Exception as e:
        logging.warning(f"Не удалось загрузить профиль {profile_name}: {e}")
        return []


def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_arguments()

    logger.info("=" * 85)
    logger.info("ПОЛНЫЙ ПАЙПЛАЙН: СБОР ВАКАНСИЙ + GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ")
    logger.info("=" * 85)

    use_multiple = (
        args.interactive or 
        args.queries_file is not None or 
        args.regions is not None or 
        args.industry is not None or 
        args.it_sector
    )

    vacancies_to_process = []
    skill_freq = {}

    # ====================== 1. СБОР ВАКАНСИЙ ======================
    if use_multiple:
        if args.interactive:
            params = interactive_config()
            args.query = params.get("query", args.query)
            args.queries = params.get("queries", [args.query])
            args.area_ids = params.get("area_ids", [args.area_id])
            args.industry = params.get("industry")
            args.period = params.get("period", args.period)
            args.max_pages = params.get("max_pages", args.max_pages)
            args.skip_details = params.get("skip_details", args.skip_details)
            args.show_vacancies = params.get("show_vacancies", args.show_vacancies)
            args.excel = params.get("excel", args.excel)
            args.no_filter = params.get("no_filter", args.no_filter)
            args.max_vacancies_per_query = params.get("max_vacancies_per_query", args.max_vacancies_per_query)
        else:
            if args.it_sector:
                args.queries = [
                    "Data Scientist", "Data Analyst", "Machine Learning Engineer",
                    "Python Developer", "Java Developer", "Frontend Developer",
                    "Backend Developer", "Fullstack Developer", "DevOps Engineer",
                    "QA Engineer", "Системный аналитик"
                ]
                args.industry = 7
                args.max_vacancies_per_query = 500
                logger.info("Режим: поиск по всему IT-сектору")

            elif args.queries_file:
                args.queries = load_queries_from_file(Path(args.queries_file))
            else:
                args.queries = [args.query]

            if args.regions:
                args.area_ids = [int(x.strip()) for x in args.regions.split(',')]
            else:
                args.area_ids = [args.area_id]

        hh_api = HeadHunterAPI()
        parser = VacancyParser()

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
            logger.error("Не найдено вакансий.")
            return

        if args.skip_details:
            vacancies_to_process = basic_vacancies
        else:
            logger.info("Загрузка детальной информации...")
            detailed = []
            for i, vac in enumerate(basic_vacancies, 1):
                if i % 20 == 0:
                    logger.info(f"Прогресс деталей: {i}/{len(basic_vacancies)}")
                det = hh_api.get_vacancy_details(vac.get('id'))
                if det:
                    detailed.append(det)
                time.sleep(config.REQUEST_DELAY)
            vacancies_to_process = detailed

        if args.show_vacancies:
            parser.print_vacancies_list(vacancies_to_process)

    else:
        # Стандартный режим
        hh_api = HeadHunterAPI()
        parser = VacancyParser()

        basic_vacancies = hh_api.search_vacancies(
            text=args.query,
            area=args.area_id,
            period_days=args.period,
            max_pages=args.max_pages
        )

        if not basic_vacancies:
            logger.error("Не найдено вакансий.")
            return

        if args.skip_details:
            vacancies_to_process = basic_vacancies
        else:
            logger.info("Загрузка детальной информации...")
            detailed = []
            for i, vac in enumerate(basic_vacancies, 1):
                if i % 20 == 0:
                    logger.info(f"Прогресс: {i}/{len(basic_vacancies)}")
                det = hh_api.get_vacancy_details(vac.get('id'))
                if det:
                    detailed.append(det)
                time.sleep(config.REQUEST_DELAY)
            vacancies_to_process = detailed

    # ====================== 2. ОБРАБОТКА НАВЫКОВ ======================
    skill_freq = extract_and_count_skills(vacancies_to_process, parser)
    if not skill_freq:
        logger.error("Не удалось извлечь навыки.")
        return

    parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)
    print_top_skills(skill_freq)

    # Преобразование в компетенции
    try:
        mapping = load_competency_mapping()
        if mapping:
            comp_counter = map_to_competencies(skill_freq, mapping)
            if comp_counter:
                comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency_mapped.json"
                with open(comp_freq_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(comp_counter.most_common()), f, ensure_ascii=False, indent=2)
                print_top_competencies(comp_counter)
    except Exception as e:
        logger.exception(f"Ошибка преобразования компетенций: {e}")

    if args.excel:
        df = parser.aggregate_to_dataframe(vacancies_to_process)
        if not df.empty:
            filename = "vacancies_it_sector.xlsx" if getattr(args, 'it_sector', False) else f"vacancies_{args.query.replace(' ', '_')}.xlsx"
            parser.save_to_excel(df, filename)

    # ====================== 3. GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ ======================
    if args.run_gap_analysis:
        logger.info("\n" + "="*85)
        logger.info("ЗАПУСК GAP-АНАЛИЗА И ГЕНЕРАЦИИ РЕКОМЕНДАЦИЙ")
        logger.info("="*85)

        try:
            mapping = load_competency_mapping()
            if not mapping or not skill_freq:
                logger.error("Недостаточно данных для gap-анализа")
            else:
                gap_analyzer = GapAnalyzer(mapping)
                recommendation_engine = RecommendationEngine()   # ← Новый движок рекомендаций

                for profile_name in ["base", "dc", "top_dc"]:
                    comp_list = load_student_competencies(profile_name)
                    if not comp_list:
                        logger.warning(f"Профиль {profile_name} пуст")
                        continue

                    student = StudentProfile(
                        student_id=profile_name,
                        name=profile_name.upper(),
                        competencies=comp_list
                    )

                    report = gap_analyzer.analyze(student, skill_freq)
                    recommendations_list = recommendation_engine.generate_recommendations(report, student)

                    # Сохранение по твоей структуре
                    student_dir = config.DATA_DIR / "result" / profile_name
                    student_dir.mkdir(parents=True, exist_ok=True)

                    # comparison_report_{profile}.json
                    with open(student_dir / f"comparison_report_{profile_name}.json", "w", encoding="utf-8") as f:
                        f.write(report.model_dump_json(indent=2))

                    # recommendations_{profile}.json
                    rec_data = {
                        "student": profile_name,
                        "high_priority": [g.skill for g in report.high_demand_gaps[:10]],
                        "recommendations": recommendations_list
                    }

                    with open(student_dir / f"recommendations_{profile_name}.json", "w", encoding="utf-8") as f:
                        json.dump(rec_data, f, ensure_ascii=False, indent=2)

                    logger.info(f"✅ {profile_name.upper():<8} | "
                                f"Покрытие: {report.coverage_percent:6.2f}% | "
                                f"Рекомендаций сгенерировано: {len(recommendations_list)}")

        except Exception as e:
            logger.exception(f"Ошибка при gap-анализе и генерации рекомендаций: {e}")

    # Обновление профилей из CSV
    try:
        csv_path = config.DATA_RAW_DIR / "competency_matrix.csv"
        if not csv_path.exists():
            csv_path = config.DATA_DIR / "last_uploaded" / "competency_matrix.csv"
        if csv_path.exists():
            generate_profiles_from_csv(csv_path)
            logger.info("Профили студентов обновлены из competency_matrix.csv")
    except Exception as e:
        logger.warning(f"Не удалось обновить профили из CSV: {e}")

    logger.info("=" * 85)
    logger.info("ПОЛНЫЙ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН")
    logger.info("=" * 85)


if __name__ == "__main__":
    main()