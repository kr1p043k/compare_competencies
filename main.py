#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py — Полный пайплайн анализа вакансий и генерации персонализированных рекомендаций
Исправленная версия с LTR-движком без data leakage
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import json
import hashlib
import pickle

if __name__ == "__main__" and sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
from src.parsing.utils import load_it_skills, filter_skills_by_whitelist
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.hh_api_async import HeadHunterAPIAsync
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.vacancy_parser import VacancyParser
from src.models.vacancy import Vacancy
from src.models.student import StudentProfile, merge_skills_hierarchically
from src.parsing.utils import (
    setup_logging,
    collect_vacancies_multiple,
    load_queries_from_file,
    interactive_config,
    print_top_skills,
    print_top_competencies,
    map_to_competencies,
)
from src import config
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.utils import load_competency_mapping

# === Gap-анализ и рекомендации ===
from src.analyzers.vacancy_clustering import VacancyClusterer
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.trends import TrendAnalyzer
from src.analyzers.skill_filter import SkillFilter
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer
from src.analyzers.profile_evaluator import ProfileEvaluator
from src.predictors.recommendation_engine import RecommendationEngine
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine
from src.visualization.charts import (
    show_context_info,
    run_notebook,
    save_all_charts
)


def save_detailed_vacancies(vacancies, logger):
    """Сохраняет детальные вакансии в data/result/ для повторного использования."""
    detailed_file = config.DATA_RESULT_DIR / "hh_vacancies_detailed.json"
    config.DATA_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = []
    for v in vacancies:
        if isinstance(v, Vacancy):
            data_to_save.append(v.raw_data)   # оригинальный JSON от API
        else:
            data_to_save.append(v)
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Детальные вакансии сохранены в {detailed_file} (содержат key_skills и description)")


def convert_float32(obj):
    import numpy as np
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_float32(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_float32(item) for item in obj]
    return obj


def parse_arguments():
    parser = argparse.ArgumentParser(description="Полный пайплайн: сбор вакансий + gap-анализ + рекомендации")

    parser.add_argument('--query', '-q', type=str, default="Backend developer")
    parser.add_argument('--area-id', '-a', type=int, default=1)
    parser.add_argument('--max-pages', '-p', type=int, default=5)
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

    parser.add_argument('--use-async', action='store_true', default=True)
    parser.add_argument('--async-workers', type=int, default=3)
    parser.add_argument('--async-threshold', type=int, default=10)

    parser.add_argument('--run-gap-analysis', action='store_true', default=True)
    parser.add_argument('--run-notebooks', action='store_true')

    parser.add_argument('--train-model', action='store_true',
                        help='Обучить LTR-модель на текущих данных и выйти')
    parser.add_argument('--use-llm', action='store_true', default=False,
                        help='Использовать LLM (YandexGPT) для живых объяснений рекомендаций')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Пропустить сбор вакансий, использовать существующие файлы')
    return parser.parse_args()


def load_student_competencies(profile_name: str):
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


def calculate_expected_vacancies(args) -> int:
    if args.it_sector:
        return 11 * 500
    num_queries = 1
    if args.queries_file:
        try:
            with open(args.queries_file, 'r', encoding='utf-8') as f:
                num_queries = len([line for line in f if line.strip()])
        except:
            num_queries = 1
    return num_queries * min(args.max_vacancies_per_query, 1000)


def get_load_mode(total_vacancies: int, args, logger) -> tuple:
    threshold = args.async_threshold
    if not args.use_async:
        return False, 0, "Пользователь выключил асинхронность"
    if total_vacancies > threshold:
        logger.warning("=" * 90)
        logger.warning("⚠️  ЗАЩИТА ОТ RATE-LIMIT АКТИВИРОВАНА")
        logger.warning("=" * 90)
        logger.warning(f"Ожидаемо вакансий: {total_vacancies}")
        logger.warning(f"Порог: {threshold} → синхронная загрузка")
        return False, 0, f"Большой объём → синхрон"
    logger.info("=" * 90)
    logger.info("✓ АСИНХРОННАЯ ЗАГРУЗКА АКТИВИРОВАНА")
    logger.info("=" * 90)
    return True, args.async_workers, f"Малый объём → асинхрон"


def load_vacancies_details(
    basic_vacancies: list,
    hh_api: HeadHunterAPI,
    use_async: bool,
    async_workers: int,
    parser: VacancyParser,
    logger
) -> list:
    logger.info("Загрузка детальной информации...")

    if use_async:
        logger.info(f"Используется асинхронная загрузка ({async_workers} рабочих)...")
        try:
            from src.parsing.hh_api_async import HeadHunterAPIAsync

            api_async = HeadHunterAPIAsync(
                max_concurrent=async_workers,
                request_delay=config.REQUEST_DELAY
            )

            vacancy_ids = []
            for v in basic_vacancies:
                if isinstance(v, dict):
                    vacancy_ids.append(v.get('id'))
                else:
                    vacancy_ids.append(v.id)

            start_time = time.time()
            raw_detailed = api_async.get_vacancies_details_sync(vacancy_ids)
            elapsed = time.time() - start_time

            detailed = []
            for raw_data in raw_detailed:
                try:
                    detailed.append(Vacancy.from_api(raw_data))
                except ValueError as e:
                    logger.warning(f"Невалидная вакансия: {e}")
                    continue

            logger.info(f"✓ Асинхронная загрузка завершена за {elapsed:.1f} сек")
            logger.info(f"  Загружено: {len(detailed)}/{len(vacancy_ids)} вакансий")
            return detailed

        except Exception as e:
            logger.warning(f"⚠️  Ошибка при асинхронной загрузке: {e}")
            logger.warning("   Переключение на синхронную загрузку...")
            use_async = False

    if not use_async:
        logger.info("Используется синхронная загрузка (последовательная)...")
        detailed = []
        total = len(basic_vacancies)

        start_time = time.time()
        for i, vac in enumerate(basic_vacancies, 1):
            vac_id = vac.get('id') if isinstance(vac, dict) else vac.id

            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                logger.info(f"Прогресс: {i}/{total} ({i*100//total}%) | Осталось ~{remaining/60:.1f} мин")

            det = hh_api.get_vacancy_details_as_object(vac_id)
            if det:
                detailed.append(det)
            time.sleep(config.REQUEST_DELAY)

        elapsed = time.time() - start_time
        logger.info(f"✓ Синхронная загрузка завершена за {elapsed/60:.1f} мин")
        logger.info(f"  Загружено: {len(detailed)}/{total} вакансий")
        return detailed


def get_file_hash(filepath: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_arguments()

    logger.info("=" * 85)
    logger.info("ПОЛНЫЙ ПАЙПЛАЙН: СБОР ВАКАНСИЙ + GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ")
    logger.info("=" * 85)

    # ====================== ОБУЧЕНИЕ LTR-МОДЕЛИ ======================
    if args.train_model:
        logger.info("\n" + "=" * 85)
        logger.info("ЗАПУСК ОБУЧЕНИЯ LTR-МОДЕЛИ")
        logger.info("=" * 85)

        # Приоритет: детальный файл, иначе базовый
        detailed_file = config.DATA_RESULT_DIR / "hh_vacancies_detailed.json"
        basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        if detailed_file.exists():
            raw_file = detailed_file
            logger.info(f"Используем детальные данные из {raw_file}")
        elif basic_file.exists():
            raw_file = basic_file
            logger.warning("Детальный файл не найден, используем базовый (может не хватать навыков)")
        else:
            logger.error(f"Файлы вакансий не найдены. Сначала выполните сбор.")
            sys.exit(1)

        with open(raw_file, 'r', encoding='utf-8') as f:
            training_vacancies = json.load(f)

        logger.info(f"Загружено {len(training_vacancies)} вакансий для обучения")

        ltr_engine = LTRRecommendationEngine()
        ltr_engine.fit(training_vacancies)

        logger.info("✅ Обучение LTR-модели завершено.")
        logger.info(f"Модель сохранена в: {ltr_engine.model_path}")
        logger.info("Для генерации рекомендаций с новой моделью перезапустите main.py без флага --train-model")
        return

    use_multiple = (
        args.interactive or
        args.queries_file is not None or
        args.regions is not None or
        args.industry is not None or
        args.it_sector
    )

    vacancies_to_process = []
    skill_freq = {}
    hybrid_weights = {}
    level_vacancies_data = []
    vacancies_skills = []

    # ====================== ПРОПУСК СБОРА, ЕСЛИ УКАЗАН --skip-collection ======================
    if args.skip_collection:
        # Приоритет: детальный файл из result, затем базовый из raw
        detailed_file = config.DATA_RESULT_DIR / "hh_vacancies_detailed.json"
        basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        if detailed_file.exists():
            raw_file = detailed_file
            logger.info(f"Используем детальные данные из {raw_file}")
        elif basic_file.exists():
            raw_file = basic_file
            logger.warning("Детальный файл не найден, используем базовый (без key_skills и описаний)")
        else:
            logger.error(f"Файлы вакансий не найдены. Сначала выполните сбор.")
            sys.exit(1)

        logger.info(f"Загружаем существующие вакансии из {raw_file}")
        with open(raw_file, 'r', encoding='utf-8') as f:
            basic_vacancies = json.load(f)

        parser = VacancyParser()
        cache_dir = config.DATA_PROCESSED_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "parsed_skills.pkl"
        vacancies_hash = get_file_hash(raw_file)

        cached_result = None
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('source_hash') == vacancies_hash:
                    logger.info("✅ Загружен кэш результатов парсинга навыков")
                    cached_result = cached['result']
                else:
                    logger.info("Файл вакансий изменился, кэш недействителен")
            except Exception as e:
                logger.warning(f"Не удалось загрузить кэш парсинга: {e}")

        if cached_result is not None:
            result = cached_result
            skill_freq = result["frequencies"]
            hybrid_weights = result.get("hybrid_weights", {})
        else:
            result = parser.extract_skills_from_vacancies(basic_vacancies)
            skill_freq = result["frequencies"]
            hybrid_weights = result.get("hybrid_weights", {})

            cache_data = {'source_hash': vacancies_hash, 'result': result}
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"💾 Кэш результатов парсинга сохранён: {cache_path}")

        # === ФИЛЬТРАЦИЯ ГИБРИДНЫХ ВЕСОВ ===
        filter_engine = SkillFilter()
        competency_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
        competency_freq = {}
        if competency_freq_path.exists():
            with open(competency_freq_path, 'r', encoding='utf-8') as f:
                competency_freq = json.load(f)

        hybrid_weights = filter_engine.get_clean_weights(
            hybrid_weights,
            competency_freq=competency_freq,
            use_reference=True
        )
        logger.info(f"📊 hybrid_weights после фильтрации: {len(hybrid_weights)} навыков")

        if not skill_freq:
            logger.error("Не удалось извлечь навыки из существующих вакансий.")
            sys.exit(1)

        logger.info(f"Извлечено {len(skill_freq)} уникальных навыков")

        for vac in basic_vacancies:
            vac_skills = []
            if 'extracted_skills' in vac:
                vac_skills = vac['extracted_skills']
            else:
                desc = vac.get('description', '')
                snippet = vac.get('snippet', {})
                req = snippet.get('requirement', '')
                resp = snippet.get('responsibility', '')
                combined = f"{desc} {req} {resp}"
                vac_skills = parser.extract_skills_from_description(combined)
            vac_copy = vac.copy()
            vac_copy['extracted_skills'] = vac_skills
            vacancies_to_process.append(vac_copy)

            experience = 'middle'
            if 'experience' in vac:
                exp_obj = vac['experience']
                if isinstance(exp_obj, dict):
                    exp_id = exp_obj.get('id', '').lower()
                    if 'less1' in exp_id or 'junior' in exp_id or 'no_experience' in exp_id:
                        experience = 'junior'
                    elif 'between1and3' in exp_id or 'between3and6' in exp_id:
                        experience = 'middle'
                    elif 'between6and10' in exp_id or 'morethan10' in exp_id:
                        experience = 'senior'
                elif isinstance(exp_obj, str):
                    exp_lower = exp_obj.lower()
                    if 'junior' in exp_lower or 'нет опыта' in exp_lower or 'стажер' in exp_lower:
                        experience = 'junior'
                    elif 'senior' in exp_lower or 'более 6' in exp_lower:
                        experience = 'senior'

            if experience == 'middle':
                name = vac.get('name', '').lower()
                if 'junior' in name or 'младший' in name or 'стажер' in name or 'intern' in name:
                    experience = 'junior'
                elif 'senior' in name or 'старший' in name or 'ведущий' in name:
                    experience = 'senior'

            if vac_skills:
                level_vacancies_data.append({
                    'skills': vac_skills,
                    'description': vac.get('description', ''),
                    'experience': experience
                })
                vacancies_skills.append(vac_skills)

        logger.info(f"Подготовлено {len(level_vacancies_data)} вакансий для анализа уровней")

        # === ИНИЦИАЛИЗАЦИЯ АНАЛИЗАТОРА УРОВНЕЙ ===
        level_analyzer = SkillLevelAnalyzer()
        level_analyzer.analyze_vacancies(level_vacancies_data)
        logger.info("✅ Анализатор уровней инициализирован")

        args.run_gap_analysis = True

    else:
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
                        "Computer Vision Engineer", "NLP Engineer", "Data Architect", "ETL Developer",
                        "Python Developer", "Java Developer", "Frontend Developer",
                        "Backend Developer", "Fullstack Developer", "DevOps Engineer",
                        "Embedded Developer", "Blockchain Developer",
                        "iOS Developer", "Android Developer", "React Native Developer", "Flutter Developer",
                        "QA Engineer", "Automation QA Engineer", "Performance QA Engineer",
                        "Специалист по кибербезопасности", "Security Engineer", "DevSecOps Engineer",
                        "SRE инженер", "Системный администратор", "Облачный инженер",
                        "Сетевой инженер", "Администратор баз данных",
                        "Системный аналитик", "Бизнес-аналитик", "Архитектор программного обеспечения",
                        "Solution Architect", "Team Lead", "Tech Lead", "Project Manager IT", "Scrum Master",
                        "UX/UI дизайнер", "Product Designer",
                        "Unity Developer", "Unreal Engine Developer",
                        "Technical Writer",
                    ]
                    args.industry = 7
                    args.max_vacancies_per_query = 100000
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

            logger.info("Сохраняем базовые вакансии...")
            parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json")
            logger.info("Базовые вакансии сохранены.")

            if args.skip_details:
                vacancies_to_process = basic_vacancies
            else:
                total_vacs = len(basic_vacancies)
                use_async, async_workers, reason = get_load_mode(total_vacs, args, logger)
                logger.info(f"Режим загрузки: {reason}")
                vacancies_to_process = load_vacancies_details(
                    basic_vacancies=basic_vacancies,
                    hh_api=hh_api,
                    use_async=use_async,
                    async_workers=async_workers,
                    parser=parser,
                    logger=logger
                )
                save_detailed_vacancies(vacancies_to_process, logger)

            if args.show_vacancies:
                parser.print_vacancies_list(vacancies_to_process)

        else:
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

            logger.info("Сохраняем базовые вакансии...")
            parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json")
            logger.info("Базовые вакансии сохранены.")

            if args.skip_details:
                vacancies_to_process = basic_vacancies
            else:
                total_vacs = len(basic_vacancies)
                use_async, async_workers, reason = get_load_mode(total_vacs, args, logger)
                logger.info(f"Режим загрузки: {reason}")
                vacancies_to_process = load_vacancies_details(
                    basic_vacancies=basic_vacancies,
                    hh_api=hh_api,
                    use_async=use_async,
                    async_workers=async_workers,
                    parser=parser,
                    logger=logger
                )
                save_detailed_vacancies(vacancies_to_process, logger)

        # ====================== 2. ОБРАБОТКА НАВЫКОВ ======================
        logger.info("=" * 85)
        logger.info("ИЗВЛЕЧЕНИЕ И ВАЛИДАЦИЯ НАВЫКОВ")
        logger.info("=" * 85)

        result = parser.extract_skills_from_vacancies(vacancies_to_process)
        skill_freq = result["frequencies"]
        hybrid_weights_raw = result.get("hybrid_weights", {})

        # === ДИАГНОСТИКА ===
        logger.info(f"📊 Получено hybrid_weights_raw: {len(hybrid_weights_raw)} навыков")
        if hybrid_weights_raw:
            sample = list(hybrid_weights_raw.items())[:5]
            logger.info(f"   Примеры: {sample}")
        else:
            logger.warning("⚠️ hybrid_weights_raw пусты!")

        # === ФИЛЬТРАЦИЯ ГИБРИДНЫХ ВЕСОВ ===
        filter_engine = SkillFilter()
        competency_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
        competency_freq = {}
        if competency_freq_path.exists():
            with open(competency_freq_path, 'r', encoding='utf-8') as f:
                competency_freq = json.load(f)

        hybrid_weights = filter_engine.get_clean_weights(
            hybrid_weights_raw,
            competency_freq=competency_freq,
            use_reference=True
        )
        logger.info(f"📊 hybrid_weights после фильтрации: {len(hybrid_weights)} навыков")
        if hybrid_weights:
            sample = list(hybrid_weights.items())[:5]
            logger.info(f"   Примеры после фильтрации: {sample}")

        logger.info(f"Извлечено {len(skill_freq)} уникальных валидных навыков "
                    f"(Гибридные веса: {len(hybrid_weights)})")

        parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)
        print_top_skills(skill_freq)

        whitelist = load_it_skills()
        if whitelist:
            skill_freq_filtered = filter_skills_by_whitelist(skill_freq, whitelist)
            logger.info(f"После фильтрации по белому списку осталось {len(skill_freq_filtered)} навыков (было {len(skill_freq)})")
        else:
            skill_freq_filtered = skill_freq

        trend_analyzer = TrendAnalyzer(skill_freq_filtered)
        trend_analyzer.save_snapshot(skill_freq_filtered, apply_whitelist=False)
        logger.info(f"📸 Снимок рынка сохранён в {trend_analyzer.history_dir}")

        if hybrid_weights:
            print("\n" + "=" * 80)
            print("ТОП-15 НАВЫКОВ ПО ГИБРИДНОМУ ВЕСУ (BM25 + Embeddings)")
            print("=" * 80)
            top_weights = sorted(hybrid_weights.items(), key=lambda x: x[1], reverse=True)[:15]
            for i, (skill, weight) in enumerate(top_weights, 1):
                print(f"{i:2}. {skill:<40} {weight:.4f}")
        else:
            logger.warning("⚠️ hybrid_weights пустой")

        try:
            mapping = load_competency_mapping()
            if mapping:
                comp_counter = map_to_competencies(skill_freq, mapping)
                if comp_counter:
                    filter_engine = SkillFilter()
                    cleaned_comp = {}
                    for skill, count in comp_counter.most_common():
                        skill_clean = skill.lower().strip()
                        if skill_clean in filter_engine.GENERIC_WORDS:
                            continue
                        cleaned_comp[skill_clean] = count
                    comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency_mapped.json"
                    with open(comp_freq_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_comp, f, ensure_ascii=False, indent=2)
                    logger.info(f"✓ Сохранено {len(cleaned_comp)} очищенных компетенций")
                    print_top_competencies(comp_counter)
        except Exception as e:
            logger.exception(f"Ошибка преобразования компетенций: {e}")

        if args.excel:
            df = parser.aggregate_to_dataframe(vacancies_to_process)
            if not df.empty:
                filename = "vacancies_it_sector.xlsx" if getattr(args, 'it_sector', False) else \
                    f"vacancies_{args.query.replace(' ', '_')}.xlsx"
                parser.save_to_excel(df, filename)

        logger.info("\n" + "=" * 85)
        logger.info("ИНИЦИАЛИЗАЦИЯ АНАЛИЗАТОРА УРОВНЕЙ ОПЫТА")
        logger.info("=" * 85)

        for vac in vacancies_to_process:
            if isinstance(vac, Vacancy):
                vac_skills = []
                if hasattr(vac, 'key_skills') and vac.key_skills:
                    vac_skills = [s.name if hasattr(s, 'name') else str(s) for s in vac.key_skills]
                elif hasattr(vac, 'extracted_skills') and vac.extracted_skills:
                    vac_skills = vac.extracted_skills

                vac_experience = 'middle'
                if hasattr(vac, 'experience') and vac.experience:
                    exp_obj = vac.experience
                    if hasattr(exp_obj, 'id'):
                        exp_id = exp_obj.id.lower()
                        if 'less1' in exp_id or 'junior' in exp_id or 'no_experience' in exp_id:
                            vac_experience = 'junior'
                        elif 'between1and3' in exp_id or 'between3and6' in exp_id:
                            vac_experience = 'middle'
                        elif 'between6and10' in exp_id or 'morethan10' in exp_id:
                            vac_experience = 'senior'
                    elif isinstance(exp_obj, str):
                        exp_lower = exp_obj.lower()
                        if 'junior' in exp_lower or 'нет опыта' in exp_lower or 'стажер' in exp_lower:
                            vac_experience = 'junior'
                        elif 'senior' in exp_lower or 'более 6' in exp_lower:
                            vac_experience = 'senior'
                        else:
                            vac_experience = 'middle'

                if vac_experience == 'middle':
                    name = vac.name.lower() if hasattr(vac, 'name') else ''
                    if 'junior' in name or 'младший' in name or 'стажер' in name or 'intern' in name:
                        vac_experience = 'junior'
                    elif 'senior' in name or 'старший' in name or 'ведущий' in name:
                        vac_experience = 'senior'

                if vac_skills:
                    level_vacancies_data.append({
                        'skills': vac_skills,
                        'description': vac.description or '',
                        'experience': vac_experience
                    })
            else:
                vac_skills = [s['name'] for s in vac.get('key_skills', [])]
                if vac_skills:
                    experience = 'middle'
                    exp_obj = vac.get('experience', {})
                    if isinstance(exp_obj, dict):
                        exp_id = exp_obj.get('id', '').lower()
                        if 'less1' in exp_id or 'junior' in exp_id or 'no_experience' in exp_id:
                            experience = 'junior'
                        elif 'between1and3' in exp_id or 'between3and6' in exp_id:
                            experience = 'middle'
                        elif 'between6and10' in exp_id or 'morethan10' in exp_id:
                            experience = 'senior'
                    elif isinstance(exp_obj, str):
                        exp_lower = exp_obj.lower()
                        if 'junior' in exp_lower or 'нет опыта' in exp_lower or 'стажер' in exp_lower:
                            experience = 'junior'
                        elif 'senior' in exp_lower or 'более 6' in exp_lower:
                            experience = 'senior'
                    if experience == 'middle':
                        name = vac.get('name', '').lower()
                        if 'junior' in name or 'младший' in name or 'стажер' in name or 'intern' in name:
                            experience = 'junior'
                        elif 'senior' in name or 'старший' in name or 'ведущий' in name:
                            experience = 'senior'
                    level_vacancies_data.append({
                        'skills': vac_skills,
                        'description': vac.get('description', ''),
                        'experience': experience
                    })

        logger.info(f"Подготовлено {len(level_vacancies_data)} вакансий для анализа уровней")

    # ====================== 3. GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ ======================
    if args.run_gap_analysis:
        logger.info("\n" + "=" * 85)
        logger.info("ЗАПУСК GAP-АНАЛИЗА И ГЕНЕРАЦИИ РЕКОМЕНДАЦИЙ")
        logger.info("=" * 85)

        try:
            if not skill_freq:
                logger.error("Нет навыков для анализа")
                return

            competency_mapping = load_competency_mapping()
            if not competency_mapping:
                logger.warning("Маппинг компетенций не загружен. Студенческие навыки не будут преобразованы.")
            else:
                logger.info(f"Загружен маппинг для {len(competency_mapping)} компетенций")

            def map_codes_to_skills(codes):
                if not competency_mapping:
                    return codes
                skills = set()
                for code in codes:
                    code_norm = ''.join(c for c in code if c.isalnum()).upper()
                    for key, value in competency_mapping.items():
                        key_norm = ''.join(c for c in key if c.isalnum()).upper()
                        if code_norm == key_norm:
                            skills.update(value)
                            break
                return list(skills)

            logger.info("\n" + "=" * 85)
            logger.info("ПОДГОТОВКА ДАННЫХ")
            logger.info("=" * 85)

            # Заполняем vacancies_skills из level_vacancies_data
            if not vacancies_skills:
                for vac_data in level_vacancies_data:
                    skills = vac_data.get('skills', [])
                    if skills:
                        vacancies_skills.append(skills)

            logger.info(f"✓ Подготовлено {len(vacancies_skills)} вакансий")

            if not vacancies_skills:
                logger.error("❌ Не удалось подготовить данные")
                return

            logger.info("\n" + "=" * 85)
            logger.info("ИНИЦИАЛИЗАЦИЯ EMBEDDINGS + FALLBACK")
            logger.info("=" * 85)
            skill_weights_raw = hybrid_weights
            logger.info(f"✓ Использованы гибридные BM25+Embeddings веса после фильтрации ({len(skill_weights_raw)} навыков)")

            if not skill_weights_raw:
                logger.error("❌ Даже fallback не смог создать веса")
                return

            logger.info("\n" + "=" * 85)
            logger.info("ПРОВЕРКА ВЕСОВ")
            logger.info("=" * 85)

            skill_weights = hybrid_weights

            if not skill_weights:
                logger.error("❌ skill_weights пусты после фильтрации!")
                return

            logger.info(f"✓ Используем {len(skill_weights)} очищенных весов навыков")

            weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
            with open(weights_path, "w", encoding="utf-8") as f:
                json.dump(skill_weights, f, ensure_ascii=False, indent=2)

            logger.info("\n" + "=" * 85)
            logger.info("ИНИЦИАЛИЗАЦИЯ АНАЛИЗАТОРА УРОВНЕЙ ОПЫТА")
            logger.info("=" * 85)

            level_analyzer = SkillLevelAnalyzer()
            level_analyzer.analyze_vacancies(level_vacancies_data)
            logger.info("✅ Анализатор уровней инициализирован")

            logger.info("\n" + "=" * 85)
            logger.info("АНАЛИЗ ПРОФИЛЕЙ СТУДЕНТОВ")
            logger.info("=" * 85)

            all_codes = {}
            for name in ['base', 'dc', 'top_dc']:
                codes = load_student_competencies(name)
                if codes:
                    all_codes[name] = codes
                else:
                    logger.warning(f"Профиль {name} не загружен")

            profiles: dict[str, StudentProfile] = {}
            profile_levels = {'base': 'junior', 'dc': 'middle', 'top_dc': 'senior'}

            for profile_name, target_level_str in profile_levels.items():
                if profile_name not in all_codes:
                    continue

                if profile_name == 'top_dc':
                    top_codes = all_codes.get('top_dc', [])
                    dc_codes = all_codes.get('dc', [])
                    base_codes = all_codes.get('base', [])

                    top_skills = map_codes_to_skills(top_codes)
                    dc_skills = map_codes_to_skills(dc_codes)
                    base_skills = map_codes_to_skills(base_codes)

                    student_skills = merge_skills_hierarchically(top_skills, dc_skills, base_skills)
                    student_codes = top_codes
                    logger.info(f"top_dc: объединено {len(top_skills)}+{len(dc_skills)}+{len(base_skills)} → {len(student_skills)} навыков")
                else:
                    student_codes = all_codes[profile_name]
                    student_skills = map_codes_to_skills(student_codes)

                # === НОРМАЛИЗАЦИЯ ДЛЯ ВСЕХ ПРОФИЛЕЙ ===
                student_skills = [SkillNormalizer.normalize(s) for s in student_skills if SkillNormalizer.normalize(s)]
                student_skills = list(dict.fromkeys(student_skills))
                logger.info(f"Профиль {profile_name}: student_skills = {student_skills}")                
                profiles[profile_name] = StudentProfile(
                    profile_name=profile_name,
                    competencies=student_codes,
                    skills=student_skills,
                    target_level=target_level_str
                )
            skill_weights_by_level = {}
            for level in ['junior', 'middle', 'senior']:
                skill_weights_by_level[level] = level_analyzer.get_weights_for_level(
                    skill_weights, level
                )

            hybrid_weights_by_level = {}
            if hybrid_weights:
                for level in ['junior', 'middle', 'senior']:
                    hybrid_weights_by_level[level] = level_analyzer.get_weights_for_level(
                        hybrid_weights, level
                    )

            # --- Создаём evaluator и recommendation_engine ---
            evaluator = ProfileEvaluator(
                skill_weights=skill_weights,
                vacancies_skills=vacancies_skills,
                vacancies_skills_dict=level_vacancies_data,
                hybrid_weights=hybrid_weights,
                skill_weights_by_level=skill_weights_by_level,
            )

            recommendation_engine = RecommendationEngine(
                use_ltr=True,
                use_llm=args.use_llm,
                profile_evaluator=evaluator
            )
            recommendation_engine.comparator = CompetencyComparator(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                use_embeddings=True,
                level="middle",
                similarity_threshold=0.80
            )
            recommendation_engine.fit(vacancies_skills, skill_weights=hybrid_weights)
            logger.info("✓ RecommendationEngine.fit выполнен с hybrid_weights")

            # Оцениваем каждый профиль через новую модель
            evaluations_new = {}
            for profile_name, student in profiles.items():
                logger.info(f"Оценка профиля {profile_name} через новую модель...")
                eval_result = evaluator.evaluate_profile(student, user_type='student')
                evaluations_new[profile_name] = eval_result
                logger.info(f"  {profile_name}: readiness={eval_result['readiness_score']:.2f}%, "
                            f"market_cov={eval_result['market_coverage_score']:.2f}%")

            # Формируем сводку
            readiness_scores = [ev['readiness_score'] for ev in evaluations_new.values()]
            avg_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0
            best_profile = max(evaluations_new.items(), key=lambda x: x[1]['readiness_score'])
            best_profile_name, best_eval = best_profile

            def get_recommendation(readiness_score: float, level: str) -> str:
                if readiness_score >= 80:
                    return f"✅ Готов к {level} уровню"
                elif readiness_score >= 60:
                    return f"📈 Неплохо для {level}, но есть пробелы"
                elif readiness_score >= 40:
                    return f"⚠️ Нужно подготовиться к {level}"
                else:
                    return f"❌ Недостаточно готов к {level}"

            best_level = profiles[best_profile_name].target_level
            best_recommendation = get_recommendation(best_eval['readiness_score'], best_level)

            summary_text = (
                f"Сравнение {len(evaluations_new)} профилей студентов\n"
                f"Средняя готовность: {avg_readiness:.1f}%\n"
                f"Лучший профиль: {best_profile_name} — {best_eval['readiness_score']:.1f}%\n\n"
                f"Рекомендация для лучшего профиля:\n{best_recommendation}"
            )

            summary_dict = {
                "summary_text": summary_text,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "average_readiness": round(avg_readiness, 2),
                "best_profile": best_profile_name,
                "evaluations": {
                    profile_name: {
                        "market_coverage_score": eval_data["market_coverage_score"],
                        "skill_coverage": eval_data["skill_coverage"],
                        "domain_coverage_score": eval_data["domain_coverage_score"],
                        "readiness_score": eval_data["readiness_score"],
                        "avg_gap": eval_data.get("avg_gap", 0),
                        "gaps": eval_data.get("gaps", {}),
                        "top_recommendations": eval_data.get("top_recommendations", [])
                    }
                    for profile_name, eval_data in evaluations_new.items()
                }
            }

            summary_path = config.DATA_DIR / "processed" / "profiles_comparison_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ profiles_comparison_summary.json сохранён → {summary_path}")
            logger.info("\n" + "=" * 60)
            logger.info(summary_text)
            logger.info("=" * 60)

            logger.info("\n" + "=" * 85)
            logger.info("ГЕНЕРАЦИЯ ПЕРСОНАЛИЗИРОВАННЫХ РЕКОМЕНДАЦИЙ")
            logger.info("=" * 85)

            rec_engine = recommendation_engine
            if rec_engine.ltr_engine is None:
                rec_engine.ltr_engine = LTRRecommendationEngine()
                model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
                if model_path.exists():
                    rec_engine.ltr_engine.load_model(model_path)

            all_recommendations = {}
            v2_results = {}
            for profile_name, student in profiles.items():
                student_skills = student.skills
                try:
                    v2_result = evaluator.evaluate_profile(student, user_type='student')
                    v2_results[profile_name] = v2_result

                    skill_weights_context = {}
                    cluster_ctx = v2_result.get('cluster_context') or {}
                    cluster_skills = cluster_ctx.get('skills', {})
                    for skill, metric_dict in v2_result['skill_metrics'].items():
                        if skill in cluster_skills:
                            weight = cluster_skills[skill]
                        else:
                            weight = metric_dict.get('cluster_relevance', 0.15)
                        skill_weights_context[skill] = weight

                    rec_engine.set_cluster_context(skill_weights_context)
                    logger.info(f"Для {profile_name} установлен контекст из {len(skill_weights_context)} навыков (v2)")

                    full_rec = rec_engine.generate_recommendations(student, user_type='student')
                    if 'summary' not in full_rec:
                        full_rec['summary'] = {}
                    full_rec['summary']['market_coverage_score'] = v2_result['market_coverage_score']
                    full_rec['summary']['skill_coverage'] = v2_result['skill_coverage']
                    full_rec['summary']['domain_coverage_score'] = v2_result['domain_coverage_score']
                    full_rec['domain_coverage'] = v2_result['domain_coverage']
                    all_recommendations[profile_name] = full_rec
                    logger.info(f"Рекомендации (v2) для {profile_name} сгенерированы")
                except Exception as e:
                    logger.error(f"Ошибка генерации рекомендаций для {profile_name}: {e}")
                    all_recommendations[profile_name] = None

            for profile_name, full_rec in all_recommendations.items():
                if not full_rec:
                    continue

                print(f"\n📌 РЕКОМЕНДАЦИИ ДЛЯ ПРОФИЛЯ '{profile_name}'")
                print("=" * 70)
                summ = full_rec.get("summary", {})
                print(f"Match score: {summ.get('match_score', 0):.2f} | Confidence: {summ.get('confidence', 0):.2f}")

                if 'market_coverage_score' in summ:
                    print(f"Общее покрытие рынка (v2): {summ['market_coverage_score']:.1f}%")
                    print(f"  - навыковое: {summ.get('skill_coverage', 0):.1f}%")
                    print(f"  - доменное: {summ.get('domain_coverage_score', 0):.1f}%")

                coverage_val = summ.get('coverage', 0)
                details = summ.get('coverage_details', {})
                covered = details.get('covered_skills_count', '?')
                total = details.get('total_market_skills', '?')
                print(f"Покрытие рынка (старое): {coverage_val:.1f}% ({covered}/{total} навыков)")

                print("\nТОП-5 РЕКОМЕНДАЦИЙ:")
                for rec in full_rec.get("recommendations", [])[:5]:
                    print(f"{rec['rank']:2}. {rec['skill']:<25} важность: {rec['importance_score']:.3f} ({rec['priority']})")
                    print(f"    Почему: {rec['why_important']}")
                    print(f"    Как учить: {rec['how_to_learn']}")
                    print(f"    Время: {rec['expected_timeframe']}")
                    print()

                rec_file = config.DATA_DIR / "result" / profile_name / f"full_recommendations_{profile_name}.json"
                rec_file.parent.mkdir(parents=True, exist_ok=True)
                full_rec_serializable = convert_float32(full_rec)
                with open(rec_file, "w", encoding="utf-8") as f:
                    json.dump(full_rec_serializable, f, ensure_ascii=False, indent=2)
                logger.info(f"✓ Полные рекомендации для {profile_name} сохранены в {rec_file}")

            logger.info("\n" + "=" * 85)
            logger.info("✅ GAP-АНАЛИЗ УСПЕШНО ЗАВЕРШЁН")
            logger.info("=" * 85)

        except Exception as e:
            logger.exception(f"❌ Ошибка при gap-анализе: {e}")
            import traceback
            traceback.print_exc()
            if 'comparison' not in locals():
                comparison = type('obj', (object,), {'evaluations': []})()
            return

    show_context_info()

    logger.info("\n" + "=" * 85)
    logger.info("ГЕНЕРАЦИЯ ПРЕЗЕНТАЦИОННЫХ ГРАФИКОВ")
    logger.info("=" * 85)

    output_viz_dir = config.DATA_DIR / "result"
    output_viz_dir.mkdir(parents=True, exist_ok=True)
    save_all_charts(evaluations_new, output_viz_dir, use_ml=True)

    logger.info(f"✅ Презентационные графики сохранены в {output_viz_dir}")

    if args.run_notebooks:
        logger.info("Запуск Jupyter ноутбуков...")
        run_notebook("01_hh_analysis.ipynb", output_dir=config.DATA_DIR / "notebooks")
        run_notebook("02_competency_matching.ipynb", output_dir=config.DATA_DIR / "notebooks")

    try:
        csv_path = config.DATA_RAW_DIR / "competency_matrix.csv"
        if not csv_path.exists():
            csv_path = config.DATA_DIR / "last_uploaded" / "competency_matrix.csv"
        if csv_path.exists():
            generate_profiles_from_csv(csv_path)
            logger.info("✓ Профили студентов обновлены из competency_matrix.csv")
    except Exception as e:
        logger.warning(f"Не удалось обновить профили из CSV: {e}")

    logger.info("\n" + "=" * 85)
    logger.info("✅ ПОЛНЫЙ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН")
    logger.info("=" * 85)
    logger.info(f"📁 Результаты сохранены в: {config.DATA_DIR / 'result'}")
    logger.info(f"⏰ Время выполнения: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 85)


if __name__ == "__main__":
    main()