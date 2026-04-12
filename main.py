#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import time
from pathlib import Path
import json
from typing import List, Dict, Tuple

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.hh_api import HeadHunterAPI
from src.parsing.hh_api_async import HeadHunterAPIAsync 
from src.parsing.vacancy_parser import VacancyParser
from src.models.vacancy import Vacancy
from src.models.student import StudentProfile
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
from src.analyzers.gap_analyzer import GapAnalyzer
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Полный пайплайн: сбор вакансий + gap-анализ + рекомендации")
    
    parser.add_argument('--query', '-q', type=str, default="Python developer")
    parser.add_argument('--area-id', '-a', type=int, default=1)
    parser.add_argument('--max-pages', '-p', type=int, default=3)
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
    
    parser.add_argument('--train-model', action='store_true', help='Обучить LTR-модель на текущих данных (hh_vacancies_basic.json) и выйти')
    parser.add_argument('--use-llm', action='store_true', default=False, help='Использовать LLM (YandexGPT) для живых объяснений рекомендаций')
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
    """Загружает детали вакансий с умным выбором режима."""
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

    # ====================== 2. ОБРАБОТКА НАВЫКОВ ======================
    logger.info("=" * 85)
    logger.info("ИЗВЛЕЧЕНИЕ И ВАЛИДАЦИЯ НАВЫКОВ")
    logger.info("=" * 85)

    result = parser.extract_skills_from_vacancies(vacancies_to_process)

    skill_freq: Dict[str, int] = result["frequencies"]
    hybrid_weights: Dict[str, float] = result.get("hybrid_weights", {})

    if not skill_freq:
        logger.error("Не удалось извлечь навыки.")
        return

    logger.info(f"Извлечено {len(skill_freq)} уникальных валидных навыков "
                f"(Гибридные веса: {len(hybrid_weights)})")

    parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)
    print_top_skills(skill_freq)

    if hybrid_weights:
        print("\n" + "=" * 80)
        print("ТОП-15 НАВЫКОВ ПО ГИБРИДНОМУ ВЕСУ (BM25 + Embeddings)")
        print("=" * 80)
        top_weights = sorted(hybrid_weights.items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (skill, weight) in enumerate(top_weights, 1):
            print(f"{i:2}. {skill:<40} {weight:.4f}")
    else:
        logger.warning("⚠️ hybrid_weights пустой")

    # ====================== МАППИНГ КОМПЕТЕНЦИЙ ======================
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

    # ====================== EXCEL ======================
    if args.excel:
        df = parser.aggregate_to_dataframe(vacancies_to_process)
        if not df.empty:
            filename = "vacancies_it_sector.xlsx" if getattr(args, 'it_sector', False) else \
                        f"vacancies_{args.query.replace(' ', '_')}.xlsx"
            parser.save_to_excel(df, filename)

    # ====================== ОБУЧЕНИЕ LTR-МОДЕЛИ ======================
    if args.train_model:
        logger.info("\n" + "=" * 85)
        logger.info("ЗАПУСК ОБУЧЕНИЯ LTR-МОДЕЛИ")
        logger.info("=" * 85)

        raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        if not raw_file.exists():
            logger.error(f"Файл {raw_file} не найден. Сначала выполните сбор вакансий.")
            sys.exit(1)

        with open(raw_file, 'r', encoding='utf-8') as f:
            training_vacancies = json.load(f)

        logger.info(f"Загружено {len(training_vacancies)} сырых вакансий для обучения")

        ltr_engine = LTRRecommendationEngine()
        ltr_engine.fit(training_vacancies)

        logger.info("✅ Обучение LTR-модели завершено.")
        logger.info(f"Модель сохранена в: {ltr_engine.model_path}")
        logger.info("Для генерации рекомендаций с новой моделью перезапустите main.py без флага --train-model")
        return

    # ====================== 3. GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ ======================
    if args.run_gap_analysis:
        logger.info("\n" + "="*85)
        logger.info("ЗАПУСК GAP-АНАЛИЗА И ГЕНЕРАЦИИ РЕКОМЕНДАЦИЙ")
        logger.info("="*85)

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
                    code_clean = code.strip('. ').upper()
                    if code_clean in competency_mapping:
                        skills.update(competency_mapping[code_clean])
                    elif code.strip('.') in competency_mapping:
                        skills.update(competency_mapping[code.strip('.')])
                return list(skills)

            logger.info("\n" + "="*85)
            logger.info("ПОДГОТОВКА ДАННЫХ")
            logger.info("="*85)
            
            vacancies_skills = []
            for vacancy in vacancies_to_process:
                skills_in_vacancy = []
                if hasattr(vacancy, 'extracted_skills') and vacancy.extracted_skills:
                    skills_in_vacancy = vacancy.extracted_skills
                elif hasattr(vacancy, 'description'):
                    for skill in skill_freq.keys():
                        if skill.lower() in vacancy.description.lower():
                            skills_in_vacancy.append(skill)
                if skills_in_vacancy:
                    vacancies_skills.append(skills_in_vacancy)

            logger.info(f"✓ Подготовлено {len(vacancies_skills)} вакансий")

            if not vacancies_skills:
                logger.error("❌ Не удалось подготовить данные")
                return

            logger.info("\n" + "="*85)
            logger.info("ИНИЦИАЛИЗАЦИЯ EMBEDDINGS + FALLBACK")
            logger.info("="*85)
            
            recommendation_engine = RecommendationEngine(use_ltr=True, use_llm=args.use_llm)
            
            # Временно используем старый comparator для совместимости
            recommendation_engine.comparator = CompetencyComparator(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                use_embeddings=True,
                level="middle"
            )
            recommendation_engine.fit(vacancies_skills)

            skill_weights_raw = recommendation_engine.comparator.get_skill_weights()
            
            if not skill_weights_raw or len(skill_weights_raw) == 0:
                logger.warning("⚠️ Embedding mode вернул пустые веса → берём ГИБРИДНЫЕ веса из парсера")
                if hybrid_weights:
                    skill_weights_raw = hybrid_weights
                    logger.info(f"✓ Использованы гибридные BM25+Embeddings веса ({len(skill_weights_raw)} навыков)")
                else:
                    skill_weights_raw = {k: float(v) for k, v in skill_freq.items() if v > 0}
                    logger.info(f"✓ Fallback на частоты: {len(skill_weights_raw)} навыков")
            
            if not skill_weights_raw:
                logger.error("❌ Даже fallback не смог создать веса")
                return

            logger.info("\n" + "="*85)
            logger.info("ФИЛЬТРАЦИЯ И ОЧИСТКА НАВЫКОВ")
            logger.info("="*85)
            
            filter_engine = SkillFilter()
            competency_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
            competency_freq = {}
            if competency_freq_path.exists():
                with open(competency_freq_path, 'r', encoding='utf-8') as f:
                    competency_freq = json.load(f)
                    
            if not skill_weights_raw or all(v == 0 for v in skill_weights_raw.values()):
                logger.warning("⚠️  Все веса обнулились после фильтра. Используем сырые частоты как веса.")
                skill_weights_raw = {k: float(v) for k, v in skill_freq.items() if v > 0}
            if not skill_weights_raw or len(skill_weights_raw) == 0 or all(abs(v - 1.0) < 0.01 for v in skill_weights_raw.values()):
                logger.warning("⚠️  Embedding mode или плоские веса → используем реальные частоты из skill_freq")
                skill_weights_raw = {k: float(v) for k, v in skill_freq.items() if v > 0}
                logger.info(f"✓ Создано {len(skill_weights_raw)} весов на основе частот")
                
            skill_weights = filter_engine.get_clean_weights(
                skill_weights_raw,
                competency_freq=competency_freq,
                use_reference=True
            )

            weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
            with open(weights_path, "w", encoding="utf-8") as f:
                json.dump(skill_weights, f, ensure_ascii=False, indent=2)

            logger.info("\n" + "="*85)
            logger.info("ИНИЦИАЛИЗАЦИЯ АНАЛИЗАТОРА УРОВНЕЙ ОПЫТА")
            logger.info("="*85)
            
            level_vacancies_data = []
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
                            if 'junior' in exp_lower or 'нет опыта' in exp_lower:
                                vac_experience = 'junior'
                            elif 'senior' in exp_lower or 'более 6' in exp_lower:
                                vac_experience = 'senior'
                            else:
                                vac_experience = 'middle'
                    
                    if vac_skills:
                        level_vacancies_data.append({
                            'skills': vac_skills,
                            'description': vac.description or '',
                            'experience': vac_experience
                        })
                else:
                    vac_skills = [s['name'] for s in vac.get('key_skills', [])]
                    if vac_skills:
                        level_vacancies_data.append({
                            'skills': vac_skills,
                            'description': vac.get('description', ''),
                            'experience': 'middle'
                        })
            
            logger.info(f"Подготовлено {len(level_vacancies_data)} вакансий для анализа уровней")
            
            level_analyzer = SkillLevelAnalyzer()
            level_analyzer.analyze_vacancies(level_vacancies_data)
            logger.info("✅ Анализатор уровней инициализирован")

            logger.info("\n" + "="*85)
            logger.info("АНАЛИЗ ПРОФИЛЕЙ СТУДЕНТОВ")
            logger.info("="*85)
            
            logger.info("\n" + "="*85)
            logger.info("ФОРМИРОВАНИЕ СВОДКИ ПО ПРОФИЛЯМ ЧЕРЕЗ PROFILEEVALUATOR")
            logger.info("="*85)

            profiles: dict[str, StudentProfile] = {}
            profile_levels = {'base': 'junior', 'dc': 'middle', 'top_dc': 'senior'}

            for profile_name, target_level_str in profile_levels.items():
                student_codes = load_student_competencies(profile_name)
                if not student_codes:
                    logger.warning(f"Профиль {profile_name} пуст")
                    continue

                student_skills = map_codes_to_skills(student_codes)
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

            evaluator = ProfileEvaluator(
                skill_weights=skill_weights,
                vacancies_skills=vacancies_skills
            )

            comparison = evaluator.evaluate_multiple_profiles(
                profiles=profiles,
                level_analyzer=level_analyzer,
                skill_weights_by_level=skill_weights_by_level
            )

            summary_path = config.DATA_DIR / "processed" / "profiles_comparison_summary.json"
            summary_text = (
                f"Сравнение {len(comparison.evaluations)} профилей студентов\n"
                f"Средняя готовность: {comparison.average_readiness:.1f}%\n"
                f"Лучший профиль: {comparison.best_evaluation.profile_name} "
                f"({comparison.best_evaluation.level}) — {comparison.best_evaluation.readiness_score:.1f}%\n\n"
                f"Рекомендация для лучшего профиля:\n{comparison.best_evaluation.recommendation}"
            )

            summary_dict = comparison.to_dict_for_json()
            summary_dict["summary_text"] = summary_text
            summary_dict["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ profiles_comparison_summary.json сохранён → {summary_path}")
            logger.info("\n" + "="*60)
            logger.info(summary_text)
            logger.info("="*60)

            # =====================================================================
            # ИНТЕГРАЦИЯ РЕКОМЕНДАЦИЙ (LTR + LLM)
            # =====================================================================
            logger.info("\n" + "="*85)
            logger.info("ГЕНЕРАЦИЯ ПЕРСОНАЛИЗИРОВАННЫХ РЕКОМЕНДАЦИЙ")
            logger.info("="*85)

            rec_engine = RecommendationEngine(use_ltr=True, use_llm=args.use_llm)
            rec_engine.fit(vacancies_skills, skill_weights=skill_weights)

            all_recommendations = {}
            for profile_name, student in profiles.items():
                student_skills = student.skills
                try:
                    full_rec = rec_engine.generate_recommendations(student_skills, student_profile=student)
                    all_recommendations[profile_name] = full_rec
                    logger.info(f"Рекомендации для {profile_name} сгенерированы")
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
                print(f"Покрытие рынка: {summ.get('coverage', 0):.1f}% ({summ.get('covered_skills', 0)}/{summ.get('total_market_skills', 0)} навыков)")
                print("\nТОП-5 РЕКОМЕНДАЦИЙ:")
                for rec in full_rec.get("recommendations", [])[:5]:
                    print(f"{rec['rank']:2}. {rec['skill']:<25} важность: {rec['importance_score']:.3f} ({rec['priority']})")
                    print(f"    Почему: {rec['why_important']}")
                    print(f"    Как учить: {rec['how_to_learn']}")
                    print(f"    Время: {rec['expected_timeframe']}")
                    print()

                rec_file = config.DATA_DIR / "result" / profile_name / f"full_recommendations_{profile_name}.json"
                rec_file.parent.mkdir(parents=True, exist_ok=True)
                with open(rec_file, "w", encoding="utf-8") as f:
                    json.dump(full_rec, f, ensure_ascii=False, indent=2)
                logger.info(f"✓ Полные рекомендации для {profile_name} сохранены в {rec_file}")

            logger.info("\n" + "="*85)
            logger.info("✅ GAP-АНАЛИЗ УСПЕШНО ЗАВЕРШЁН")
            logger.info("="*85)

        except Exception as e:
            logger.exception(f"❌ Ошибка при gap-анализе: {e}")
            import traceback
            traceback.print_exc()

    # === ФИНАЛЬНАЯ ИНФОРМАЦИЯ ===
    show_context_info()

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