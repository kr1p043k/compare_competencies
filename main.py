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
from src.parsing.hh_api_async import HeadHunterAPIAsync 
from src.parsing.vacancy_parser import VacancyParser
from src.models.vacancy import Vacancy
from src.models.student import StudentProfile, ExperienceLevel
from src.parsing.utils import (
    setup_logging,
    collect_vacancies_multiple,
    load_queries_from_file,
    interactive_config,
    print_top_skills,
    print_top_competencies,
    extract_and_count_skills,
    map_to_competencies,
    load_it_skills
)
from src import config
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.utils import load_competency_mapping

# === Gap-анализ и рекомендации ===
from src.models.student import StudentProfile
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.skill_filter import SkillFilter
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer
from src.analyzers.profile_evaluator import ProfileEvaluator
from src.analyzers.trends import TrendAnalyzer
from src.predictors.recommendation_engine import RecommendationEngine
from src.predictors.ml_recommendation_engine import MLRecommendationEngine
from src.visualization.charts import (
    show_context_info,
    run_notebook,
    save_all_charts
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Полный пайплайн: сбор вакансий + gap-анализ + рекомендации")
    
    parser.add_argument('--query', '-q', type=str, default="Fullstack developer")
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
    
    parser.add_argument('--use-async', action='store_true', default=True)
    parser.add_argument('--async-workers', type=int, default=3)
    parser.add_argument('--async-threshold', type=int, default=500)
    
    parser.add_argument('--run-gap-analysis', action='store_true', default=True)
    parser.add_argument('--run-notebooks', action='store_true')
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
                    logger.warning(f"Невалидная ваканси��: {e}")
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
    
    skill_freq = parser.extract_skills_from_vacancies(vacancies_to_process)
    
    if not skill_freq:
        logger.error("Не удалось извлечь навыки.")
        return

    parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)
    print_top_skills(skill_freq)

    try:
        mapping = load_competency_mapping()
        if mapping:
            comp_counter = map_to_competencies(skill_freq, mapping)
            if comp_counter:
                # === ОЧИЩАЕМ ПЕРЕД СОХРАНЕНИЕМ ===
                filter_engine = SkillFilter()
                
                # Фильтруем competency_counter
                cleaned_comp = {}
                for skill, count in comp_counter.most_common():
                    skill_clean = skill.lower().strip()
                    if skill_clean == "frontend":
                        print("DEBUG: frontend source:", skill, "=>", count)
                    # Исключаем generic слова
                    if skill_clean in filter_engine.GENERIC_WORDS:
                        logger.debug(f"  ⊘ удаляем generic: '{skill}'")
                        continue
                    
                    # Исключаем bigrams (если надо)
                    # (опционально - зависит от ваших требований)
                    
                    cleaned_comp[skill_clean] = count
                
                # === Добавь ПЕЧАТЬ для отладки после фильтрации ===
                print([k for k in cleaned_comp if k in filter_engine.GENERIC_WORDS])
                logger.info(f"📝 CHECK: generic words после фильтрации: {[k for k in cleaned_comp if k in filter_engine.GENERIC_WORDS]}")
                
                comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency_mapped.json"
                with open(comp_freq_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_comp, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✓ Сохранено {len(cleaned_comp)} очищенных компетенций (удалено {len(comp_counter) - len(cleaned_comp)})")
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

            # === ПОДГОТОВКА ДАННЫХ ДЛЯ TF-IDF ===
            logger.info("\n" + "="*85)
            logger.info("ПОДГОТОВКА ДАННЫХ ДЛЯ TF-IDF АНАЛИЗА")
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

            logger.info(f"✓ Подготовлено {len(vacancies_skills)} вакансий для TF-IDF")

            if not vacancies_skills:
                logger.error("❌ Не удалось подготовить данные для TF-IDF")
                return

            # === TF-IDF (оригинальный код) ===
            logger.info("\n" + "="*85)
            logger.info("ИНИЦИАЛИЗАЦИЯ TF-IDF С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ")
            logger.info("="*85)
            
            recommendation_engine = RecommendationEngine()
            recommendation_engine.comparator = CompetencyComparator(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            recommendation_engine.fit(vacancies_skills)

            skill_weights_raw = recommendation_engine.comparator.get_skill_weights()
            if not skill_weights_raw:
                logger.error("❌ Не удалось получить веса навыков")
                return

            # === ФИЛЬТРАЦИЯ ===
            logger.info("\n" + "="*85)
            logger.info("ФИЛЬТРАЦИЯ И ОЧИСТКА НАВЫКОВ")
            logger.info("="*85)
            
            filter_engine = SkillFilter()
            competency_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
            competency_freq = {}
            if competency_freq_path.exists():
                with open(competency_freq_path, 'r', encoding='utf-8') as f:
                    competency_freq = json.load(f)

            skill_weights = filter_engine.get_clean_weights(
                skill_weights_raw,
                competency_freq=competency_freq,
                use_reference=True
            )

            weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
            with open(weights_path, "w", encoding="utf-8") as f:
                json.dump(skill_weights, f, ensure_ascii=False, indent=2)

            # =====================================================================
            # ← НОВАЯ ИНТЕГРАЦИЯ: АНАЛИЗАТОР УРОВНЕЙ
            # =====================================================================
            logger.info("\n" + "="*85)
            logger.info("ИНИЦИАЛИЗАЦИЯ АНАЛИЗАТОРА УРОВНЕЙ ОПЫТА")
            logger.info("="*85)
            
            # Подготавливаем данные для level analyzer
            ml_vacancies_data = []
            for vac in vacancies_to_process:
                if isinstance(vac, Vacancy):
                    # Извлекаем навыки
                    vac_skills = []
                    if hasattr(vac, 'key_skills') and vac.key_skills:
                        vac_skills = [s.name if hasattr(s, 'name') else str(s) for s in vac.key_skills]
                    elif hasattr(vac, 'extracted_skills') and vac.extracted_skills:
                        vac_skills = vac.extracted_skills
                    
                    # Извлекаем опыт
                    vac_experience = 'middle'  # default
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
                    
                    if vac_skills:  # только если есть навыки
                        vac_data = {
                            'skills': vac_skills,
                            'description': vac.description or '',
                            'experience': vac_experience
                        }
                        ml_vacancies_data.append(vac_data)
                else:
                    # dict формат (для совместимости)
                    vac_skills = [s['name'] for s in vac.get('key_skills', [])]
                    vac_experience = 'middle'  # default для dict
                    
                    if vac_skills:
                        vac_data = {
                            'skills': vac_skills,
                            'description': vac.get('description', ''),
                            'experience': vac_experience
                        }
                        ml_vacancies_data.append(vac_data)
            
            logger.info(f"Подготовлено {len(ml_vacancies_data)} вакансий для анализа уровней")
            
            # Инициализируем level analyzer
            level_analyzer = SkillLevelAnalyzer()
            level_analyzer.analyze_vacancies(ml_vacancies_data)
            logger.info("✅ Анализатор уровней инициализирован")

            # =====================================================================
            # ← АНАЛИЗ ПРОФИЛЕЙ С УЧЁТОМ УРОВНЯ
            # =====================================================================
            results_for_charts = {}
            all_profiles_results = []

            logger.info("\n" + "="*85)
            logger.info("АНАЛИЗ ПРОФИЛЕЙ СТУДЕНТОВ")
            logger.info("="*85)
            
            # Маппим профили на уровни опыта
            profile_levels = {
                'base': 'junior',      # базовый профиль → junior
                'dc': 'middle',        # основной курс → middle
                'top_dc': 'senior'     # продвинутый → senior
            }
            
            for profile_name in ["base", "dc", "top_dc"]:
                logger.info(f"\n{'-'*85}")
                target_level = profile_levels[profile_name]
                logger.info(f"Профиль: {profile_name.upper()} (уровень: {target_level.upper()})")
                logger.info(f"{'-'*85}")
                
                student_codes = load_student_competencies(profile_name)
                if not student_codes:
                    logger.warning(f"Профиль {profile_name} пуст - пропускаем")
                    continue

                student_skills = map_codes_to_skills(student_codes)
                
                # ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: получаем веса для уровня
                logger.info(f"Получаем веса навыков для уровня {target_level}...")
                profile_skill_weights = level_analyzer.get_weights_for_level(
                    skill_weights,
                    target_level
                )
                logger.info(f"  Переупорядочено {len(profile_skill_weights)} навыков")
                
                # Создаём уровень-специфичный gap analyzer
                from src.analyzers.gap_analyzer import GapAnalyzer
                profile_gap_analyzer = GapAnalyzer(profile_skill_weights)

                # TF-IDF (используем общий comparator)
                score, confidence = recommendation_engine.comparator.compare(student_skills)

                # ← используем профильный analyzer с уровень-специфичными весами
                gaps = profile_gap_analyzer.analyze_gap(student_skills)
                coverage, coverage_details = profile_gap_analyzer.coverage(student_skills)
                recommendations_text = profile_gap_analyzer.get_recommendations(student_skills, gaps)

                # === ВЫВОД ПРОГРЕССА ===
                logger.info(f"Результаты анализа для {profile_name} ({target_level}):")
                logger.info(f"  📊 TF-IDF скор: {score:.4f} (confidence: {confidence:.4f})")
                logger.info(f"  📈 Coverage: {coverage_details['coverage_percent']:.2f}%")
                logger.info(f"  🔴 High priority gaps: {len(gaps['high_priority'])}")
                logger.info(f"  🟡 Medium priority gaps: {len(gaps['medium_priority'])}")
                logger.info(f"  🟢 Low priority gaps: {len(gaps['low_priority'])}")

                # === ML АНАЛИЗ (анализ на основе важности) ===
                ml_impacts = []
                if len(gaps['high_priority']) > 0 or len(gaps['medium_priority']) > 0:
                    missing_skills = (
                        [g['skill'] for g in gaps.get('high_priority', [])[:15]] +
                        [g['skill'] for g in gaps.get('medium_priority', [])[:15]]
                    )
                    
                    # Используем fallback версию (без ML модели)
                    if missing_skills:
                        for skill in missing_skills[:10]:
                            skill_lower = skill.lower().strip()
                            importance = profile_skill_weights.get(skill, 0)
                            
                            # Получаем информацию о навыке
                            roadmap = level_analyzer.get_skill_roadmap(skill)
                            skill_level = level_analyzer.get_skill_level(skill)
                            
                            explanation = f"Уровень: {skill_level} | "
                            if roadmap[target_level]:
                                explanation += f"Требуется для {target_level}"
                            else:
                                explanation += f"Встречается в вакансиях"
                            
                            # Добавляем информацию о дорожке развития
                            levels_present = [l for l in ['junior', 'middle', 'senior'] if roadmap[l]]
                            if len(levels_present) > 1:
                                explanation += f" | Путь развития: {' → '.join(levels_present)}"
                            
                            ml_impacts.append((skill, round(importance * 100), explanation))
                        
                        ml_impacts.sort(key=lambda x: x[1], reverse=True)
                        ml_impacts = ml_impacts[:10]
                        
                        if ml_impacts:
                            logger.info(f"💡 ML рекомендации ({len(ml_impacts)}):")
                            for skill, score, expl in ml_impacts[:3]:
                                logger.info(f"  - {skill} (скор: {score}) | {expl}")

                # === СОХРАНЕНИЕ ===
                student_dir = config.DATA_DIR / "result" / profile_name
                student_dir.mkdir(parents=True, exist_ok=True)

                detailed_analysis = {
                    "student": profile_name,
                    "level": target_level,
                    "timestamp": str(time.strftime('%Y-%m-%d %H:%M:%S')),
                    "tfidf_scores": {"match_score": round(score, 4), "confidence": round(confidence, 4)},
                    "coverage_stats": {
                        "coverage_percent": round(coverage_details["coverage_percent"], 2),
                        "covered_skills_count": coverage_details["covered_skills_count"],
                        "total_market_skills": coverage_details["total_market_skills"],
                        "covered_weight": round(coverage_details.get("covered_weight", 0), 2),
                        "total_weight": round(coverage_details.get("total_weight", 0), 2)
                    },
                    "gaps": {
                        "high_priority": gaps["high_priority"][:15],
                        "medium_priority": gaps["medium_priority"][:15],
                        "low_priority": gaps["low_priority"][:10],
                        "total_gaps": gaps.get("total_gaps", 0),
                        "gaps_stats": gaps.get("stats", {})
                    },
                    "recommendations": recommendations_text,
                    "ml_recommendations": [
                        {
                            "skill": skill, 
                            "importance_score": impact,
                            "explanation": expl,
                            "roadmap": level_analyzer.get_skill_roadmap(skill)
                        }
                        for skill, impact, expl in ml_impacts
                    ],
                    "student_skills": sorted(student_skills)[:50],
                    "market_top_skills": profile_gap_analyzer.top_market_skills(30)
                }
                
                with open(student_dir / f"detailed_analysis_{profile_name}.json", "w", encoding="utf-8") as f:
                    json.dump(detailed_analysis, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✓ Детальный анализ + ML сохранён для {profile_name}")

            # === СОЗДАНИЕ СВОДКИ ПО ПРОФИЛЯМ ===
            logger.info("\n" + "="*85)
            logger.info("СОЗДАНИЕ СВОДКИ ПО ПРОФИЛЯМ")
            logger.info("="*85)
            
            profile_comparison_data = {
                "timestamp": str(time.strftime('%Y-%m-%d %H:%M:%S')),
                "total_profiles_analyzed": 0,
                "profiles": [],
                "best_profile": None,
                "average_coverage": 0.0
            }
            
            profile_coverages = []
            profile_details_list = []
            
            for profile_name in ["base", "dc", "top_dc"]:
                target_level = profile_levels[profile_name]
                student_codes = load_student_competencies(profile_name)
                
                if not student_codes:
                    logger.warning(f"Профиль {profile_name} пуст - пропускаем в сводку")
                    continue
                
                student_skills = map_codes_to_skills(student_codes)
                
                # Получаем веса для уровня
                profile_skill_weights = level_analyzer.get_weights_for_level(
                    skill_weights,
                    target_level
                )
                
                # Создаём gap analyzer для этого профиля
                profile_gap_analyzer = GapAnalyzer(profile_skill_weights)
                
                # Анализируем
                score, confidence = recommendation_engine.comparator.compare(student_skills)
                gaps = profile_gap_analyzer.analyze_gap(student_skills)
                coverage, _ = profile_gap_analyzer.coverage(student_skills)
                
                profile_detail = {
                    "profile": profile_name,
                    "level": target_level,
                    "score": round(score, 4),
                    "confidence": round(confidence, 4),
                    "coverage": round(coverage, 2),
                    "high_gaps": len(gaps['high_priority']),
                    "medium_gaps": len(gaps['medium_priority']),
                    "low_gaps": len(gaps['low_priority'])
                }
                
                profile_comparison_data['profiles'].append(profile_detail)
                profile_coverages.append(coverage)
                profile_details_list.append(profile_detail)
                
                logger.info(f"✓ Добавлена сводка для {profile_name}: coverage={coverage:.2f}%")
            
            # Заполняем общие статистики
            profile_comparison_data['total_profiles_analyzed'] = len(profile_comparison_data['profiles'])
            
            if profile_coverages:
                profile_comparison_data['average_coverage'] = round(sum(profile_coverages) / len(profile_coverages), 2)
                
                # Находим best profile по coverage
                best_idx = profile_coverages.index(max(profile_coverages))
                profile_comparison_data['best_profile'] = profile_details_list[best_idx]
            
            # Сохраняем сводку
            summary_path = config.DATA_DIR / "processed" / "profiles_comparison_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(profile_comparison_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Сводка по профилям сохранена в {summary_path}")
            logger.info(f"   Total profiles: {profile_comparison_data['total_profiles_analyzed']}")
            logger.info(f"   Average coverage: {profile_comparison_data['average_coverage']}%")
            if profile_comparison_data['best_profile']:
                logger.info(f"   Best profile: {profile_comparison_data['best_profile']['profile']} ({profile_comparison_data['best_profile']['coverage']}%)")

            # === СОЗДАНИЕ ГРАФИКОВ ===
            if results_for_charts:
                logger.info("\n" + "="*85)
                logger.info("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
                logger.info("="*85)
                
                logger.info("Генерируются графики...")
                save_all_charts(results_for_charts, config.DATA_DIR / "result")
                logger.info("✓ Графики сохранены в data/result/")
            
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
            logger.info("✓ Профили студент��в обновлены из competency_matrix.csv")
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