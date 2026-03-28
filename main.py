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
from src.analyzers.trends import TrendAnalyzer
from src.predictors.recommendation_engine import RecommendationEngine

from src.visualization.charts import (
    show_context_info,
    run_notebook,
    save_all_charts
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Полный пайплайн: сбор вакансий + gap-анализ + рекомендации")
    
    parser.add_argument('--query', '-q', type=str, default="Backend Developer")
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
    
    parser.add_argument(
        '--use-async',
        action='store_true',
        default=True,
        help="Использовать асинхронную загрузку деталей вакансий (по умолчанию True)"
    )
    parser.add_argument(
        '--async-workers',
        type=int,
        default=3,
        help="Количество параллельных рабочих для асинхронной загрузки (по умолчанию 3)"
    )
    parser.add_argument(
        '--async-threshold',
        type=int,
        default=1500,
        help="Максимальное количество вакансий для асинхронной загрузки (больше -> синхрон)"
    )
    
    parser.add_argument('--run-gap-analysis', action='store_true', default=True,
                        help="Выполнить gap-анализ и генерацию рекомендаций (по умолчанию True)")
    
    parser.add_argument('--run-notebooks', action='store_true',
                        help="Запустить Jupyter ноутбуки после сбора данных")
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


def calculate_expected_vacancies(args) -> int:
    """Рассчитывает ожидаемое количество вакансий для загрузки."""
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
    """Определяет режим загрузки (асинхронный или синхронный)."""
    threshold = args.async_threshold
    
    if not args.use_async:
        return False, 0, "Пользователь выключил асинхронность (--use-async False)"
    
    if total_vacancies > threshold:
        logger.warning("=" * 90)
        logger.warning("⚠️  ЗАЩИТА ОТ RATE-LIMIT АКТИВИРОВАНА")
        logger.warning("=" * 90)
        logger.warning(f"Ожидаемо вакансий: {total_vacancies}")
        logger.warning(f"Порог асинхронности: {threshold}")
        logger.warning(f"Рекомендация: Синхронная загрузка (безопаснее для hh.ru API)")
        logger.warning(f"Время: ~{int(total_vacancies * 0.5 / 60)} минут")
        logger.warning("=" * 90)
        
        return False, 0, f"Большой объём ({total_vacancies} вак > {threshold}) -> синхронная загрузка"
    
    logger.info("=" * 90)
    logger.info("✓ АСИНХРОННАЯ ЗАГРУЗКА АКТИВИРОВАНА")
    logger.info("=" * 90)
    logger.info(f"Ожидаемо вакансий: {total_vacancies}")
    logger.info(f"Рабочих процессов: {args.async_workers}")
    logger.info(f"Ожидаемое время: ~{max(total_vacancies * 0.5 / args.async_workers / 60, 1):.1f} минут")
    logger.info("=" * 90)
    
    return True, args.async_workers, f"Малый объём ({total_vacancies} вак < {threshold}) -> асинхронная загрузка"


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
            
            # skill_freq уже содержит очищенные, отфильтрованные навыки!
            logger.info(f"✓ Используем уже очищенные навыки из парсинга")
            logger.info(f"  - Уникальных навыков: {len(skill_freq)}")
            
            # Преобразуем словарь частот в формат для TF-IDF
            # Каждый навык повторяется столько раз, сколько раз встречался
            vacancies_skills = []
            for skill_name, count in skill_freq.items():
                if count > 0:
                    # Повторяем навык count раз (пропорционально частоте)
                    for _ in range(min(count, 100)):  # Ограничиваем 100 для производительности
                        vacancies_skills.append([skill_name])
            
            logger.info(f"✓ Подготовлено {len(vacancies_skills)} документов для TF-IDF")
            logger.info(f"  (каждый документ = один навык повторённый по частоте)")
            
            if not vacancies_skills:
                logger.error("❌ Не удалось подготовить данные для TF-IDF")
                return

            # === ИНИЦИАЛИЗАЦИЯ ENGINES (с оптимальными параметрами) ===
            logger.info("\n" + "="*85)
            logger.info("ИНИЦИАЛИЗАЦИЯ TF-IDF С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ")
            logger.info("="*85)
            
            recommendation_engine = RecommendationEngine()
            
            # === ЖЁСТКИЕ ТРЕБОВАНИЯ: ТОЛЬКО UNIGRAMS ===
            recommendation_engine.comparator = CompetencyComparator(
                ngram_range=(1, 1),  # ← ТОЛЬКО unigrams, БЕЗ bigrams!
                min_df=1,            # Даже одно упоминание OK (уже отфильтровано)
                max_df=0.95
            )
            
            logger.info("Параметры TF-IDF:")
            logger.info("  - ngram_range: (1, 1) - ТОЛЬКО отдельные слова")
            logger.info("  - min_df: 1 - даже редкие навыки в порядке")
            logger.info("  - max_df: 0.95 - исключаются частые слова")
            logger.info("  - max_features: 300 - максимум для чистоты")
            
            recommendation_engine.fit(vacancies_skills)

            skill_weights_raw = recommendation_engine.comparator.get_skill_weights()
            
            if not skill_weights_raw:
                logger.error("❌ Не удалось получить веса навыков")
                return
            
            # === ФИЛЬТРАЦИЯ И ОЧИСТКА ===
            logger.info("\n" + "="*85)
            logger.info("ФИЛЬТРАЦИЯ И ОЧИСТКА НАВЫКОВ")
            logger.info("="*85)
            
            filter_engine = SkillFilter()
            
            # Загружаем competency_frequency как reference
            competency_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
            competency_freq = {}
            if competency_freq_path.exists():
                with open(competency_freq_path, 'r', encoding='utf-8') as f:
                    competency_freq = json.load(f)
                logger.info(f"✓ Загружена competency_frequency: {len(competency_freq)} навыков")
            
            # Применяем фильтрацию
            skill_weights = filter_engine.get_clean_weights(
                skill_weights_raw,
                competency_freq=competency_freq,
                use_reference=True
            )
            
            if not skill_weights:
                logger.error("❌ После фильтрации навыков не осталось")
                return
            
            # Сохраняем очищенные веса
            weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
            with open(weights_path, "w", encoding="utf-8") as f:
                json.dump(skill_weights, f, ensure_ascii=False, indent=2)
            logger.info(f"\n✓ Очищенные веса сохранены в {weights_path}")
            
            logger.info(f"\n✓ TF-IDF модель готова")
            logger.info(f"✓ Получено {len(skill_weights)} чистых навыков (после фильтрации)")
            
            # Логируем топ-20 ЧИСТЫХ навыков
            logger.info("\nТоп-20 навыков по спросу на рынке (ЧИСТЫЕ):")
            top_20_skills = sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)[:20]
            for rank, (skill, weight) in enumerate(top_20_skills, 1):
                logger.info(f"  {rank:2d}. {skill:25s} - {weight:.4f}")
            
            # === СТАТИСТИКА МОДЕЛИ ===
            comparator_stats = recommendation_engine.comparator.get_stats()
            logger.info(f"\nСтатистика TF-IDF модели:")
            logger.info(f"  - Размер словаря (исходный): {len(skill_weights_raw)}")
            logger.info(f"  - Размер словаря (после фильтрации): {len(skill_weights)}")
            logger.info(f"  - Удалено: {len(skill_weights_raw) - len(skill_weights)} навыков")

            results_for_charts = {}
            all_profiles_results = []

            # === АНАЛИЗ КАЖДОГО ПРОФИЛЯ ===
            logger.info("\n" + "="*85)
            logger.info("АНАЛИЗ ПРОФИЛЕЙ СТУДЕНТОВ")
            logger.info("="*85)
            
            for profile_name in ["base", "dc", "top_dc"]:
                logger.info(f"\n{'-'*85}")
                logger.info(f"Профиль: {profile_name.upper()}")
                logger.info(f"{'-'*85}")
                
                student_codes = load_student_competencies(profile_name)
                if not student_codes:
                    logger.warning(f"Профиль {profile_name} пуст - пропускаем")
                    continue

                student_skills = map_codes_to_skills(student_codes)
                logger.info(f"📚 Входные данные:")
                logger.info(f"  - Кодов компетенций: {len(student_codes)}")
                logger.info(f"  - Мэпированных навыков: {len(student_skills)}")
                logger.info(f"  - Уникальных навыков: {len(set(s.lower() for s in student_skills))}")

                # === СРАВНЕНИЕ С РЫНКОМ (TF-IDF) ===
                score, confidence = recommendation_engine.comparator.compare(student_skills)
                logger.info(f"\n📊 TF-IDF сравнение с рынком:")
                logger.info(f"  - Match Score: {score:.4f} (косинусное сходство)")
                logger.info(f"  - Confidence: {confidence:.4f} (уверенность в оценке)")
                
                student_dir = config.DATA_DIR / "result" / profile_name
                student_dir.mkdir(parents=True, exist_ok=True)

                # === GAP АНАЛИЗ ===
                gaps = recommendation_engine.gap_analyzer.analyze_gap(student_skills)
                coverage, coverage_details = recommendation_engine.gap_analyzer.coverage(student_skills)
                recommendations_text = recommendation_engine.gap_analyzer.get_recommendations(student_skills, gaps)

                logger.info(f"\n📈 Анализ пробелов (Gaps):")
                logger.info(f"  - Coverage: {coverage_details['coverage_percent']:.2f}%")
                logger.info(f"  - Covered skills: {coverage_details['covered_skills_count']}/{coverage_details['total_market_skills']}")
                logger.info(f"  - High priority gaps: {len(gaps['high_priority'])}")
                logger.info(f"  - Medium priority gaps: {len(gaps['medium_priority'])}")
                logger.info(f"  - Low priority gaps: {len(gaps['low_priority'])}")

                if gaps['stats']:
                    logger.info(f"\n📉 Статистика по пробелам:")
                    logger.info(f"  - Среднее значение: {gaps['stats'].get('avg_weight', 0):.4f}")
                    logger.info(f"  - Максимум: {gaps['stats'].get('max_weight', 0):.4f}")
                    logger.info(f"  - Минимум: {gaps['stats'].get('min_weight', 0):.4f}")

                # === РЕКОМЕНДАЦИИ ===
                logger.info(f"\n💡 Рекомендации для {profile_name}:")
                for i, rec in enumerate(recommendations_text, 1):
                    logger.info(f"  {i}. {rec}")

                # === ТОП ДЕФИЦИТНЫЕ НАВЫКИ ===
                if gaps['high_priority']:
                    logger.info(f"\n🔴 Топ-5 КРИТИЧНЫХ ДЕФИЦИТОВ:")
                    for i, gap in enumerate(gaps['high_priority'][:5], 1):
                        logger.info(f"  {i}. {gap['skill']:25s} - важность: {gap['importance']:.4f}")

                if gaps['medium_priority']:
                    logger.info(f"\n🟡 Топ-5 СРЕДНИХ ДЕФИЦИТОВ:")
                    for i, gap in enumerate(gaps['medium_priority'][:5], 1):
                        logger.info(f"  {i}. {gap['skill']:25s} - важность: {gap['importance']:.4f}")

                # === СОХРАНЕНИЕ ДЕТАЛЬНОГО АНАЛИЗА ===
                detailed_analysis = {
                    "student": profile_name,
                    "timestamp": str(time.strftime('%Y-%m-%d %H:%M:%S')),
                    "tfidf_scores": {
                        "match_score": round(score, 4),
                        "confidence": round(confidence, 4)
                    },
                    "coverage_stats": {
                        "coverage_percent": round(coverage_details["coverage_percent"], 2),
                        "covered_skills_count": coverage_details["covered_skills_count"],
                        "total_market_skills": coverage_details["total_market_skills"],
                        "covered_weight": round(coverage_details["covered_weight"], 2),
                        "total_weight": round(coverage_details["total_weight"], 2)
                    },
                    "gaps": {
                        "high_priority": gaps["high_priority"][:15],
                        "medium_priority": gaps["medium_priority"][:15],
                        "low_priority": gaps["low_priority"][:10],
                        "total_gaps": gaps["total_gaps"],
                        "gaps_stats": gaps["stats"]
                    },
                    "recommendations": recommendations_text,
                    "student_skills": sorted(student_skills)[:50],
                    "market_top_skills": recommendation_engine.gap_analyzer.top_market_skills(30)
                }
                
                with open(student_dir / f"detailed_analysis_{profile_name}.json", "w", encoding="utf-8") as f:
                    json.dump(detailed_analysis, f, ensure_ascii=False, indent=2)
                
                logger.info(f"\n✓ Детальный анализ сохранён:")
                logger.info(f"   {student_dir}/detailed_analysis_{profile_name}.json")

                # === ДАННЫЕ ДЛЯ ГРАФИКОВ ===
                results_for_charts[profile_name] = {
                    'coverage_percent': coverage_details["coverage_percent"],
                    'match_score': score,
                    'confidence': confidence,
                    'high_demand_gaps': gaps["high_priority"][:10],
                    'covered_skills': student_skills
                }

                # === АРХИВ РЕЗУЛЬТАТОВ ДЛЯ СРАВНЕНИЯ ===
                all_profiles_results.append({
                    'profile': profile_name,
                    'score': round(score, 4),
                    'confidence': round(confidence, 4),
                    'coverage': round(coverage_details["coverage_percent"], 2),
                    'high_gaps': len(gaps['high_priority']),
                    'medium_gaps': len(gaps['medium_priority'])
                })

                # === ИТОГОВАЯ СТРОКА ===
                logger.info("")
                logger.info(f"✅ ИТОГ для {profile_name.upper()}")
                logger.info(f"   Score: {score:.4f} | Confidence: {confidence:.4f} | Coverage: {coverage_details['coverage_percent']:.1f}%")

            # === СРАВНЕНИЕ ВСЕХ ПРОФИЛЕЙ ===
            if all_profiles_results:
                logger.info("\n" + "="*85)
                logger.info("СРАВНЕНИЕ ВСЕХ ПРОФИЛЕЙ")
                logger.info("="*85)
                
                # Сортируем по score
                sorted_profiles = sorted(all_profiles_results, key=lambda x: x['score'], reverse=True)
                
                logger.info(f"\nРейтинг профилей:")
                for rank, profile in enumerate(sorted_profiles, 1):
                    logger.info(
                        f"  {rank}. {profile['profile'].upper():10s} - "
                        f"Score: {profile['score']:.4f}, "
                        f"Coverage: {profile['coverage']:.1f}%, "
                        f"Gaps: {profile['high_gaps']} high/{profile['medium_gaps']} med"
                    )
                
                best_profile = sorted_profiles[0]
                worst_profile = sorted_profiles[-1]
                avg_coverage = sum(p['coverage'] for p in all_profiles_results) / len(all_profiles_results)
                
                logger.info(f"\n📊 Статистика по профилям:")
                logger.info(f"  - Лучший: {best_profile['profile'].upper()} (score: {best_profile['score']:.4f})")
                logger.info(f"  - Худший: {worst_profile['profile'].upper()} (score: {worst_profile['score']:.4f})")
                logger.info(f"  - Средний coverage: {avg_coverage:.1f}%")
                
                # Сохраняем итоговую сводку
                summary_path = config.DATA_PROCESSED_DIR / "profiles_comparison_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "timestamp": str(time.strftime('%Y-%m-%d %H:%M:%S')),
                        "total_profiles_analyzed": len(all_profiles_results),
                        "profiles": all_profiles_results,
                        "best_profile": best_profile,
                        "average_coverage": round(avg_coverage, 2)
                    }, f, ensure_ascii=False, indent=2)
                logger.info(f"\n✓ Сводка сохранена: {summary_path}")

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