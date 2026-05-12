#!/usr/bin/env python
"""
main.py — Полный пайплайн анализа вакансий и генерации персонализированных рекомендаций
Исправленная версия с LTR-движком без data leakage
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__" and sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

import structlog

from src import config
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.profile_evaluator import ProfileEvaluator
from src.analyzers.skill_filter import SkillFilter
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer
from src.analyzers.trends import TrendAnalyzer
from src.artifacts import ArtifactManifest
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.logging_config import setup_structlog
from src.models.enums import ComparisonLevel, ExperienceLevel
from src.models.student import StudentProfile, merge_skills_hierarchically
from src.models.vacancy import Vacancy
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.utils import (
    collect_vacancies_multiple,
    filter_skills_by_whitelist,
    interactive_config,
    load_it_skills,
    load_queries_from_file,
    map_to_competencies,
    print_top_competencies,
    print_top_skills,
)
from src.parsing.vacancy_parser import VacancyParser
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine
from src.predictors.recommendation_engine import RecommendationEngine
from src.utils import (
    atomic_write_json,
    load_competency_mapping,
    safe_load_pickle,
    safe_read_competency_json,
    safe_read_json,
    validate_safe_path,
)
from src.visualization.charts import run_notebook, save_all_charts, show_context_info

logger = structlog.get_logger("main")


def console_info(msg: str):
    """Человекочитаемое сообщение в консоль."""
    print(f"  {msg}")


def console_header(msg: str):
    """Заголовок этапа в консоли."""
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def save_detailed_vacancies(vacancies, log):
    """Сохраняет детальные вакансии в data/result/ для повторного использования."""
    detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    config.DATA_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = []
    for v in vacancies:
        if isinstance(v, Vacancy):
            data_to_save.append(v.raw_data)
        else:
            data_to_save.append(v)
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    log.info("detailed_vacancies_saved", path=str(detailed_file), count=len(vacancies))


def convert_float32(obj):
    import numpy as np

    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [convert_float32(item) for item in obj]
    return obj


def parse_arguments():
    parser = argparse.ArgumentParser(description="Полный пайплайн: сбор вакансий + gap-анализ + рекомендации")

    parser.add_argument("--query", "-q", type=str, default="Python developer")
    parser.add_argument("--area-id", "-a", type=int, default=1)
    parser.add_argument("--max-pages", "-p", type=int, default=10)
    parser.add_argument("--period", "-d", type=int, default=30)
    parser.add_argument("--show-vacancies", "-v", action="store_true")
    parser.add_argument("--skip-details", "-s", action="store_true")
    parser.add_argument("--excel", "-e", action="store_true")
    parser.add_argument("--no-filter", "-nf", action="store_true")
    parser.add_argument("--queries-file", "-qf", type=str)
    parser.add_argument("--regions", "-r", type=str)
    parser.add_argument("--industry", "-i", type=int)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-vacancies-per-query", type=int, default=1000)
    parser.add_argument("--it-sector", action="store_true")

    parser.add_argument("--use-async", action="store_true", default=True)
    parser.add_argument("--async-workers", type=int, default=3)
    parser.add_argument("--async-threshold", type=int, default=10000)

    parser.add_argument("--run-gap-analysis", action="store_true", default=False)
    parser.add_argument("--run-notebooks", action="store_true")
    parser.add_argument("--status", action="store_true", help="Показать состояние файлов и моделей")

    parser.add_argument("--train-model", action="store_true", help="Обучить LTR-модель на текущих данных и выйти")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Использовать LLM (YandexGPT) для живых объяснений рекомендаций",
    )
    parser.add_argument(
        "--skip-collection", action="store_true", help="Пропустить сбор вакансий, использовать существующие файлы"
    )
    return parser.parse_args()


def validate_args(args) -> None:
    """Проверяет совместимость аргументов и выводит понятные ошибки."""
    errors = []

    # --train-model требует наличия файлов вакансий
    if args.train_model:
        detailed_ok = (config.DATA_RESULT_DIR / "hh_vacancies_detailed.json").exists()
        basic_ok = (config.DATA_RAW_DIR / "hh_vacancies_basic.json").exists()
        if not detailed_ok and not basic_ok:
            errors.append(
                "Для --train-model нужен файл вакансий (hh_vacancies_detailed.json или hh_vacancies_basic.json). "
                "Сначала выполните сбор данных."
            )

    # --skip-collection и отсутствие выходных файлов
    if args.skip_collection and not args.train_model:
        detailed_exists = (config.DATA_RESULT_DIR / "hh_vacancies_detailed.json").exists()
        basic_exists = (config.DATA_RAW_DIR / "hh_vacancies_basic.json").exists()
        if not detailed_exists and not basic_exists:
            errors.append(
                "--skip-collection указан, но нет файлов вакансий. Сначала выполните сбор или уберите этот флаг."
            )

    # --use-llm без ключей
    if args.use_llm and (not config.YC_API_KEY or not config.YC_FOLDER_ID):
        errors.append("Для --use-llm необходимо задать YC_API_KEY и YC_FOLDER_ID в .env или переменных окружения.")

    # Прочие проверки можно добавить при необходимости

    if errors:
        for msg in errors:
            print(f"❌ Ошибка: {msg}")
        sys.exit(1)


def load_student_competencies(profile_name: str):
    path = config.DATA_DIR / "students" / f"{profile_name}_competency.json"
    codes = safe_read_competency_json(path)
    if not codes:
        # попробовать альтернативный файл (как было раньше)
        alt_path = config.DATA_DIR / "students" / f"{profile_name}.json"
        codes = safe_read_competency_json(alt_path)
    return codes


def calculate_expected_vacancies(args) -> int:
    if args.it_sector:
        return 11 * 500
    num_queries = 1
    if args.queries_file:
        try:
            with open(args.queries_file, encoding="utf-8") as f:
                num_queries = len([line for line in f if line.strip()])
        except Exception:
            num_queries = 1
    return num_queries * min(args.max_vacancies_per_query, 1000)


def get_load_mode(total_vacancies: int, args, log) -> tuple:
    threshold = args.async_threshold
    if not args.use_async:
        return False, 0, "async_disabled_by_user"
    if total_vacancies > threshold:
        console_info(f"⚠️  Ожидается {total_vacancies} вакансий — переключение на синхронную загрузку")
        log.warning("rate_limit_protection_activated", expected=total_vacancies, threshold=threshold)
        return False, 0, "sync_mode_large_volume"
    console_info(f"✓ Асинхронная загрузка ({args.async_workers} воркеров)")
    log.info("async_mode_activated", workers=args.async_workers)
    return True, args.async_workers, "async_mode"


def load_vacancies_details(
    basic_vacancies: list, hh_api: HeadHunterAPI, use_async: bool, async_workers: int, parser: VacancyParser, log
) -> list:
    console_info("Загрузка детальной информации по вакансиям...")
    log.info("loading_vacancy_details_started")

    if use_async:
        try:
            from src.parsing.hh_api_async import HeadHunterAPIAsync

            api_async = HeadHunterAPIAsync(
                max_concurrent=async_workers,
                request_delay=config.REQUEST_DELAY,
                token=hh_api._token,
                token_expires_at=hh_api._token_expires_at,
            )

            vacancy_ids = [v.get("id") if isinstance(v, dict) else v.id for v in basic_vacancies]

            start_time = time.time()

            if config.PYDANTIC_VALIDATION_ENABLED:
                detailed_responses = api_async.get_vacancies_details_sync_validated(vacancy_ids)
                detailed = [Vacancy.from_api(r.model_dump()) for r in detailed_responses]
            else:
                raw_detailed = api_async.get_vacancies_details_sync(vacancy_ids)
                detailed = []
                for raw_data in raw_detailed:
                    try:
                        detailed.append(Vacancy.from_api(raw_data))
                    except ValueError:
                        continue

            elapsed = time.time() - start_time
            console_info(f"✓ Загружено {len(detailed)}/{len(vacancy_ids)} вакансий за {elapsed:.1f} сек")
            log.info("async_loading_completed", elapsed=round(elapsed, 1), loaded=len(detailed), total=len(vacancy_ids))
            return detailed

        except Exception as e:
            console_info(f"⚠️  Ошибка асинхронной загрузки: {e}")
            log.warning("async_loading_failed_fallback_to_sync", error=str(e))

    detailed = []
    total = len(basic_vacancies)
    start_time = time.time()

    for _, vac in tqdm(enumerate(basic_vacancies, 1), total=total, desc="Загрузка вакансий"):
        vac_id = vac.get("id") if isinstance(vac, dict) else vac.id

        if config.PYDANTIC_VALIDATION_ENABLED:
            try:
                validated = hh_api.get_vacancy_details_validated(vac_id)
                det = Vacancy.from_api(validated.model_dump())
            except Exception:
                det = None
        else:
            det = hh_api.get_vacancy_details_as_object(vac_id)

        if det:
            detailed.append(det)
        time.sleep(config.REQUEST_DELAY)

    elapsed = time.time() - start_time
    console_info(f"✓ Загружено {len(detailed)}/{total} вакансий за {elapsed / 60:.1f} мин")
    log.info("sync_loading_completed", elapsed=round(elapsed / 60, 1), loaded=len(detailed), total=total)
    return detailed


def get_file_hash(filepath: Path) -> str:
    hash_sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()


def show_status():
    """Выводит сводку о наличии ключевых файлов и моделей."""
    from src.artifacts import ArtifactManifest

    console_info("📋 СТАТУС ПРОЕКТА")
    # Файлы вакансий
    for fname, desc in [
        ("hh_vacancies_detailed.json", "Детальные вакансии"),
        ("hh_vacancies_basic.json", "Базовые вакансии"),
    ]:
        path = config.DATA_PROCESSED_DIR / fname if "detailed" in fname else config.DATA_RAW_DIR / fname
        if path.exists():
            console_info(f"✅ {desc}: {path}")
        else:
            console_info(f"❌ {desc}: ОТСУТСТВУЕТ")
    # Кэш парсинга
    cache = config.PARSED_SKILLS_CACHE_PATH
    if cache.exists():
        console_info(f"✅ Кэш парсинга: {cache}")
    else:
        console_info("❌ Кэш парсинга: ОТСУТСТВУЕТ")
    # Модель LTR
    model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
    if model_path.exists():
        manifest = model_path.with_suffix(".manifest.json")
        if manifest.exists():
            try:
                m = ArtifactManifest.load(model_path)
                console_info(f"✅ LTR-модель: {model_path}, R²={m.metrics.get('r2', '?')}")
            except Exception:
                console_info(f"✅ LTR-модель: {model_path}, манифест повреждён")
        else:
            console_info(f"✅ LTR-модель: {model_path} (без манифеста)")
    else:
        console_info("❌ LTR-модель: ОТСУТСТВУЕТ")
    # Кластеры
    for lvl in ["junior", "middle", "senior"]:
        cluster_file = config.VACANCY_CLUSTERS_CACHE_DIR / f"vacancy_clusters_{lvl}.pkl"
        if cluster_file.exists():
            console_info(f"✅ Кластеры {lvl}: {cluster_file}")
        else:
            console_info(f"❌ Кластеры {lvl}: ОТСУТСТВУЮТ")
    # Профили студентов
    for profile in ["base", "dc", "top_dc"]:
        student_file = config.STUDENTS_DIR / f"{profile}_competency.json"
        if student_file.exists():
            console_info(f"✅ Профиль {profile}: {student_file}")
        else:
            console_info(f"❌ Профиль {profile}: ОТСУТСТВУЕТ")
    console_info("")


def main():
    setup_structlog()
    logger = structlog.get_logger("main")
    args = parse_arguments()
    validate_args(args)
    console_header("ПОЛНЫЙ ПАЙПЛАЙН: СБОР ВАКАНСИЙ + GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ")
    logger.info("pipeline_started", mode="train_model" if args.train_model else "full_pipeline")

    # ====================== Статус (информация о моделях, файлах) ======================
    if args.status:
        show_status()
        return
    # ====================== ОБУЧЕНИЕ LTR-МОДЕЛИ ======================
    if args.train_model:
        console_header("ОБУЧЕНИЕ LTR-МОДЕЛИ")

        detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
        basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        if detailed_file.exists():
            raw_file = detailed_file
        elif basic_file.exists():
            raw_file = basic_file
            console_info("⚠️  Детальный файл не найден, используется базовый")
        else:
            console_info("❌ Файлы вакансий не найдены. Сначала выполните сбор.")
            sys.exit(1)

        training_vacancies = safe_read_json(raw_file)
        if not training_vacancies:
            console_info("❌ Не удалось прочитать или файл повреждён.")
            sys.exit(1)

        console_info(f"Загружено {len(training_vacancies)} вакансий для обучения")
        logger.info("training_data_loaded", count=len(training_vacancies))

        ltr_engine = LTRRecommendationEngine()
        ltr_engine.fit(training_vacancies)
        model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        if model_path.exists():
            console_info("⚠️  Существующая модель будет перезаписана.")
            logger.warning("overwriting_existing_ltr_model", path=str(model_path))
        else:
            console_info("Новая модель будет обучена и сохранена.")
        if hasattr(ltr_engine, "last_metrics"):
            m = ltr_engine.last_metrics
            console_info(f"R²={m['r2']:.4f}, MAE={m['mae']:.4f}, NDCG@5={m['ndcg']:.4f}")
        if not ltr_engine.is_fitted:
            console_info("❌ Обучение не удалось (недостаточно навыков)")
            return
        if ltr_engine.is_fitted and hasattr(ltr_engine, "last_metrics"):
            m = ltr_engine.last_metrics
            console_info(f"R²={m['r2']:.4f}, MAE={m['mae']:.4f}, NDCG@5={m['ndcg']:.4f}")
        console_info("✅ Обучение LTR-модели завершено")
        console_info(f"Модель сохранена в: {ltr_engine.model_path}")
        return

    use_multiple = (
        args.interactive
        or args.queries_file is not None
        or args.regions is not None
        or args.industry is not None
        or args.it_sector
    )
    trend_analyzer = None
    vacancies_to_process = []
    skill_freq = {}
    hybrid_weights = {}
    level_vacancies_data = []
    vacancies_skills = []

    # ====================== ПРОПУСК СБОРА, ЕСЛИ УКАЗАН --skip-collection ======================
    if args.skip_collection:
        console_header("ЗАГРУЗКА СУЩЕСТВУЮЩИХ ДАННЫХ")

        detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
        basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        if detailed_file.exists():
            raw_file = detailed_file
        elif basic_file.exists():
            raw_file = basic_file
            console_info("⚠️  Детальный файл не найден, используется базовый (без key_skills)")
        else:
            console_info("❌ Файлы вакансий не найдены. Сначала выполните сбор.")
            sys.exit(1)

        console_info(f"Загружаем вакансии из {raw_file}...")
        basic_vacancies = safe_read_json(raw_file)
        if not basic_vacancies:
            console_info("❌ Не удалось прочитать файл вакансий.")
            sys.exit(1)

        parser = VacancyParser()
        cache_dir = config.DATA_PROCESSED_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = config.PARSED_SKILLS_CACHE_PATH
        vacancies_hash = get_file_hash(raw_file)

        cached_result = None
        if cache_path.exists():
            manifest_path = cache_path.with_suffix(".manifest.json")
            # Проверяем манифест на совместимость версий
            if manifest_path.exists():
                try:
                    manifest = ArtifactManifest.load(cache_path)
                    if not manifest.is_compatible():
                        logger.warning("parsed_skills_cache_incompatible_manifest")
                        cache_path.unlink()
                        manifest_path.unlink()
                        cached_result = None
                except Exception as e:
                    logger.warning("parsed_skills_manifest_check_failed", error=str(e))
            # Если манифест в порядке или его не было, пробуем загрузить кэш
            if cached_result is None and cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached = safe_load_pickle(cache_path)
                    if cached.get("source_hash") == vacancies_hash:
                        console_info("✅ Загружен кэш результатов парсинга навыков")
                        cached_result = cached["result"]
                    else:
                        console_info("Файл вакансий изменился, кэш недействителен")
                except Exception as e:
                    console_info(f"⚠️  Не удалось загрузить кэш: {e}")

        if cached_result is not None:
            result = cached_result
            skill_freq = result["frequencies"]
            hybrid_weights = result.get("hybrid_weights", {})
        else:
            console_info("Извлечение навыков из вакансий...")
            result = parser.extract_skills_from_vacancies(basic_vacancies)
            skill_freq = result["frequencies"]
            hybrid_weights = result.get("hybrid_weights", {})

            cache_data = {"source_hash": vacancies_hash, "result": result}
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                console_info("💾 Кэш результатов сохранён")
                # Создаём манифест
                manifest = ArtifactManifest(
                    artifact_path=cache_path,
                    metrics={"num_skills": len(skill_freq)},
                )
                manifest.save()
            console_info("💾 Кэш результатов сохранён")

        filter_engine = SkillFilter()
        competency_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
        competency_freq = {}
        if competency_freq_path.exists():
            with open(competency_freq_path, encoding="utf-8") as f:
                competency_freq = json.load(f)

        hybrid_weights = filter_engine.get_clean_weights(
            hybrid_weights, competency_freq=competency_freq, use_reference=True
        )
        console_info(f"📊 После фильтрации: {len(hybrid_weights)} навыков")

        if not skill_freq:
            console_info("❌ Не удалось извлечь навыки")
            sys.exit(1)

        console_info(f"Извлечено {len(skill_freq)} уникальных навыков")
        # Инициализация анализатора трендов (даже если нет сбора)
        whitelist = load_it_skills()
        skill_freq_for_trends = filter_skills_by_whitelist(skill_freq, whitelist) if whitelist else skill_freq
        trend_analyzer = TrendAnalyzer(skill_freq_for_trends)
        console_info(f"📈 Тренды загружены (исторических снимков: {len(trend_analyzer.load_all_snapshots())})")

        for vac in basic_vacancies:
            vac_skills = []
            if "extracted_skills" in vac:
                vac_skills = vac["extracted_skills"]
            else:
                desc = vac.get("description", "")
                snippet = vac.get("snippet") or {}
                req = snippet.get("requirement", "")
                resp = snippet.get("responsibility", "")
                combined = f"{desc} {req} {resp}"
                vac_skills = parser.extract_skills_from_description(combined)
            vac_copy = vac.copy()
            vac_copy["extracted_skills"] = vac_skills
            vacancies_to_process.append(vac_copy)

            experience = ExperienceLevel.MIDDLE
            if "experience" in vac:
                exp_obj = vac["experience"]
                if isinstance(exp_obj, dict):
                    exp_id = exp_obj.get("id", "").lower()
                    if "less1" in exp_id or "junior" in exp_id or "no_experience" in exp_id:
                        experience = ExperienceLevel.JUNIOR
                    elif "between1and3" in exp_id or "between3and6" in exp_id:
                        experience = ExperienceLevel.MIDDLE
                    elif "between6and10" in exp_id or "morethan10" in exp_id:
                        experience = ExperienceLevel.SENIOR
                elif isinstance(exp_obj, str):
                    exp_lower = exp_obj.lower()
                    if "junior" in exp_lower or "нет опыта" in exp_lower or "стажер" in exp_lower:
                        experience = ExperienceLevel.JUNIOR
                    elif "senior" in exp_lower or "более 6" in exp_lower:
                        experience = ExperienceLevel.SENIOR

            if experience == ExperienceLevel.MIDDLE:
                name = vac.get("name", "").lower()
                if "junior" in name or "младший" in name or "стажер" in name or "intern" in name:
                    experience = ExperienceLevel.JUNIOR
                elif "senior" in name or "старший" in name or "ведущий" in name:
                    experience = ExperienceLevel.SENIOR

            if vac_skills:
                level_vacancies_data.append(
                    {"skills": vac_skills, "description": vac.get("description", ""), "experience": experience}
                )
                vacancies_skills.append(vac_skills)

        console_info(f"Подготовлено {len(level_vacancies_data)} вакансий для анализа уровней")

        level_analyzer = SkillLevelAnalyzer()
        level_analyzer.analyze_vacancies(level_vacancies_data)

        args.run_gap_analysis = True

    else:
        # ====================== 1. СБОР ВАКАНСИЙ ======================
        console_header("СБОР ВАКАНСИЙ С HH.RU")

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
                        "Data Scientist",
                        "Data Analyst",
                        "Machine Learning Engineer",
                        "Computer Vision Engineer",
                        "NLP Engineer",
                        "Data Architect",
                        "ETL Developer",
                        "Python Developer",
                        "Java Developer",
                        "Frontend Developer",
                        "Backend Developer",
                        "Fullstack Developer",
                        "DevOps Engineer",
                        "Embedded Developer",
                        "Blockchain Developer",
                        "iOS Developer",
                        "Android Developer",
                        "React Native Developer",
                        "Flutter Developer",
                        "QA Engineer",
                        "Automation QA Engineer",
                        "Performance QA Engineer",
                        "Специалист по кибербезопасности",
                        "Security Engineer",
                        "DevSecOps Engineer",
                        "SRE инженер",
                        "Системный администратор",
                        "Облачный инженер",
                        "Сетевой инженер",
                        "Администратор баз данных",
                        "Системный аналитик",
                        "Бизнес-аналитик",
                        "Архитектор программного обеспечения",
                        "Solution Architect",
                        "Team Lead",
                        "Tech Lead",
                        "Project Manager IT",
                        "Scrum Master",
                        "UX/UI дизайнер",
                        "Product Designer",
                        "Unity Developer",
                        "Unreal Engine Developer",
                        "Technical Writer",
                    ]
                    args.industry = 7
                    args.max_vacancies_per_query = 100000
                    console_info("Режим: поиск по всему IT-сектору (40+ профессий)")
                elif args.queries_file:
                    safe_path = validate_safe_path(args.queries_file)
                    args.queries = load_queries_from_file(safe_path)
                else:
                    args.queries = [args.query]

                if args.regions:
                    args.area_ids = [int(x.strip()) for x in args.regions.split(",")]
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
                max_vacancies_per_query=args.max_vacancies_per_query,
            )

            if not basic_vacancies:
                console_info("❌ Не найдено вакансий.")
                return

            console_info(f"Найдено {len(basic_vacancies)} базовых вакансий")
            parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json")

            if args.skip_details:
                vacancies_to_process = basic_vacancies
                console_info("Пропуск загрузки деталей (--skip-details)")
            else:
                total_vacs = len(basic_vacancies)
                use_async, async_workers, reason = get_load_mode(total_vacs, args, logger)
                vacancies_to_process = load_vacancies_details(
                    basic_vacancies=basic_vacancies,
                    hh_api=hh_api,
                    use_async=use_async,
                    async_workers=async_workers,
                    parser=parser,
                    log=logger,
                )
                save_detailed_vacancies(vacancies_to_process, logger)

            if args.show_vacancies:
                parser.print_vacancies_list(vacancies_to_process)

        else:
            hh_api = HeadHunterAPI()
            parser = VacancyParser()

            console_info(f"Поиск: '{args.query}', регион {args.area_id}")
            basic_vacancies = hh_api.search_vacancies(
                text=args.query, area=args.area_id, period_days=args.period, max_pages=args.max_pages
            )

            if not basic_vacancies:
                console_info("❌ Не найдено вакансий.")
                return

            console_info(f"Найдено {len(basic_vacancies)} вакансий")
            parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json")

            if args.skip_details:
                vacancies_to_process = basic_vacancies
            else:
                total_vacs = len(basic_vacancies)
                use_async, async_workers, reason = get_load_mode(total_vacs, args, logger)
                vacancies_to_process = load_vacancies_details(
                    basic_vacancies=basic_vacancies,
                    hh_api=hh_api,
                    use_async=use_async,
                    async_workers=async_workers,
                    parser=parser,
                    log=logger,
                )
                save_detailed_vacancies(vacancies_to_process, logger)

        # ====================== 2. ОБРАБОТКА НАВЫКОВ ======================
        console_header("ИЗВЛЕЧЕНИЕ И ВАЛИДАЦИЯ НАВЫКОВ")

        result = parser.extract_skills_from_vacancies(vacancies_to_process)
        skill_freq = result["frequencies"]
        hybrid_weights_raw = result.get("hybrid_weights", {})

        filter_engine = SkillFilter()
        competency_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
        competency_freq = {}
        if competency_freq_path.exists():
            with open(competency_freq_path, encoding="utf-8") as f:
                competency_freq = json.load(f)

        hybrid_weights = filter_engine.get_clean_weights(
            hybrid_weights_raw, competency_freq=competency_freq, use_reference=True
        )

        console_info(f"Извлечено {len(skill_freq)} уникальных навыков")
        console_info(f"Гибридных весов после фильтрации: {len(hybrid_weights)}")

        parser.save_processed_frequencies(skill_freq, apply_filter=not args.no_filter)
        print_top_skills(skill_freq)

        whitelist = load_it_skills()
        if whitelist:
            skill_freq_filtered = filter_skills_by_whitelist(skill_freq, whitelist)
            console_info(f"После фильтрации по белому списку: {len(skill_freq_filtered)} навыков")

        trend_analyzer = TrendAnalyzer(skill_freq_filtered if whitelist else skill_freq)
        trend_analyzer.save_snapshot(skill_freq_filtered if whitelist else skill_freq, apply_whitelist=False)
        console_info("📸 Снимок рынка сохранён")

        if hybrid_weights:
            print("\n" + "=" * 80)
            print("ТОП-15 НАВЫКОВ ПО ГИБРИДНОМУ ВЕСУ (BM25 + Embeddings)")
            print("=" * 80)
            top_weights = sorted(hybrid_weights.items(), key=lambda x: x[1], reverse=True)[:15]
            for i, (skill, weight) in enumerate(top_weights, 1):
                print(f"{i:2}. {skill:<40} {weight:.4f}")

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
                    with open(comp_freq_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_comp, f, ensure_ascii=False, indent=2)
                    print_top_competencies(comp_counter)
        except Exception as e:
            logger.exception("competency_mapping_error", error=str(e))

        if args.excel:
            df = parser.aggregate_to_dataframe(vacancies_to_process)
            if not df.empty:
                filename = (
                    "vacancies_it_sector.xlsx"
                    if getattr(args, "it_sector", False)
                    else f"vacancies_{args.query.replace(' ', '_')}.xlsx"
                )
                parser.save_to_excel(df, filename)
                console_info(f"✅ Excel сохранён: {filename}")

        console_info("Инициализация анализатора уровней...")

        for vac in tqdm(vacancies_to_process, desc="Разбор вакансий по уровням"):
            if isinstance(vac, Vacancy):
                vac_skills = []
                if hasattr(vac, "key_skills") and vac.key_skills:
                    vac_skills = [s.name if hasattr(s, "name") else str(s) for s in vac.key_skills]
                elif hasattr(vac, "extracted_skills") and vac.extracted_skills:
                    vac_skills = vac.extracted_skills

                vac_experience = ExperienceLevel.MIDDLE
                if hasattr(vac, "experience") and vac.experience:
                    exp_obj = vac.experience
                    if hasattr(exp_obj, "id"):
                        exp_id = exp_obj.id.lower()
                        if "less1" in exp_id or "junior" in exp_id or "no_experience" in exp_id:
                            vac_experience = ExperienceLevel.JUNIOR
                        elif "between1and3" in exp_id or "between3and6" in exp_id:
                            vac_experience = ExperienceLevel.MIDDLE
                        elif "between6and10" in exp_id or "morethan10" in exp_id:
                            vac_experience = ExperienceLevel.SENIOR
                    elif isinstance(exp_obj, str):
                        exp_lower = exp_obj.lower()
                        if "junior" in exp_lower or "нет опыта" in exp_lower or "стажер" in exp_lower:
                            vac_experience = ExperienceLevel.JUNIOR
                        elif "senior" in exp_lower or "более 6" in exp_lower:
                            vac_experience = ExperienceLevel.SENIOR
                        else:
                            vac_experience = ExperienceLevel.MIDDLE

                if vac_experience == "middle":
                    name = vac.name.lower() if hasattr(vac, "name") else ""
                    if "junior" in name or "младший" in name or "стажер" in name or "intern" in name:
                        vac_experience = ExperienceLevel.JUNIOR
                    elif "senior" in name or "старший" in name or "ведущий" in name:
                        vac_experience = ExperienceLevel.SENIOR

                if vac_skills:
                    level_vacancies_data.append(
                        {"skills": vac_skills, "description": vac.description or "", "experience": vac_experience}
                    )
            else:
                vac_skills = [s["name"] for s in vac.get("key_skills", [])]
                if vac_skills:
                    experience = "middle"
                    exp_obj = vac.get("experience", {})
                    if isinstance(exp_obj, dict):
                        exp_id = exp_obj.get("id", "").lower()
                        if "less1" in exp_id or "junior" in exp_id or "no_experience" in exp_id:
                            experience = ExperienceLevel.JUNIOR
                        elif "between1and3" in exp_id or "between3and6" in exp_id:
                            experience = ExperienceLevel.MIDDLE
                        elif "between6and10" in exp_id or "morethan10" in exp_id:
                            experience = ExperienceLevel.SENIOR
                    elif isinstance(exp_obj, str):
                        exp_lower = exp_obj.lower()
                        if "junior" in exp_lower or "нет опыта" in exp_lower or "стажер" in exp_lower:
                            experience = ExperienceLevel.JUNIOR
                        elif "senior" in exp_lower or "более 6" in exp_lower:
                            experience = ExperienceLevel.SENIOR
                    if experience == ExperienceLevel.MIDDLE:
                        name = vac.get("name", "").lower()
                        if "junior" in name or "младший" in name or "стажер" in name or "intern" in name:
                            experience = ExperienceLevel.JUNIOR
                        elif "senior" in name or "старший" in name or "ведущий" in name:
                            experience = ExperienceLevel.SENIOR
                    level_vacancies_data.append(
                        {"skills": vac_skills, "description": vac.get("description", ""), "experience": experience}
                    )

        console_info(f"Подготовлено {len(level_vacancies_data)} вакансий для анализа уровней")

    # ====================== 3. GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ ======================
    if args.run_gap_analysis:
        console_header("GAP-АНАЛИЗ И ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ")

        try:
            if not skill_freq:
                console_info("❌ Нет навыков для анализа")
                return

            competency_mapping = load_competency_mapping()
            if not competency_mapping:
                console_info("⚠️  Маппинг компетенций не загружен")
            else:
                console_info(f"Загружен маппинг для {len(competency_mapping)} компетенций")

            def map_codes_to_skills(codes):
                if not competency_mapping:
                    return codes
                skills = set()
                for code in codes:
                    code_norm = "".join(c for c in code if c.isalnum()).upper()
                    for key, value in competency_mapping.items():
                        key_norm = "".join(c for c in key if c.isalnum()).upper()
                        if code_norm == key_norm:
                            skills.update(value)
                            break
                return list(skills)

            if not vacancies_skills:
                for vac_data in level_vacancies_data:
                    skills = vac_data.get("skills", [])
                    if skills:
                        vacancies_skills.append(skills)

            console_info(f"Подготовлено {len(vacancies_skills)} вакансий")

            if not vacancies_skills:
                console_info("❌ Не удалось подготовить данные")
                return

            skill_weights = hybrid_weights

            if not skill_weights:
                console_info("❌ skill_weights пусты после фильтрации")
                return

            console_info(f"Используем {len(skill_weights)} очищенных весов навыков")

            weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
            atomic_write_json(skill_weights, weights_path)

            level_analyzer = SkillLevelAnalyzer()
            level_analyzer.analyze_vacancies(level_vacancies_data)

            all_codes = {}
            for name in ["base", "dc", "top_dc"]:
                codes = load_student_competencies(name)
                if codes:
                    all_codes[name] = codes
                else:
                    console_info(f"⚠️  Профиль {name} не загружен")

            profiles: dict[str, StudentProfile] = {}
            profile_levels = {
                "base": ExperienceLevel.JUNIOR,
                "dc": ExperienceLevel.MIDDLE,
                "top_dc": ExperienceLevel.SENIOR,
            }

            for profile_name, target_level_str in profile_levels.items():
                if profile_name not in all_codes:
                    continue

                if profile_name == "top_dc":
                    top_codes = all_codes.get("top_dc", [])
                    dc_codes = all_codes.get("dc", [])
                    base_codes = all_codes.get("base", [])

                    top_skills = map_codes_to_skills(top_codes)
                    dc_skills = map_codes_to_skills(dc_codes)
                    base_skills = map_codes_to_skills(base_codes)

                    student_skills = merge_skills_hierarchically(top_skills, dc_skills, base_skills)
                    student_codes = top_codes
                else:
                    student_codes = all_codes[profile_name]
                    student_skills = map_codes_to_skills(student_codes)

                student_skills = [SkillNormalizer.normalize(s) for s in student_skills if SkillNormalizer.normalize(s)]
                student_skills = list(dict.fromkeys(student_skills))
                profiles[profile_name] = StudentProfile(
                    profile_name=profile_name,
                    competencies=student_codes,
                    skills=student_skills,
                    target_level=target_level_str,
                )

            skill_weights_by_level = {}
            for level in ExperienceLevel:
                skill_weights_by_level[level] = level_analyzer.get_weights_for_level(skill_weights, level)

            console_info("Оценка профилей...")

            evaluator = ProfileEvaluator(
                skill_weights=skill_weights,
                vacancies_skills=vacancies_skills,
                vacancies_skills_dict=level_vacancies_data,
                hybrid_weights=hybrid_weights,
                skill_weights_by_level=skill_weights_by_level,
            )

            recommendation_engine = RecommendationEngine(
                use_ltr=True, use_llm=args.use_llm, profile_evaluator=evaluator, trend_analyzer=trend_analyzer
            )
            recommendation_engine.comparator = CompetencyComparator(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                use_embeddings=True,
                level=ComparisonLevel.MIDDLE,
                similarity_threshold=0.80,
            )
            recommendation_engine.fit(vacancies_skills, skill_weights=hybrid_weights)

            evaluations_new = {}
            with tqdm(total=len(profiles), desc="Оценка профилей") as pbar:
                for profile_name, student in profiles.items():
                    console_info(f"  Оценка профиля {profile_name}...")
                    eval_result = evaluator.evaluate_profile(student, user_type="student")
                    evaluations_new[profile_name] = eval_result
                    pbar.update(1)

            # Вывод сводки метрик
            print(f"\n{'=' * 70}")
            print("  СВОДКА МЕТРИК ПО ПРОФИЛЯМ")
            print(f"{'=' * 70}")
            for profile_name, eval_data in evaluations_new.items():
                print(f"\n  📊 {profile_name.upper()} (целевой уровень: {profiles[profile_name].target_level}):")
                print(f"     Общее покрытие рынка: {eval_data['market_coverage_score']:.1f}%")
                print(f"     Навыковое покрытие:   {eval_data['skill_coverage']:.1f}%")
                print(f"     Доменное покрытие:    {eval_data['domain_coverage_score']:.1f}%")
                print(f"     Реальное покрытие:    {eval_data['market_skill_coverage']:.1f}%")
                print(f"     Готовность к уровню:  {eval_data['readiness_score']:.1f}%")

            readiness_scores = [ev["readiness_score"] for ev in evaluations_new.values()]
            avg_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0
            best_profile = max(evaluations_new.items(), key=lambda x: x[1]["readiness_score"])
            best_profile_name, best_eval = best_profile

            console_info(f"\nСредняя готовность: {avg_readiness:.1f}%")
            console_info(f"Лучший профиль: {best_profile_name} ({best_eval['readiness_score']:.1f}%)")

            console_header("ГЕНЕРАЦИЯ ПЕРСОНАЛИЗИРОВАННЫХ РЕКОМЕНДАЦИЙ")

            rec_engine = recommendation_engine
            if rec_engine.ltr_engine is None or not rec_engine.ltr_engine.is_fitted:
                console_info("⚠️ LTR-модель не загружена. Рекомендации будут построены только на основе анализа рынка.")
                logger.warning("ltr_model_unavailable_recommendations_without_ml")

            all_recommendations = {}
            for profile_name, student in tqdm(profiles.items(), desc="Оценка профилей"):
                try:
                    v2_result = evaluator.evaluate_profile(student, user_type="student")

                    # Проверка, что результат не None
                    if v2_result is None:
                        logger.error("evaluate_profile_returned_none", profile=profile_name)
                        continue

                    skill_weights_context = {}
                    cluster_ctx = v2_result.get("cluster_context") or {}
                    cluster_skills = cluster_ctx.get("skills", {})
                    for skill, metric_dict in v2_result.get("skill_metrics", {}).items():
                        if skill in cluster_skills:
                            weight = cluster_skills[skill]
                        else:
                            weight = metric_dict.get("cluster_relevance", 0.15)
                        skill_weights_context[skill] = weight

                    rec_engine.set_cluster_context(skill_weights_context)

                    full_rec = rec_engine.generate_recommendations(student, user_type="student")
                    if full_rec is None:
                        logger.error("generate_recommendations_returned_none", profile=profile_name)
                        continue

                    if "summary" not in full_rec:
                        full_rec["summary"] = {}
                    full_rec["summary"]["market_coverage_score"] = v2_result["market_coverage_score"]
                    full_rec["summary"]["skill_coverage"] = v2_result["skill_coverage"]
                    full_rec["summary"]["domain_coverage_score"] = v2_result["domain_coverage_score"]
                    full_rec["domain_coverage"] = v2_result.get("domain_coverage", {})
                    all_recommendations[profile_name] = full_rec
                    logger.info("recommendations_generated", profile=profile_name)
                except Exception as e:
                    logger.error("recommendation_generation_failed", profile=profile_name, error=str(e))
                    all_recommendations[profile_name] = None

            for profile_name, full_rec in all_recommendations.items():
                if not full_rec:
                    continue

                print(f"\n📌 РЕКОМЕНДАЦИИ ДЛЯ ПРОФИЛЯ '{profile_name}'")
                print("=" * 70)
                summ = full_rec.get("summary", {})
                print(f"Match score: {summ.get('match_score', 0):.2f} | Готовность: {summ.get('confidence', 0):.2f}%")
                if full_rec.get("trend_bonuses_count"):
                    print(f"📈 Трендовые бонусы применены к {full_rec['trend_bonuses_count']} навыкам.")
                if full_rec.get("dominant_domain_name"):
                    print(f"📊 Доминирующий домен: {full_rec['dominant_domain_name']}.")
                print(
                    f"Реальное покрытие рынка: {summ['market_skill_coverage']:.1f}% "
                    f"(студент знает {summ['coverage_details']['covered_skills_count']} "
                    f"из {summ['coverage_details']['total_market_skills']} востребованных навыков)"
                )

                roles = full_rec.get("closest_roles", [])
                if roles:
                    print("\n🎯 БЛИЖАЙШИЕ РОЛИ:")
                    for i, role in enumerate(roles[:3], 1):
                        print(f"  {i}. {role['role']}")
                        print(f"     Семантическая близость: {role['semantic_similarity']}%")
                        print(f"     Покрытие навыков: {role['skills_covered']} ({role['coverage_percent']}%)")
                        if i == 1 and role.get("coverage_explanation"):
                            print(f"     ℹ️  {role['coverage_explanation']}")

                print("\n📋 РЕКОМЕНДАЦИИ К ИЗУЧЕНИЮ:")
                for rec in full_rec.get("recommendations", [])[:5]:
                    print(
                        f"{rec['rank']:2}. {rec['skill']:<25} важность: {rec['importance_score']:.3f} "
                        f"({rec['priority']})"
                    )
                    print(f"    {rec['why_important']}")
                    print(f"    Как учить: {rec['how_to_learn']}")
                    print(f"    Время: {rec['expected_timeframe']}")
                    print(f"    Результат: {rec['expected_outcome']}")
                    print()

                rec_file = config.DATA_DIR / "result" / profile_name / f"full_recommendations_{profile_name}.json"
                rec_file.parent.mkdir(parents=True, exist_ok=True)
                full_rec_serializable = convert_float32(full_rec)
                atomic_write_json(full_rec_serializable, rec_file)
                console_info(f"  ✓ Рекомендации сохранены: {rec_file}")

            console_header("GAP-АНАЛИЗ УСПЕШНО ЗАВЕРШЁН")

        except Exception as e:
            logger.exception("gap_analysis_failed", error=str(e))
            import traceback

            traceback.print_exc()
            return

    show_context_info()

    if "evaluations_new" in locals():
        console_header("ГЕНЕРАЦИЯ ПРЕЗЕНТАЦИОННЫХ ГРАФИКОВ")
        output_viz_dir = config.DATA_DIR / "result"
        output_viz_dir.mkdir(parents=True, exist_ok=True)
        save_all_charts(evaluations_new, output_viz_dir, use_ml=True, vacancies_skills_list=vacancies_skills)
        console_info(f"✅ Графики сохранены в {output_viz_dir}")

    if args.run_notebooks:
        console_info("Запуск Jupyter ноутбуков...")
        run_notebook("01_hh_analysis.ipynb", output_dir=config.DATA_DIR / "notebooks")
        run_notebook("02_competency_matching.ipynb", output_dir=config.DATA_DIR / "notebooks")

    try:
        csv_path = config.DATA_RAW_DIR / "competency_matrix.csv"
        if not csv_path.exists():
            csv_path = config.DATA_DIR / "last_uploaded" / "competency_matrix.csv"
        if csv_path.exists():
            generate_profiles_from_csv(csv_path)
            console_info("✓ Профили студентов обновлены из CSV")
    except Exception as e:
        logger.warning("csv_profile_update_failed", error=str(e))

    console_header("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН")
    console_info(f"📁 Результаты: {config.DATA_DIR / 'result'}")
    console_info(f"📋 Логи: {config.LOG_FILE}")
    console_info(f"⏰ Завершено: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
