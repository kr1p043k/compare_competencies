#!/usr/bin/env python
"""
main.py — Stage-based пайплайн анализа вакансий и генерации персональных рекомендаций.
Версия 3.0 — с выделенными этапами (pipeline).
"""

import argparse
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__" and sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

import structlog

from src import Err, Ok, config, timed_block
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.logging_config import setup_structlog
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.models.enums import ExperienceLevel
from src.models.data_contracts import PipelineContext
from src.models.student import StudentProfile, merge_skills_hierarchically
from src.pipeline.helpers import console_header, console_info
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.progress import write as _write_pipeline_progress
from src.pipeline.stages import (
    ClusterTrainingStage,
    DataCollectionStage,
    GapAnalysisStage,
    LevelBuildingStage,
    ModelTrainingStage,
    QualityScoringStage,
    SkillExtractionStage,
    WeightCleaningStage,
)
from src.predictors import create_ranking_predictor, create_recommender
from src.utils import (
    atomic_write_json,
    load_competency_mapping,
    safe_read_competency_json,
    safe_read_json,
)
import matplotlib
matplotlib.use("Agg")
from src.visualization.orchestration import run_notebook, save_all_charts, show_context_info

logger = structlog.get_logger("main")




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
    parser.add_argument("--query", "-q", type=str, default="1с разработчик")
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
    parser.add_argument("--max-vacancies-per-query", type=int, default=2000)
    parser.add_argument("--it-sector", action="store_true")
    parser.add_argument("--use-async", action="store_true", default=True)
    parser.add_argument("--async-workers", type=int, default=3)
    parser.add_argument("--async-threshold", type=int, default=10000)
    parser.add_argument("--run-gap-analysis", action="store_true", default=False,
        help="Устарел. Используйте --skip-gap-analysis для пропуска")
    parser.add_argument("--run-notebooks", action="store_true")
    parser.add_argument("--skip-gap-analysis", action="store_true", default=False)
    parser.add_argument("--status", action="store_true", help="Показать состояние файлов и моделей")
    parser.add_argument("--train-model", action="store_true", help="Обучить LTR-модель на текущих данных и выйти")
    parser.add_argument("--force", action="store_true", help="Принудительное переобучение (пропускает проверки кэша)")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Использовать LLM (YandexGPT) для живых объяснений рекомендаций",
    )
    parser.add_argument(
        "--skip-collection", action="store_true", help="Пропустить сбор вакансий,использовать существующие файлы"
    )
    args = parser.parse_args()
    if args.query and args.query.startswith("b64:"):
        import base64
        args.query = base64.b64decode(args.query[4:]).decode("utf-8")
    return args


def validate_args(args) -> None:
    errors = []
    if args.train_model:
        detailed_ok = (config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json").exists()
        basic_ok = (config.DATA_RAW_DIR / "hh_vacancies_basic.json").exists()
        if not detailed_ok and not basic_ok:
            errors.append(
                "Для --train-model нужен файл вакансий "
                "(hh_vacancies_detailed.json или hh_vacancies_basic.json). Сначала выполните сбор данных."
            )
    if args.skip_collection and not args.train_model:
        detailed_exists = (config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json").exists()
        basic_exists = (config.DATA_RAW_DIR / "hh_vacancies_basic.json").exists()
        if not detailed_exists and not basic_exists:
            errors.append(
                "--skip-collection указан, но нет файлов вакансий. Сначала выполните сбор или уберите этот флаг."
            )
    if args.use_llm and (not config.YC_API_KEY or not config.YC_FOLDER_ID):
        errors.append("Для --use-llm необходимо задать YC_API_KEY и YC_FOLDER_ID в .env или переменных окружения.")
    if errors:
        for msg in errors:
            print(f"❌ Ошибка: {msg}")
        sys.exit(1)


def load_student_competencies(profile_name: str):
    path = config.DATA_DIR / "students" / f"{profile_name}_competency.json"
    codes = safe_read_competency_json(path)
    if not codes:
        alt_path = config.DATA_DIR / "students" / f"{profile_name}.json"
        codes = safe_read_competency_json(alt_path)
    return codes


def build_profiles(all_codes: dict, competency_mapping: dict) -> dict:
    profiles: dict[str, StudentProfile] = {}
    profile_levels = {
        "base": ExperienceLevel.JUNIOR,
        "dc": ExperienceLevel.MIDDLE,
        "top_dc": ExperienceLevel.SENIOR,
    }

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

    for profile_name, target_level in profile_levels.items():
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
            target_level=target_level,
        )
    return profiles


def print_recommendations(profiles, all_recommendations):
    for profile_name, full_rec in all_recommendations.items():
        if not full_rec:
            continue
        target_prof = full_rec.get("target_profession", "не задана")
        print(f"\n📌 РЕКОМЕНДАЦИИ ДЛЯ ПРОФИЛЯ '{profile_name}' (цель: {target_prof})")
        print("=" * 70)
        summ = full_rec.get("summary", {})
        print(f"Match score: {summ.get('match_score', 0):.2f} | Готовность: {summ.get('confidence', 0):.2f}%")
        prof_cov = summ.get('profession_coverage', 0)
        print(f"Покрытие по профессии: {prof_cov:.1f}%")
        if full_rec.get("trend_bonuses_count"):
            print(f"Трендовые бонусы применены к {full_rec['trend_bonuses_count']} навыкам.")
        if full_rec.get("dominant_domain_name"):
            print(f"Доминирующий домен: {full_rec['dominant_domain_name']}.")
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
            print(f"{rec['rank']:2}. {rec['skill']:<25} важность: {rec['importance_score']:.3f} ({rec['priority']})")
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


def show_status():
    from src.artifacts import ArtifactManifest

    console_info("📋 СТАТУС ПРОЕКТА")
    for fname, desc in [
        ("hh_vacancies_detailed.json", "Детальные вакансии"),
        ("hh_vacancies_basic.json", "Базовые вакансии"),
    ]:
        path = config.DATA_PROCESSED_DIR / fname if "detailed" in fname else config.DATA_RAW_DIR / fname
        if path.exists():
            console_info(f"✅ {desc}: {path}")
        else:
            console_info(f"❌ {desc}: ОТСУТСТВУЕТ")
    cache = config.PARSED_SKILLS_CACHE_PATH
    if cache.exists():
        console_info(f"✅ Кэш парсинга: {cache}")
    else:
        console_info("❌ Кэш парсинга: ОТСУТСТВУЕТ")
    model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
    if model_path.exists():
        manifest = model_path.with_suffix(".manifest.json")
        if manifest.exists():
            match ArtifactManifest.load(model_path):
                case Ok(m):
                    console_info(f"✅ LTR-модель: {model_path}, R²={m.metrics.get('r2', '?')}")
                case Err(_):
                    console_info(f"✅ LTR-модель: {model_path}, манифест повреждён")
        else:
            console_info(f"✅ LTR-модель: {model_path} (без манифеста)")
    else:
        console_info("❌ LTR-модель: ОТСУТСТВУЕТ")
    for lvl in ["junior", "middle", "senior"]:
        cluster_file = config.VACANCY_CLUSTERS_CACHE_DIR / f"vacancy_clusters_{lvl}.pkl"
        if cluster_file.exists():
            console_info(f"✅ Кластеры {lvl}: {cluster_file}")
        else:
            console_info(f"❌ Кластеры {lvl}: ОТСУТСТВУЮТ")
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

    if args.status:
        show_status()
        return

    if args.train_model:
        console_header("ОБУЧЕНИЕ LTR-МОДЕЛИ")
        detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
        basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        raw_file = detailed_file if detailed_file.exists() else basic_file
        if not raw_file.exists():
            console_info("❌ Файлы вакансий не найдены. Сначала выполните сбор.")
            sys.exit(1)
        model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        if model_path.exists() and not args.force:
            ltr_engine = create_ranking_predictor(model_path=model_path)
            if ltr_engine and ltr_engine.is_fitted:
                model_mtime = model_path.stat().st_mtime
                data_mtime = raw_file.stat().st_mtime
                if model_mtime > data_mtime:
                    console_info("✅ Модель уже обучена и актуальна, обучение пропущено")
                    return
        training_vacancies = safe_read_json(raw_file)
        if not training_vacancies:
            console_info("❌ Не удалось прочитать или файл повреждён.")
            sys.exit(1)
        console_info(f"Загружено {len(training_vacancies)} вакансий для обучения")
        logger.info("training_data_loaded", count=len(training_vacancies))
        from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine
        ltr_engine = LTRRecommendationEngine()
        match ltr_engine.fit(training_vacancies):
            case Ok(_):
                pass
            case Err(err):
                logger.error("ltr_training_failed", error=str(err))
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
        console_info("✅ Обучение LTR-модели завершено")
        console_info(f"Модель сохранена в: {ltr_engine.model_path}")
        return

    # ------------------- Основной пайплайн (PipelineOrchestrator) -------------------
    _write_pipeline_progress(0, "Инициализация...")

    stages = [
        DataCollectionStage(args),
    ]
    if not args.skip_collection:
        stages.append(QualityScoringStage(args))
    stages.extend([
        SkillExtractionStage(args),
        WeightCleaningStage(),
        LevelBuildingStage(),
        ClusterTrainingStage(),
        ModelTrainingStage(),
    ])
    show_context_info()

    orchestrator = PipelineOrchestrator(stages, num_retries=1)
    run = orchestrator.run(name="full_pipeline")

    if run.status != "completed":
        console_info(f"❌ Пайплайн не завершён: {run.stages[-1].error if run.stages else 'unknown'}")
        return

    ctx_data = {}
    for sr in run.stages:
        if sr.data:
            ctx_data.update(sr.data)

    vacancies = ctx_data.get("vacancies", [])
    parser = ctx_data.get("parser")
    raw_file = ctx_data.get("raw_file")
    skill_freq = ctx_data.get("skill_freq", {})
    hybrid_weights = ctx_data.get("hybrid_weights", {})
    trend_analyzer = ctx_data.get("trend_analyzer")
    level_data = ctx_data.get("level_data", [])
    vacancies_skills = ctx_data.get("vacancies_skills", [])

    # Профили студентов
    _write_pipeline_progress(60, "Загрузка профилей студентов...")
    with timed_block("Stage4b.student_profiles"):
        competency_mapping = load_competency_mapping()
        if not competency_mapping:
            console_info("⚠️  Маппинг компетенций не загружен")
        all_codes = {}
        for name in ["base", "dc", "top_dc"]:
            codes = load_student_competencies(name)
            if codes:
                all_codes[name] = codes
                _write_pipeline_progress(63, f"Загружен профиль {name}")
            else:
                console_info(f"⚠️  Профиль {name} не загружен")
        profiles = build_profiles(all_codes, competency_mapping)
        _write_pipeline_progress(68, f"Построено {len(profiles)} профилей")

    # Stage 5: Gap-анализ и рекомендации
    evaluations = None
    if not args.skip_gap_analysis and profiles:
        _write_pipeline_progress(72, "Инициализация GAP-анализа...")
        console_header("GAP-АНАЛИЗ И ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ")
        ctx = PipelineContext(
            skill_freq=skill_freq,
            hybrid_weights=hybrid_weights,
            vacancies_skills=vacancies_skills,
            level_vacancies_data=level_data,
            trend_analyzer=trend_analyzer,
        )
        gap_stage = GapAnalysisStage(profiles, ctx, args)
        gap_result = gap_stage.run()
        match gap_result:
            case Ok(data):
                evaluations = data.get("evaluations")
                recs = data.get("recommendations")
                _write_pipeline_progress(88, "GAP-анализ завершён")
                if recs:
                    print_recommendations(profiles, recs)
                    console_header("GAP-АНАЛИЗ УСПЕШНО ЗАВЕРШЁН")
            case Err(err):
                logger.error("gap_analysis_failed", error=str(err))
                sys.exit(1)

    _write_pipeline_progress(93, "Генерация графиков...")
    if evaluations:
        console_header("ГЕНЕРАЦИЯ ПРЕЗЕНТАЦИОННЫХ ГРАФИКОВ")
        output_viz_dir = config.REPORTS_DIR
        output_viz_dir.mkdir(parents=True, exist_ok=True)
        save_all_charts(evaluations, output_viz_dir, use_ml=True, vacancies_skills_list=vacancies_skills)
        _write_pipeline_progress(97, "Графики сохранены")

    if args.run_notebooks:
        console_info("Запуск Jupyter ноутбуков...")
        run_notebook("01_hh_analysis.ipynb", output_dir=config.DATA_DIR / "notebooks")
        run_notebook("02_competency_matching.ipynb", output_dir=config.DATA_DIR / "notebooks")

    csv_path = config.DATA_RAW_DIR / "competency_matrix.csv"
    if not csv_path.exists():
        csv_path = config.DATA_DIR / "last_uploaded" / "competency_matrix.csv"
    if csv_path.exists():
        match generate_profiles_from_csv(csv_path):
            case Ok(_):
                console_info("✓ Профили студентов обновлены из CSV")
            case Err(e):
                logger.warning("csv_profile_update_failed", error=str(e.message))

    _write_pipeline_progress(100, "Пайплайн завершён")
    console_header("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН")
    console_info(f"📁 Результаты: {config.DATA_DIR / 'result'}")
    console_info(f"📋 Логи бэкенда: {config.LOG_FILE}")
    console_info(f"📋 Логи фронтенда: frontend/logs/app.log")
    console_info(f"⏰ Завершено за {run.elapsed:.1f} сек" if run.elapsed < 120 else f"⏰ Завершено за {run.elapsed / 60:.1f} мин")


if __name__ == "__main__":
    main()
