"""Pipeline runner — reusable sync/async entry point for CLI and API."""

import asyncio
import sys
from typing import Any

import structlog
import matplotlib

matplotlib.use("Agg")

from src import Err, Ok, config, timed_block
from src.loaders_student.student_loader import generate_profiles_from_csv
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
from src.predictors import create_ranking_predictor
from src.utils import (
    atomic_write_json,
    load_competency_mapping,
    safe_read_competency_json,
    safe_read_json,
)
from src.visualization.orchestration import run_notebook, save_all_charts, show_context_info

logger = structlog.get_logger(__name__)


def convert_float32(obj):
    import numpy as np
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [convert_float32(item) for item in obj]
    return obj


def load_student_competencies(profile_name: str) -> list:
    path = config.DATA_DIR / "students" / f"{profile_name}_competency.json"
    codes = safe_read_competency_json(path)
    if not codes:
        alt_path = config.DATA_DIR / "students" / f"{profile_name}.json"
        codes = safe_read_competency_json(alt_path)
    return codes


def build_profiles(all_codes: dict, competency_mapping: dict) -> dict[str, StudentProfile]:
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

        normalized = []
        for s in student_skills:
            match SkillNormalizer.normalize(s):
                case Ok(norm):
                    normalized.append(norm)
                case _:
                    pass
        student_skills = list(dict.fromkeys(normalized))
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


def run_status(args) -> None:
    show_status()


def run_train_model(args=None) -> None:
    console_header("ОБУЧЕНИЕ LTR-МОДЕЛИ")
    detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    raw_file = detailed_file if detailed_file.exists() else basic_file
    if not raw_file.exists():
        console_info("❌ Файлы вакансий не найдены. Сначала выполните сбор.")
        sys.exit(1)
    model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
    force = getattr(args, 'force', False) if args else False
    if model_path.exists() and not force:
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


def clean_progress_files():
    for fp in [
        config.DATA_DIR / "cache" / "pipeline_progress.json",
        config.DATA_DIR / "cache" / "gap_progress.json",
    ]:
        if fp.exists():
            fp.unlink()


def run_full_pipeline(args) -> None:
    console_header("ПОЛНЫЙ ПАЙПЛАЙН: СБОР ВАКАНСИЙ + GAP-АНАЛИЗ + РЕКОМЕНДАЦИИ")
    logger.info("pipeline_started", mode="full_pipeline")

    _write_pipeline_progress(0, "Инициализация...")

    stages = [DataCollectionStage(args)]
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
    pipeline_result = orchestrator.run(name="full_pipeline")
    if pipeline_result.is_err():
        console_info(f"❌ Пайплайн не завершён: {pipeline_result.err()}")
        return
    run = pipeline_result.unwrap()

    ctx_data = {}
    for sr in run.stages:
        if sr.data:
            ctx_data.update(sr.data)

    skill_freq = ctx_data.get("skill_freq", {})
    hybrid_weights = ctx_data.get("hybrid_weights", {})
    trend_analyzer = ctx_data.get("trend_analyzer")
    level_data = ctx_data.get("level_data", [])
    vacancies_skills = ctx_data.get("vacancies_skills", [])

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

    if getattr(args, 'run_notebooks', False):
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
    console_info("📋 Логи фронтенда: frontend/logs/app.log")
    console_info(f"⏰ Завершено за {run.elapsed:.1f} сек" if run.elapsed < 120 else f"⏰ Завершено за {run.elapsed / 60:.1f} мин")


def rebuild() -> None:
    """Full rebuild: clean cache, run pipeline, train clusters, train model, gap analysis."""
    import shutil

    console_header("ПОЛНАЯ ПЕРЕСБОРКА")
    logger.info("full_rebuild_started")

    DATA = config.DATA_DIR
    to_remove = [
        DATA / "cache" / "parsed_skills.joblib",
        DATA / "processed" / "skill_weights.json",
        DATA / "cache" / "clusters" / "vacancy_clusters_junior.pkl",
        DATA / "cache" / "clusters" / "vacancy_clusters_middle.pkl",
        DATA / "cache" / "clusters" / "vacancy_clusters_senior.pkl",
        DATA / "models" / "ltr_ranker_xgb_regressor.joblib",
        DATA / "cache" / "embeddings" / "market_embeddings_junior.pkl",
        DATA / "cache" / "embeddings" / "market_embeddings_middle.pkl",
        DATA / "cache" / "embeddings" / "market_embeddings_senior.pkl",
    ]

    cache_dir = DATA / "cache" / "embeddings"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        console_info(f"✓ Удалён кэш эмбеддингов: {cache_dir}")
        logger.info("cache_directory_removed", path=str(cache_dir))

    removed_count = 0
    for f in to_remove:
        if f.exists():
            f.unlink()
            logger.info("file_removed", path=str(f))
            removed_count += 1

    clusters_dir = DATA / "cache" / "clusters"
    if clusters_dir.exists():
        shutil.rmtree(clusters_dir)
        logger.info("clusters_directory_removed", path=str(clusters_dir))
        console_info(f"✓ Удалена папка кластеров: {clusters_dir}")

    console_info(f"✓ Удалено {removed_count} файлов кэша и моделей")
    logger.info("cleanup_completed", files_removed=removed_count)


async def run_pipeline_task_async(args, task_progress_callback=None) -> dict[str, Any]:
    """Run full pipeline in executor and return result dict."""
    clean_progress_files()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_full_pipeline, args)
    return {"status": "completed"}


async def run_train_model_async(args=None, task_progress_callback=None) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_train_model, args)
    return {"status": "completed"}


async def run_status_async(args) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_status, args)
    return {"status": "completed"}


async def rebuild_async() -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, rebuild)
    return {"status": "completed"}
