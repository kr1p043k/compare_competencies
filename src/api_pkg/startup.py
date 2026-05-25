"""Инициализация API: загрузка данных, моделей, кэша."""

import hashlib
import json

import joblib
import structlog

from src import config
from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.skill_filter import SkillFilter
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer
from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
from src.analyzers.skills.trends import TrendAnalyzer
from src.models.enums import ComparisonLevel, ExperienceLevel
from src.models.student import StudentProfile, merge_skills_hierarchically
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.skills.vacancy_parser import VacancyParser
from src.parsing.utils import filter_skills_by_whitelist, load_it_skills
from src.predictors.recommendation_engine import RecommendationEngine
from src.utils import load_competency_mapping

import src.api_pkg.deps as deps

logger = structlog.get_logger("api")


async def run_startup(app):
    """Загружает все необходимые данные и модели в глобальное состояние."""
    logger.info("Запуск API-сервера, инициализация движков...")

    # 1. Загрузка вакансий
    detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    raw_file = detailed_file if detailed_file.exists() else basic_file

    deps.is_ready = False

    if not raw_file.exists():
        logger.warning("Нет файлов вакансий — режим ожидания. Запустите pipeline для сбора данных.")
        deps.vacancy_load_error = "not_found"
        deps.skill_freq = {}
        deps.skill_weights = {}
        deps.taxonomy = None
        deps.current_skills_set = set()
        deps.competency_mapping = {}
        deps.hybrid_weights = {}
        deps.basic_vacancies = []
        deps.evaluator = None
        deps.recommendation_engine = None
        deps.trend_analyzer = None
        deps.student_profiles = {}
        for lvl in ExperienceLevel:
            deps.clusterer.load_model(lvl)
        logger.info("API запущен в режиме ожидания данных")
        return

    try:
        with open(raw_file, encoding="utf-8") as f:
            basic_vacancies = json.load(f)
    except Exception as exc:
        logger.error("Файл вакансий повреждён", error=str(exc))
        deps.vacancy_load_error = f"corrupted: {exc}"
        basic_vacancies = []
    else:
        deps.vacancy_load_error = None
    if not basic_vacancies:
        logger.warning("Нет данных в файле вакансий — режим ожидания")
        deps.skill_freq = {}
        deps.skill_weights = {}
        deps.taxonomy = None
        deps.current_skills_set = set()
        deps.competency_mapping = {}
        deps.hybrid_weights = {}
        deps.basic_vacancies = []
        deps.evaluator = None
        deps.recommendation_engine = None
        deps.trend_analyzer = None
        deps.student_profiles = {}
        for lvl in ExperienceLevel:
            deps.clusterer.load_model(lvl)
        logger.info("API запущен в режиме ожидания данных")
        return

    logger.info("Загружено вакансий", count=len(basic_vacancies))

    parser = VacancyParser()

    # 2. Извлечение навыков (с кэшированием)
    cache_path = config.PARSED_SKILLS_CACHE_PATH
    vacancies_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    skill_freq_local = {}
    hybrid_weights_local = {}

    if cache_path.exists():
        try:
            cached = joblib.load(cache_path)
            if cached.get("source_hash") == vacancies_hash:
                result = cached["result"]
                skill_freq_local = result["frequencies"]
                hybrid_weights_local = result.get("hybrid_weights", {})
                logger.info("Загружен кэш парсинга навыков")
        except Exception as e:
            logger.warning("Не удалось загрузить кэш", error=str(e))

    if not skill_freq_local:
        result = parser.extract_skills_from_vacancies(basic_vacancies)
        skill_freq_local = result["frequencies"]
        hybrid_weights_local = result.get("hybrid_weights", {})
        cache_data = {"source_hash": vacancies_hash, "result": result}
        joblib.dump(cache_data, cache_path)
        logger.info("Кэш парсинга сохранён")
    elif not hybrid_weights_local:
        result = parser.extract_skills_from_vacancies(basic_vacancies)
        hybrid_weights_local = result.get("hybrid_weights", {})
        cache_data = {"source_hash": vacancies_hash, "result": result}
        joblib.dump(cache_data, cache_path)
        logger.info("Кэш парсинга обновлён (добавлены hybrid_weights)")

    deps.skill_freq = skill_freq_local

    # 3. Фильтрация весов
    filter_engine = SkillFilter()
    comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
    competency_freq = {}
    if comp_freq_path.exists():
        with open(comp_freq_path, encoding="utf-8") as f:
            competency_freq = json.load(f)
    hybrid_weights = filter_engine.get_clean_weights(
        hybrid_weights_local, competency_freq=competency_freq, use_reference=True
    )
    if not hybrid_weights and skill_freq_local:
        hybrid_weights = filter_engine.get_clean_weights(
            skill_freq_local, competency_freq=competency_freq, use_reference=True
        )
        logger.info("fallback_to_skill_freq_weights", count=len(hybrid_weights))
    deps.skill_weights = hybrid_weights

    # 4. Таксономия и белый список
    try:
        deps.taxonomy = SkillTaxonomy()
    except Exception:
        deps.taxonomy = None
    whitelist = load_it_skills()
    deps.current_skills_set = whitelist

    # 5. Подготовка данных по уровням
    level_analyzer = SkillLevelAnalyzer()
    level_vacancies_data = []
    vacancies_skills = []

    for vac in basic_vacancies:
        vac_skills = []
        if "extracted_skills" in vac:
            vac_skills = vac["extracted_skills"]
        else:
            desc = vac.get("description", "")
            snip = vac.get("snippet") or {}
            req = snip.get("requirement", "")
            resp = snip.get("responsibility", "")
            combined = f"{desc} {req} {resp}"
            vac_skills = parser.extract_skills_from_description(combined)

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
                if (
                    "junior" in exp_lower
                    or "нет опыта" in exp_lower
                    or "стажер" in exp_lower
                ):
                    experience = ExperienceLevel.JUNIOR
                elif "senior" in exp_lower or "более 6" in exp_lower:
                    experience = ExperienceLevel.SENIOR
        if experience == ExperienceLevel.MIDDLE:
            name = vac.get("name", "").lower()
            if (
                "junior" in name
                or "младший" in name
                or "стажер" in name
                or "intern" in name
            ):
                experience = ExperienceLevel.JUNIOR
            elif "senior" in name or "старший" in name or "ведущий" in name:
                experience = ExperienceLevel.SENIOR
        if vac_skills:
            level_vacancies_data.append(
                {
                    "skills": vac_skills,
                    "description": vac.get("description", ""),
                    "experience": experience,
                }
            )
            vacancies_skills.append(vac_skills)

    level_analyzer.analyze_vacancies(level_vacancies_data)

    # 6. Веса по уровням
    skill_weights_by_level = {}
    for level in ExperienceLevel:
        skill_weights_by_level[level] = level_analyzer.get_weights_for_level(
            deps.skill_weights, level
        )

    # 7. Студенческие профили
    deps.competency_mapping = load_competency_mapping()

    def load_student_codes(name: str) -> list[str]:
        path = config.DATA_DIR / "students" / f"{name}_competency.json"
        if not path.exists():
            path = config.DATA_DIR / "students" / f"{name}.json"
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return (
                data.get("компетенции") or data.get("навыки") or data.get("codes") or []
            )
        except Exception:
            return []

    def map_codes(codes):
        if not deps.competency_mapping:
            return codes
        skills = set()
        for code in codes:
            code_norm = "".join(c for c in code if c.isalnum()).upper()
            for key, value in deps.competency_mapping.items():
                key_norm = "".join(c for c in key if c.isalnum()).upper()
                if code_norm == key_norm:
                    skills.update(value)
                    break
        return list(skills)

    profile_levels = {
        "base": ExperienceLevel.JUNIOR,
        "dc": ExperienceLevel.MIDDLE,
        "top_dc": ExperienceLevel.SENIOR,
    }
    for pname, target in profile_levels.items():
        codes = load_student_codes(pname)
        if pname == "top_dc":
            top_codes = load_student_codes("top_dc")
            dc_codes = load_student_codes("dc")
            base_codes = load_student_codes("base")
            top_skills = map_codes(top_codes)
            dc_skills = map_codes(dc_codes)
            base_skills = map_codes(base_codes)
            skills = merge_skills_hierarchically(top_skills, dc_skills, base_skills)
        else:
            skills = map_codes(codes)
        skills = [
            SkillNormalizer.normalize(s) for s in skills if SkillNormalizer.normalize(s)
        ]
        skills = list(dict.fromkeys(skills))
        deps.student_profiles[pname] = StudentProfile(
            profile_name=pname,
            competencies=codes,
            skills=skills,
            target_level=target,
        )

    # 8. ProfileEvaluator и RecommendationEngine
    deps.hybrid_weights = hybrid_weights
    deps.basic_vacancies = basic_vacancies

    deps.evaluator = ProfileEvaluator(
        skill_weights=deps.skill_weights,
        vacancies_skills=vacancies_skills,
        vacancies_skills_dict=level_vacancies_data,
        hybrid_weights=hybrid_weights,
        skill_weights_by_level=skill_weights_by_level,
    )
    deps.recommendation_engine = RecommendationEngine(
        use_ltr=True, use_llm=False, profile_evaluator=deps.evaluator
    )
    deps.recommendation_engine.comparator = CompetencyComparator(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        use_embeddings=True,
        level=ComparisonLevel.MIDDLE,
        similarity_threshold=0.80,
    )
    deps.recommendation_engine.fit(vacancies_skills, skill_weights=hybrid_weights)

    # 9. Кластеры
    for lvl in ExperienceLevel:
        if not deps.clusterer.load_model(lvl):
            logger.warning("Модель кластеров не найдена", level=str(lvl))

    # 10. Тренды
    whitelist_set = load_it_skills()
    skill_freq_filtered = (
        filter_skills_by_whitelist(deps.skill_freq, whitelist_set)
        if whitelist_set
        else deps.skill_freq
    )
    deps.trend_analyzer = TrendAnalyzer(skill_freq_filtered)

    deps.is_ready = True
    logger.info("API готов к работе")
