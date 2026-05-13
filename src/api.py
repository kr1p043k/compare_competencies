"""
FastAPI для доступа к рекомендациям и аналитике рынка труда.
Загружает предварительно собранные данные и предоставляет REST API.
"""

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.analyzers.clustering.vacancy_clustering import VacancyClusterer
from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.skill_filter import SkillFilter
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer
from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
from src.analyzers.skills.trends import TrendAnalyzer
from src.models.enums import ComparisonLevel, ExperienceLevel
from src.models.student import StudentProfile, merge_skills_hierarchically
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.skills.skill_validator import SkillValidator
from src.parsing.skills.vacancy_parser import VacancyParser
from src.parsing.utils import filter_skills_by_whitelist, load_it_skills
from src.predictors.recommendation_engine import RecommendationEngine
from src.utils import load_competency_mapping, safe_load_pickle

logger = structlog.get_logger("api")

# Глобальные движки (инициализируются при старте)
evaluator: ProfileEvaluator | None = None
recommendation_engine: RecommendationEngine | None = None
clusterer: VacancyClusterer = VacancyClusterer()
trend_analyzer: TrendAnalyzer | None = None
student_profiles: dict[str, StudentProfile] = {}
skill_weights: dict[str, float] = {}
hybrid_weights: dict[str, float] = {}
competency_mapping: dict[str, list[str]] = {}
skill_freq: dict[str, int] = {}
taxonomy: SkillTaxonomy | None = None
current_skills_set: set[str] = set()

app = FastAPI(title="Competency Analyzer API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Загружает все необходимые данные и модели."""
    global evaluator, recommendation_engine, trend_analyzer
    global student_profiles, skill_weights, hybrid_weights, competency_mapping
    global skill_freq, taxonomy, current_skills_set

    logger.info("Запуск API-сервера, инициализация движков...")

    # 1. Загрузка вакансий
    detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    raw_file = detailed_file if detailed_file.exists() else basic_file
    if not raw_file.exists():
        logger.error("Нет файлов вакансий, API не может работать")
        raise RuntimeError("No vacancy files found")

    with open(raw_file, encoding="utf-8") as f:
        basic_vacancies = json.load(f)
    logger.info(f"Загружено {len(basic_vacancies)} вакансий")

    parser = VacancyParser()

    # 2. Извлечение навыков (с кэшированием)
    cache_path = config.PARSED_SKILLS_CACHE_PATH
    vacancies_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    skill_freq_local = {}
    hybrid_weights_local = {}

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = safe_load_pickle(cache_path)
            if cached.get("source_hash") == vacancies_hash:
                result = cached["result"]
                skill_freq_local = result["frequencies"]
                hybrid_weights_local = result.get("hybrid_weights", {})
                logger.info("Загружен кэш парсинга навыков")
        except Exception as e:
            logger.warning(f"Не удалось загрузить кэш: {e}")

    if not skill_freq_local:
        result = parser.extract_skills_from_vacancies(basic_vacancies)
        skill_freq_local = result["frequencies"]
        hybrid_weights_local = result.get("hybrid_weights", {})
        cache_data = {"source_hash": vacancies_hash, "result": result}
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            # Создаём манифест
            from src.artifacts import ArtifactManifest

            manifest = ArtifactManifest(
                artifact_path=cache_path,
                metrics={"num_skills": len(skill_freq)},
            )
            manifest.save()
        logger.info("Кэш парсинга сохранён")

    skill_freq = skill_freq_local

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
    skill_weights = hybrid_weights

    # 4. Таксономия и белый список
    try:
        taxonomy = SkillTaxonomy()
    except Exception:
        taxonomy = None
    whitelist = load_it_skills()
    current_skills_set = whitelist

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
        skill_weights_by_level[level] = level_analyzer.get_weights_for_level(skill_weights, level)

    # 7. Студенческие профили
    competency_mapping = load_competency_mapping()

    def load_student_codes(name: str) -> list[str]:
        path = config.DATA_DIR / "students" / f"{name}_competency.json"
        if not path.exists():
            path = config.DATA_DIR / "students" / f"{name}.json"
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("компетенции") or data.get("навыки") or data.get("codes") or []
        except Exception:
            return []

    def map_codes(codes):
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
        skills = [SkillNormalizer.normalize(s) for s in skills if SkillNormalizer.normalize(s)]
        skills = list(dict.fromkeys(skills))
        student_profiles[pname] = StudentProfile(
            profile_name=pname,
            competencies=codes,
            skills=skills,
            target_level=target,
        )

    # 8. ProfileEvaluator и RecommendationEngine
    evaluator = ProfileEvaluator(
        skill_weights=skill_weights,
        vacancies_skills=vacancies_skills,
        vacancies_skills_dict=level_vacancies_data,
        hybrid_weights=hybrid_weights,
        skill_weights_by_level=skill_weights_by_level,
    )
    recommendation_engine = RecommendationEngine(use_ltr=True, use_llm=False, profile_evaluator=evaluator)
    recommendation_engine.comparator = CompetencyComparator(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        use_embeddings=True,
        level=ComparisonLevel.MIDDLE,
        similarity_threshold=0.80,
    )
    recommendation_engine.fit(vacancies_skills, skill_weights=hybrid_weights)

    # 9. Кластеры
    for lvl in ExperienceLevel:
        if not clusterer.load_model(lvl):
            logger.warning(f"Модель кластеров для {lvl} не найдена")

    # 10. Тренды
    whitelist_set = load_it_skills()
    skill_freq_filtered = filter_skills_by_whitelist(skill_freq, whitelist_set) if whitelist_set else skill_freq
    trend_analyzer = TrendAnalyzer(skill_freq_filtered)

    logger.info("API готов к работе")


# ---------- Эндпоинты ----------


@app.get("/api/health")
async def health():
    return {"status": "ok", "evaluator": evaluator is not None}


@app.get("/health")
async def health_check():
    """Проверка живости сервера."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/ready")
async def ready_check():
    """Проверка готовности всех компонентов."""
    components = {
        "evaluator": evaluator is not None,
        "recommendation_engine": recommendation_engine is not None and recommendation_engine.is_fitted,
        "clusterer": clusterer.is_fitted,
        "trend_analyzer": trend_analyzer is not None,
    }
    ready = all(components.values())
    status = "ready" if ready else "not ready"
    return {"status": status, "components": components}


@app.get("/api/recommendations/{profile}")
async def get_recommendations(profile: str):
    if profile not in student_profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    student = student_profiles[profile]
    full_rec = recommendation_engine.generate_recommendations(student)
    return full_rec


@app.get("/api/market/top-skills")
async def get_top_skills(limit: int = Query(15, ge=1, le=50)):
    top = sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {"skills": [{"skill": s, "weight": round(w, 4)} for s, w in top]}


@app.get("/api/market/skill/{skill}")
async def get_skill_info(skill: str):
    weight = skill_weights.get(skill, 0.0)
    freq = skill_freq.get(skill, 0)
    category = taxonomy.get_category_label(skill) if taxonomy else "unknown"
    icon = taxonomy.get_category_icon(skill) if taxonomy else ""
    return {
        "skill": skill,
        "frequency": freq,
        "weight": round(weight, 4),
        "category": category,
        "icon": icon,
    }


@app.get("/api/clusters/{level}")
async def get_clusters(level: ExperienceLevel = ExperienceLevel.MIDDLE):
    if not clusterer.is_fitted:
        clusterer.load_model(level)
    if not clusterer.is_fitted:
        raise HTTPException(status_code=503, detail="Модели кластеров не загружены")
    clusters = []
    for cid in range(clusterer.n_clusters_):
        clusters.append(
            {
                "id": cid,
                "name": clusterer._generate_cluster_name(cid),
                "top_skills": clusterer.get_top_skills_in_cluster(cid, top_n=5),
            }
        )
    return {"level": level, "clusters": clusters}


@app.get("/api/clusters/summary")
async def clusters_summary():
    result = {}
    for lvl in ExperienceLevel:
        clusterer.load_model(lvl)
        if clusterer.is_fitted:
            result[lvl] = {
                "clusters": clusterer.n_clusters_,
                "type": clusterer.clusterer_type,
                "top_clusters": [
                    {
                        "id": cid,
                        "name": clusterer._generate_cluster_name(cid),
                        "top_skills": clusterer.get_top_skills_in_cluster(cid, top_n=5),
                    }
                    for cid in range(clusterer.n_clusters_)
                ],
            }
        else:
            result[lvl] = {"error": "not_fitted"}
    return result


@app.get("/api/profiles/compare")
async def compare_profiles():
    evaluations = {}
    for pname, student in student_profiles.items():
        try:
            eval_result = evaluator.evaluate_profile(student)
            evaluations[pname] = {
                "market_coverage_score": eval_result.get("market_coverage_score"),
                "skill_coverage": eval_result.get("skill_coverage"),
                "domain_coverage_score": eval_result.get("domain_coverage_score"),
                "readiness_score": eval_result.get("readiness_score"),
                "real_coverage": eval_result.get("market_skill_coverage"),
            }
        except Exception as e:
            logger.error(f"Ошибка оценки {pname}: {e}")
            evaluations[pname] = {"error": str(e)}
    return {"profiles": evaluations}


@app.get("/api/trends")
async def get_trends(top_n: int = Query(15), min_change: float = Query(3.0)):
    try:
        trends = trend_analyzer.get_trending_skills(top_n=top_n, min_change_percent=min_change)
        return {"trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/taxonomy/coverage")
async def taxonomy_coverage():
    if not taxonomy:
        raise HTTPException(status_code=503, detail="Таксономия не загружена")
    coverage = {}
    for cat_id in taxonomy.get_all_categories():
        cat_skills = set(s.lower() for s in taxonomy.get_skills_in_category(cat_id))
        covered = cat_skills & current_skills_set
        coverage[cat_id] = {
            "label": taxonomy.get_category_label_by_id(cat_id),
            "icon": taxonomy.get_category_icon_by_id(cat_id),
            "total": len(cat_skills),
            "covered": len(covered),
            "percent": round(len(covered) / len(cat_skills) * 100, 1) if cat_skills else 0,
        }
    return {"coverage": coverage}


@app.get("/api/skills/missing")
async def missing_skills(min_frequency: int = Query(1)):
    validator = SkillValidator(whitelist=None)
    extracted = {}
    for skill, freq in skill_freq.items():
        if skill.lower() not in current_skills_set and freq >= min_frequency and validator.validate(skill).is_valid:
            extracted[skill] = freq
    sorted_skills = sorted(extracted.items(), key=lambda x: x[1], reverse=True)
    return {"missing_skills": [{"skill": s, "frequency": f} for s, f in sorted_skills]}


@app.get("/api/skills/dead")
async def dead_skills():
    extracted_lower = {s.lower() for s in skill_freq}
    dead = sorted(s for s in current_skills_set if s.lower() not in extracted_lower)
    return {"dead_skills": dead}


@app.get("/api/status")
async def get_status():
    return {
        "vacancies_loaded": len(skill_freq) > 0,
        "skill_weights_count": len(skill_weights),
        "taxonomy_loaded": taxonomy is not None,
        "whitelist_size": len(current_skills_set),
        "profiles_available": list(student_profiles.keys()),
        "clusters": {lvl: clusterer.load_model(lvl) and clusterer.is_fitted for lvl in ExperienceLevel},
        "trends_available": trend_analyzer is not None,
        "recommendation_engine_ready": recommendation_engine is not None and recommendation_engine.is_fitted,
    }
