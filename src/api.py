"""
FastAPI для доступа к рекомендациям и аналитике рынка труда.
Загружает предварительно собранные данные и предоставляет REST API.
"""

import hashlib
import json
import pickle
import sys
import time
import asyncio
import shutil
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

# ============================================
# КОНФИГУРАЦИЯ ЛИМИТЕРА
# ============================================
limiter = Limiter(key_func=get_remote_address)

# ============================================
# ГЛОБАЛЬНЫЕ ДВИЖКИ (инициализируются при старте)
# ============================================
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
basic_vacancies: list = []  # Список всех вакансий
_regions_cache: list[str] = []  # Кэш списка регионов
_regions_cache_time: float = 0

# ============================================
# LIFESPAN
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    await startup()
    yield
    # Shutdown логика
    logger.info("API shutting down...")
    # Здесь можно добавить сохранение состояния, закрытие соединений и т.д.

app = FastAPI(
    title="Competency Analyzer API", 
    version="2.0",
    lifespan=lifespan
)

# ============================================
# MIDDLEWARE
# ============================================
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Too many requests. Please try again later."}
))

allowed_origins = config.ALLOWED_ORIGINS.split(",") if config.ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware для request ID и структурированного логирования
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Логируем входящий запрос
    logger.info(
        "Incoming request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Добавляем request ID в заголовки ответа
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Логируем завершение запроса
    logger.info(
        "Request completed",
        request_id=request_id,
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response

# ============================================
# ГЛОБАЛЬНЫЙ EXCEPTION HANDLER
# ============================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        error_type=type(exc).__name__,
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
            "error_type": type(exc).__name__
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        "HTTP exception",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": request_id
        }
    )

# ============================================
# DEPENDENCY INJECTION
# ============================================
def get_evaluator() -> ProfileEvaluator:
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Profile evaluator not initialized")
    return evaluator

def get_recommendation_engine() -> RecommendationEngine:
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    return recommendation_engine

def get_clusterer() -> VacancyClusterer:
    return clusterer

def get_trend_analyzer() -> TrendAnalyzer:
    if trend_analyzer is None:
        raise HTTPException(status_code=503, detail="Trend analyzer not initialized")
    return trend_analyzer

def get_taxonomy() -> SkillTaxonomy | None:
    return taxonomy

def get_basic_vacancies() -> list:
    if not basic_vacancies:
        raise HTTPException(status_code=503, detail="Vacancy data not loaded yet")
    return basic_vacancies

def get_student_profiles() -> dict[str, StudentProfile]:
    return student_profiles

def get_skill_weights() -> dict[str, float]:
    return skill_weights

def get_skill_freq() -> dict[str, int]:
    return skill_freq

def get_hybrid_weights() -> dict[str, float]:
    return hybrid_weights

def validate_regions(regions: list[str]) -> list[str]:
    """Проверяет, что переданные регионы существуют"""
    global _regions_cache
    
    if not _regions_cache:
        raise HTTPException(status_code=503, detail="Regions data not loaded yet")
    
    if "Все регионы" in regions:
        return regions
    
    invalid_regions = [r for r in regions if r not in _regions_cache]
    if invalid_regions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regions: {', '.join(invalid_regions)}. Use /api/regions to get valid regions."
        )
    
    return regions

# ============================================
# МОДЕЛИ ДЛЯ ПАЙПЛАЙНА
# ============================================

class PipelineAction(str, Enum):
    FULL_CYCLE = "full-cycle"
    REBUILD = "rebuild"
    TRAIN_CLUSTERS = "train-clusters"
    TRAIN_MODEL = "train-model"
    GAP_ANALYSIS = "gap-analysis"


class PipelineResponse(BaseModel):
    status: str
    message: str
    command: Optional[str] = None
    exit_code: Optional[int] = None
    output: Optional[str] = None


class PipelineTaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    message: str
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output: Optional[str] = None

class FullPipelineRequest(BaseModel):
    student_profile: StudentProfile
    regions: list[str] = Field(default=["Все регионы"], description="Список регионов для фильтрации")
    whitelist_skills: list[str] = Field(default_factory=list, description="Белый список навыков")
    level: ComparisonLevel = ComparisonLevel.MIDDLE
    use_custom_matrix: bool = False
    include_explanations: bool = Field(default=True, description="Включать ли объяснения в результаты")
    
    @field_validator('regions')
    @classmethod
    def validate_regions_list(cls, v):
        if not v:
            raise ValueError('At least one region must be specified')
        return v
    
    @field_validator('whitelist_skills')
    @classmethod
    def validate_whitelist(cls, v):
        # Убираем дубликаты и пустые строки
        cleaned = list(dict.fromkeys([s.strip() for s in v if s and s.strip()]))
        return cleaned

# Хранилище статусов фоновых задач
pipeline_tasks: dict[str, PipelineTaskStatus] = {}

# ============================================
# МОДЕЛИ ДЛЯ РЕГИОНОВ
# ============================================

class RegionsResponse(BaseModel):
    regions: list[str]
    total: int
    default: str = "Все регионы"


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "2.0",
        "evaluator": evaluator is not None,
        "recommendation_engine": recommendation_engine is not None
    }

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

@app.get("/api/status")
async def get_status(
    profiles: dict[str, StudentProfile] = Depends(get_student_profiles),
    clusterer_instance: VacancyClusterer = Depends(get_clusterer),
    trend_analyzer_instance: TrendAnalyzer = Depends(get_trend_analyzer),
    engine: RecommendationEngine = Depends(get_recommendation_engine),
    taxonomy_instance: SkillTaxonomy | None = Depends(get_taxonomy),
    weights: dict[str, float] = Depends(get_skill_weights)
):
    return {
        "vacancies_loaded": len(basic_vacancies) > 0,
        "skill_weights_count": len(weights),
        "taxonomy_loaded": taxonomy_instance is not None,
        "whitelist_size": len(current_skills_set),
        "profiles_available": list(profiles.keys()),
        "clusters": {lvl: clusterer_instance.load_model(lvl) and clusterer_instance.is_fitted for lvl in ExperienceLevel},
        "trends_available": trend_analyzer_instance is not None,
        "recommendation_engine_ready": engine is not None and engine.is_fitted,
    }

# ============================================
# ЭНДПОИНТЫ ДЛЯ РЕГИОНОВ
# ============================================

@app.get("/api/regions", response_model=RegionsResponse)
@limiter.limit("60/minute")
async def get_regions(
    request: Request,
    vacancies: list = Depends(get_basic_vacancies)
):
    """Возвращает список всех доступных регионов/городов из загруженных вакансий"""
    global _regions_cache, _regions_cache_time
    
    # Используем кэш на 5 минут
    current_time = time.time()
    if _regions_cache and (current_time - _regions_cache_time) < 300:
        return RegionsResponse(
            regions=_regions_cache,
            total=len(_regions_cache),
            default="Все регионы"
        )
    
    try:
        regions_set = set()
        
        for vac in vacancies:
            # Вариант 1: Поле area.name (самый частый)
            if isinstance(vac.get("area"), dict):
                area_name = vac["area"].get("name")
                if area_name:
                    regions_set.add(area_name)
            
            # Вариант 2: Поле region
            region = vac.get("region") or vac.get("area_name")
            if isinstance(region, str) and region:
                regions_set.add(region)
        
        regions_list = sorted({r.strip() for r in regions_set if r and str(r).strip()})
        
        # Обновляем кэш
        _regions_cache = regions_list
        _regions_cache_time = current_time
        
        return RegionsResponse(
            regions=regions_list,
            total=len(regions_list),
            default="Все регионы"
        )
        
    except Exception as e:
        logger.error("Ошибка при получении регионов", error=str(e))
        raise HTTPException(status_code=500, detail="Не удалось извлечь список регионов")

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

async def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Асинхронно выполняет команду и возвращает код возврата, stdout, stderr"""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')


async def run_pipeline_task(action: PipelineAction, task_id: str, **kwargs):
    """Запускает задачу пайплайна в фоне"""
    base_dir = Path(__file__).parent.parent
    
    # Обновляем статус задачи
    pipeline_tasks[task_id] = PipelineTaskStatus(
        task_id=task_id,
        status="running",
        message=f"Running {action.value}...",
        started_at=time.time()
    )
    
    try:
        if action == PipelineAction.REBUILD:
            rebuild_script = base_dir / "scripts" / "full_rebuild.py"
            if not rebuild_script.exists():
                raise FileNotFoundError(f"full_rebuild.py not found at {rebuild_script}")
            
            cmd = ["python", str(rebuild_script)]
            returncode, stdout, stderr = await run_command(cmd, base_dir)
            
            if returncode != 0:
                raise RuntimeError(f"Rebuild failed: {stderr[:500]}")
            
            pipeline_tasks[task_id] = PipelineTaskStatus(
                task_id=task_id,
                status="completed",
                message="Full rebuild completed successfully",
                started_at=pipeline_tasks[task_id].started_at,
                completed_at=time.time(),
                output=stdout[-2000:] if stdout else stderr[-2000:]
            )
        
        elif action == PipelineAction.FULL_CYCLE:
            cmd = ["python", str(base_dir / "main.py")]
            
            skip_collection = kwargs.get('skip_collection', False)
            run_gap_analysis = kwargs.get('run_gap_analysis', True)
            regions = kwargs.get('regions', '0')
            
            if skip_collection:
                cmd.append("--skip-collection")
            
            if run_gap_analysis:
                cmd.append("--run-gap-analysis")
            
            if regions != "0":
                cmd.extend(["--regions", regions])
            
            returncode, stdout, stderr = await run_command(cmd, base_dir)
            
            status = "completed" if returncode == 0 else "failed"
            message = "Pipeline execution completed" if returncode == 0 else f"Pipeline failed with code {returncode}"
            
            pipeline_tasks[task_id] = PipelineTaskStatus(
                task_id=task_id,
                status=status,
                message=message,
                started_at=pipeline_tasks[task_id].started_at,
                completed_at=time.time(),
                output=stdout[-2000:] if stdout else stderr[-2000:]
            )
        
        elif action == PipelineAction.TRAIN_CLUSTERS:
            cmd = ["python", str(base_dir / "scripts" / "train_clusters.py"), "--level", "all"]
            returncode, stdout, stderr = await run_command(cmd, base_dir)
            
            status = "completed" if returncode == 0 else "failed"
            message = "Cluster training completed" if returncode == 0 else f"Cluster training failed with code {returncode}"
            
            pipeline_tasks[task_id] = PipelineTaskStatus(
                task_id=task_id,
                status=status,
                message=message,
                started_at=pipeline_tasks[task_id].started_at,
                completed_at=time.time(),
                output=stdout[-2000:] if stdout else stderr[-2000:]
            )
        
        elif action == PipelineAction.TRAIN_MODEL:
            cmd = ["python", str(base_dir / "main.py"), "--train-model"]
            returncode, stdout, stderr = await run_command(cmd, base_dir)
            
            status = "completed" if returncode == 0 else "failed"
            message = "Model training completed" if returncode == 0 else f"Model training failed with code {returncode}"
            
            pipeline_tasks[task_id] = PipelineTaskStatus(
                task_id=task_id,
                status=status,
                message=message,
                started_at=pipeline_tasks[task_id].started_at,
                completed_at=time.time(),
                output=stdout[-2000:] if stdout else stderr[-2000:]
            )
        
        elif action == PipelineAction.GAP_ANALYSIS:
            cmd = ["python", str(base_dir / "main.py"), "--skip-collection", "--run-gap-analysis"]
            returncode, stdout, stderr = await run_command(cmd, base_dir)
            
            status = "completed" if returncode == 0 else "failed"
            message = "Gap analysis completed" if returncode == 0 else f"Gap analysis failed with code {returncode}"
            
            pipeline_tasks[task_id] = PipelineTaskStatus(
                task_id=task_id,
                status=status,
                message=message,
                started_at=pipeline_tasks[task_id].started_at,
                completed_at=time.time(),
                output=stdout[-2000:] if stdout else stderr[-2000:]
            )
    
    except Exception as e:
        logger.error("Ошибка выполнения задачи пайплайна", error=str(e))
        pipeline_tasks[task_id] = PipelineTaskStatus(
            task_id=task_id,
            status="failed",
            message=str(e),
            started_at=pipeline_tasks[task_id].started_at if task_id in pipeline_tasks else time.time(),
            completed_at=time.time(),
            output=str(e)
        )

# ============================================
# ЭНДПОИНТ С ФИЛЬТРАЦИЕЙ ПО ГОРОДУ
# ============================================

@app.post("/api/pipeline/full-cycle")
@limiter.limit("5/minute")
async def run_full_pipeline(
    request: Request,
    full_request: FullPipelineRequest, 
    background_tasks: BackgroundTasks
):
    """Запуск полного анализа компетенций с фильтрацией по регионам"""
    
    # Валидация регионов
    validated_regions = validate_regions(full_request.regions)
    
    task_id = f"pipeline_{int(time.time())}"
    
    logger.info(
        f"Запуск пайплайна для регионов: {validated_regions}",
        request_id=getattr(request.state, "request_id", "unknown")
    )
    
    # Передаём выбранные регионы дальше
    regions_str = ",".join(validated_regions) if "Все регионы" not in validated_regions else None
    
    # Здесь можно добавить whitelist обработку
    if full_request.whitelist_skills:
        logger.info("Применён whitelist навыков", count=len(full_request.whitelist_skills))
    
    # Запуск в фоне
    background_tasks.add_task(
        run_pipeline_task,
        PipelineAction.FULL_CYCLE,
        task_id,
        regions=regions_str
    )
    
    return {
        "task_id": task_id,
        "status": "started",
        "regions": validated_regions,
        "message": "Анализ запущен",
        "include_explanations": full_request.include_explanations
    }


# Дополнительный эндпоинт для отладки — получить вакансии по выбранному городу
@app.get("/api/vacancies/by-region")
@limiter.limit("30/minute")
async def get_vacancies_by_region(
    request: Request,
    region: str = Query(..., description="Название региона (например: Москва)"),
    limit: int = Query(50, ge=1, le=500),
    vacancies: list = Depends(get_basic_vacancies)
):
    """Возвращает вакансии по выбранному региону (для проверки фильтра)"""
    
    filtered = []
    for vac in vacancies:
        area_name = None
        if isinstance(vac.get("area"), dict):
            area_name = vac["area"].get("name")
        elif vac.get("region"):
            area_name = vac.get("region")
        
        if area_name and region.lower() in area_name.lower():
            filtered.append(vac)
            if len(filtered) >= limit:
                break
    
    return {
        "region": region,
        "count": len(filtered),
        "limit": limit,
        "vacancies": filtered[:limit]
    }

# ============================================
# STARTUP FUNCTION
# ============================================

async def startup():
    """Загружает все необходимые данные и модели."""
    global evaluator, recommendation_engine, trend_analyzer
    global student_profiles, skill_weights, hybrid_weights, competency_mapping
    global skill_freq, taxonomy, current_skills_set, basic_vacancies

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
    logger.info("Загружено вакансий", count=len(basic_vacancies))

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
            logger.warning("Не удалось загрузить кэш", error=str(e))

    if not skill_freq_local:
        result = parser.extract_skills_from_vacancies(basic_vacancies)
        skill_freq_local = result["frequencies"]
        hybrid_weights_local = result.get("hybrid_weights", {})
        cache_data = {"source_hash": vacancies_hash, "result": result}
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
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
            logger.warning("Модель кластеров не найдена", level=str(lvl))

    # 10. Тренды
    whitelist_set = load_it_skills()
    skill_freq_filtered = filter_skills_by_whitelist(skill_freq, whitelist_set) if whitelist_set else skill_freq
    trend_analyzer = TrendAnalyzer(skill_freq_filtered)

    logger.info("API готов к работе")

# ============================================
# ПРОФИЛИ И РЕКОМЕНДАЦИИ
# ============================================

@app.get("/api/recommendations/{profile}")
@limiter.limit("30/minute")
async def get_recommendations(
    request: Request,
    profile: str,
    engine: RecommendationEngine = Depends(get_recommendation_engine),
    profiles: dict[str, StudentProfile] = Depends(get_student_profiles)
):
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    student = profiles[profile]
    full_rec = engine.generate_recommendations(student)
    return full_rec


@app.get("/api/profiles/compare")
@limiter.limit("20/minute")
async def compare_profiles(
    request: Request,
    eval_instance: ProfileEvaluator = Depends(get_evaluator),
    profiles: dict[str, StudentProfile] = Depends(get_student_profiles)
):
    evaluations = {}
    for pname, student in profiles.items():
        try:
            eval_result = eval_instance.evaluate_profile(student)
            evaluations[pname] = {
                "market_coverage_score": eval_result.get("market_coverage_score"),
                "skill_coverage": eval_result.get("skill_coverage"),
                "domain_coverage_score": eval_result.get("domain_coverage_score"),
                "readiness_score": eval_result.get("readiness_score"),
                "real_coverage": eval_result.get("market_skill_coverage"),
            }
        except Exception as e:
            logger.error("Ошибка оценки профиля", profile=pname, error=str(e))
            evaluations[pname] = {"error": str(e)}
    return {"profiles": evaluations}


@app.get("/api/profiles/{profile}")
@limiter.limit("60/minute")
async def get_profile(
    request: Request,
    profile: str,
    profiles: dict[str, StudentProfile] = Depends(get_student_profiles)
):
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    student = profiles[profile]
    return {
        "profile_name": student.profile_name,
        "target_level": student.target_level,
        "skills_count": len(student.skills),
        "skills": student.skills[:50],
        "competencies_count": len(student.competencies),
        "competencies": student.competencies[:50]
    }


# ============================================
# РЫНОЧНЫЕ НАВЫКИ
# ============================================

@app.get("/api/market/top-skills")
@limiter.limit("60/minute")
async def get_top_skills(
    request: Request,
    limit: int = Query(15, ge=1, le=50),
    weights: dict[str, float] = Depends(get_skill_weights)
):
    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {"skills": [{"skill": s, "weight": round(w, 4)} for s, w in top]}


@app.get("/api/market/skill/{skill}")
@limiter.limit("60/minute")
async def get_skill_info(
    request: Request,
    skill: str,
    weights: dict[str, float] = Depends(get_skill_weights),
    freq: dict[str, int] = Depends(get_skill_freq),
    taxonomy_instance: SkillTaxonomy | None = Depends(get_taxonomy)
):
    weight = weights.get(skill, 0.0)
    freq_val = freq.get(skill, 0)
    category = taxonomy_instance.get_category_label(skill) if taxonomy_instance else "unknown"
    icon = taxonomy_instance.get_category_icon(skill) if taxonomy_instance else ""
    return {
        "skill": skill,
        "frequency": freq_val,
        "weight": round(weight, 4),
        "category": category,
        "icon": icon,
    }


@app.get("/api/market-competencies")
@limiter.limit("30/minute")
async def get_market_competencies(
    request: Request,
    weights: dict[str, float] = Depends(get_skill_weights)
):
    """Возвращает рыночные компетенции"""
    top_skills = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:100]
    return {
        "skills": [{"skill": s, "weight": w} for s, w in top_skills],
        "total": len(weights)
    }


# ============================================
# КЛАСТЕРЫ
# ============================================

@app.get("/api/clusters/summary")
@limiter.limit("20/minute")
async def clusters_summary(
    request: Request,
    clusterer_instance: VacancyClusterer = Depends(get_clusterer)
):
    result = {}
    for lvl in ExperienceLevel:
        clusterer_instance.load_model(lvl)
        if clusterer_instance.is_fitted:
            result[lvl] = {
                "clusters": clusterer_instance.n_clusters_,
                "type": clusterer_instance.clusterer_type,
                "top_clusters": [
                    {
                        "id": cid,
                        "name": clusterer_instance._generate_cluster_name(cid),
                        "top_skills": clusterer_instance.get_top_skills_in_cluster(cid, top_n=5),
                    }
                    for cid in range(clusterer_instance.n_clusters_)
                ],
            }
        else:
            result[lvl] = {"error": "not_fitted"}
    return result


@app.get("/api/clusters/{level}")
@limiter.limit("30/minute")
async def get_clusters(
    request: Request,
    level: ExperienceLevel = ExperienceLevel.MIDDLE,
    clusterer_instance: VacancyClusterer = Depends(get_clusterer)
):
    if not clusterer_instance.is_fitted:
        clusterer_instance.load_model(level)
    if not clusterer_instance.is_fitted:
        raise HTTPException(status_code=503, detail="Модели кластеров не загружены")
    clusters = []
    for cid in range(clusterer_instance.n_clusters_):
        clusters.append(
            {
                "id": cid,
                "name": clusterer_instance._generate_cluster_name(cid),
                "top_skills": clusterer_instance.get_top_skills_in_cluster(cid, top_n=5),
            }
        )
    return {"level": level, "clusters": clusters}


# ============================================
# ТРЕНДЫ
# ============================================

@app.get("/api/trends")
@limiter.limit("30/minute")
async def get_trends(
    request: Request,
    top_n: int = Query(15),
    min_change: float = Query(3.0),
    trend_analyzer_instance: TrendAnalyzer = Depends(get_trend_analyzer)
):
    try:
        trends = trend_analyzer_instance.get_trending_skills(top_n=top_n, min_change_percent=min_change)
        return {"trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================
# ТАКСОНОМИЯ
# ============================================

@app.get("/api/taxonomy/coverage")
@limiter.limit("20/minute")
async def taxonomy_coverage(
    request: Request,
    taxonomy_instance: SkillTaxonomy | None = Depends(get_taxonomy)
):
    if not taxonomy_instance:
        raise HTTPException(status_code=503, detail="Таксономия не загружена")
    coverage = {}
    for cat_id in taxonomy_instance.get_all_categories():
        cat_skills = set(s.lower() for s in taxonomy_instance.get_skills_in_category(cat_id))
        covered = cat_skills & current_skills_set
        coverage[cat_id] = {
            "label": taxonomy_instance.get_category_label_by_id(cat_id),
            "icon": taxonomy_instance.get_category_icon_by_id(cat_id),
            "total": len(cat_skills),
            "covered": len(covered),
            "percent": round(len(covered) / len(cat_skills) * 100, 1) if cat_skills else 0,
        }
    return {"coverage": coverage}


# ============================================
# НАВЫКИ (MISSING/DEAD)
# ============================================



# === Profession Taxonomy Endpoints ===

@app.get("/api/taxonomy/professions")
@limiter.limit("60/minute")
async def get_professions(request: Request):
    """Список всех профессий из таксономии."""
    try:
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
        taxonomy = ProfessionTaxonomy()
        professions = []
        for name in taxonomy.professions:
            info = taxonomy.get_profession_info(name)
            professions.append({
                "name": name,
                "domains": info.get("domains", []),
                "competency_codes": info.get("competency_codes", []),
                "hh_queries": info.get("hh_queries", []),
                "aliases": info.get("aliases", []),
            })
        return {"professions": professions, "total": len(professions)}
    except Exception as e:
        logger.error("get_professions_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/taxonomy/profession/{profession_name}")
@limiter.limit("60/minute")
async def get_profession_detail(request: Request, profession_name: str):
    """Детали профессии: домены, навыки, КРМ-компетенции."""
    try:
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
        taxonomy = ProfessionTaxonomy()
        info = taxonomy.get_profession_info(profession_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Profession '{profession_name}' not found")
        
        skills = list(taxonomy.get_profession_skills(profession_name))
        krm_codes = taxonomy.get_profession_competency_codes(profession_name)
        krm_skills = {}
        for code in krm_codes:
            krm_skills[code] = taxonomy.get_competency_skills(code)
        
        return {
            "name": profession_name,
            "domains": info.get("domains", []),
            "skill_count": len(skills),
            "skills": skills[:100],  # limit response size
            "competency_codes": krm_codes,
            "krm_competencies": {code: {"skill_count": len(s), "skills": s[:20]} for code, s in krm_skills.items()},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_profession_detail_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/taxonomy/profession/{profession_name}/krm-coverage")
@limiter.limit("30/minute")
async def get_profession_krm_coverage(request: Request, profession_name: str, skills: str = ""):
    """Покрытие КРМ-компетенций профессии для заданных навыков."""
    try:
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
        taxonomy = ProfessionTaxonomy()
        user_skills = [s.strip() for s in skills.split(",") if s.strip()] if skills else []
        
        coverage = taxonomy.compute_krm_coverage(profession_name, user_skills)
        if not coverage:
            raise HTTPException(status_code=404, detail=f"No KRM data for '{profession_name}'")
        
        return {
            "profession": profession_name,
            "user_skills": user_skills,
            "competency_coverage": coverage,
            "avg_coverage": round(
                sum(v["coverage"] for v in coverage.values()) / len(coverage) if coverage else 0, 4
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_krm_coverage_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profiles/{profile}/profession-evaluation")
@limiter.limit("30/minute")
async def get_profile_profession_evaluation(request: Request, profile: str):
    """Оценка профиля по профессии с КРМ-покрытием."""
    try:
        if profile not in student_profiles:
            raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")
        
        from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
        
        taxonomy = ProfessionTaxonomy()
        profile_config = taxonomy.get_profile_target(profile)
        if not profile_config:
            raise HTTPException(status_code=404, detail=f"No profession target for '{profile}'")
        
        student = student_profiles[profile]
        
        result = evaluator.evaluate_profile(
            student,
            user_type="student",
            target_domains=profile_config.get("target_domains", []),
            taxonomy=taxonomy,
        )
        
        return {
            "profile": profile,
            "target_profession": profile_config.get("target_profession", ""),
            "target_domains": profile_config.get("target_domains", []),
            "profession_coverage": result.get("profession_coverage", 0),
            "krm_coverage": result.get("krm_coverage", {}),
            "readiness_score": result.get("readiness_score", 0),
            "skill_coverage": result.get("skill_coverage", 0),
            "domain_coverage_score": result.get("domain_coverage_score", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_profile_profession_evaluation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/skills/missing")
@limiter.limit("30/minute")
async def missing_skills(
    request: Request,
    min_frequency: int = Query(1),
    freq: dict[str, int] = Depends(get_skill_freq)
):
    validator = SkillValidator(whitelist=None)
    extracted = {}
    for skill, freq_val in freq.items():
        if skill.lower() not in current_skills_set and freq_val >= min_frequency and validator.validate(skill).is_valid:
            extracted[skill] = freq_val
    sorted_skills = sorted(extracted.items(), key=lambda x: x[1], reverse=True)
    return {"missing_skills": [{"skill": s, "frequency": f} for s, f in sorted_skills]}


@app.get("/api/skills/dead")
@limiter.limit("30/minute")
async def dead_skills(
    request: Request,
    freq: dict[str, int] = Depends(get_skill_freq)
):
    extracted_lower = {s.lower() for s in freq}
    dead = sorted(s for s in current_skills_set if s.lower() not in extracted_lower)
    return {"dead_skills": dead}


# ============================================
# РЕЗУЛЬТАТЫ И ИЗОБРАЖЕНИЯ
# ============================================

@app.get("/api/results/summary")
@limiter.limit("30/minute")
async def get_results_summary(
    request: Request,
    profiles: dict[str, StudentProfile] = Depends(get_student_profiles)
):
    """Возвращает сводку результатов анализа"""
    summary_path = config.DATA_PROCESSED_DIR / "profiles_comparison_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {
        "message": "Результаты анализа не найдены. Запустите gap-анализ.",
        "profiles": list(profiles.keys())
    }


@app.get("/api/results/recommendations/{profile}")
@limiter.limit("30/minute")
async def get_recommendations_result(
    request: Request,
    profile: str,
    profiles: dict[str, StudentProfile] = Depends(get_student_profiles)
):
    """Возвращает рекомендации для профиля"""
    if profile not in profiles:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    
    result_path = config.DATA_DIR / "result" / profile / f"full_recommendations_{profile}.json"
    if result_path.exists():
        try:
            with open(result_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "profile": profile,
        "message": "Рекомендации не найдены. Запустите gap-анализ.",
        "recommendations": []
    }


@app.get("/api/results/images/{profile}/{image_type}")
@limiter.limit("60/minute")
async def get_profile_image(
    request: Request,
    profile: str,
    image_type: str
):
    """Получить изображение визуализации для профиля"""
    safe_types = ["radar", "ml_importance", "cluster_insights", "deficits"]
    
    if image_type not in safe_types:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    image_path = config.DATA_DIR / "result" / profile / f"{image_type}_{profile}.png"
    
    if not image_path.exists():
        image_path = config.DATA_DIR / "result" / f"{image_type}_{profile}.png"
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_type}_{profile}.png")
    
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/results/images/coverage-comparison")
@limiter.limit("30/minute")
async def get_coverage_comparison_image(request: Request):
    """Получить изображение сравнения покрытия профилей"""
    image_path = config.DATA_DIR / "result" / "coverage_comparison.png"
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Coverage comparison image not found")
    
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/results/images/skills-heatmap")
@limiter.limit("30/minute")
async def get_skills_heatmap_image(request: Request):
    """Получить изображение тепловой карты навыков"""
    image_path = config.DATA_DIR / "result" / "skills_heatmap.png"
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Skills heatmap image not found")
    
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/results/images/skill-correlation")
@limiter.limit("30/minute")
async def get_skill_correlation_image(request: Request):
    """Получить изображение корреляции навыков"""
    image_path = config.DATA_DIR / "result" / "skill_correlation_heatmap.png"
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Skill correlation image not found")
    
    return FileResponse(image_path, media_type="image/png")


# ============================================
# ПАЙПЛАЙН И ПЕРЕСТРОЕНИЕ
# ============================================

@app.post("/api/pipeline/{action}", response_model=PipelineResponse)
@limiter.limit("5/minute")
async def run_pipeline_action_sync(
    request: Request,
    action: PipelineAction,
    background_tasks: BackgroundTasks,
    skip_collection: bool = Query(False, description="Пропустить сбор вакансий"),
    run_gap_analysis: bool = Query(True, description="Запустить gap-анализ после обучения"),
    regions: str = Query("0", description="Регионы для сбора вакансий (через запятую)")
):
    """
    Запуск различных действий пайплайна:
    - full-cycle: полный цикл (сбор данных -> обработка -> кластеризация -> обучение -> gap-анализ)
    - rebuild: полная пересборка (очистка кэшей + полный цикл)
    - train-clusters: только обучение кластеров
    - train-model: только обучение LTR-модели
    - gap-analysis: только gap-анализ
    """
    base_dir = Path(__file__).parent.parent
    
    if action == PipelineAction.REBUILD:
        # Запускаем full_rebuild.py
        rebuild_script = base_dir / "scripts" / "full_rebuild.py"
        if not rebuild_script.exists():
            raise HTTPException(status_code=404, detail=f"full_rebuild.py not found at {rebuild_script}")
        
        cmd = ["python", str(rebuild_script)]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        
        if returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Rebuild failed: {stderr[:500]}"
            )
        
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Full rebuild completed",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:]
        )
    
    elif action == PipelineAction.FULL_CYCLE:
        # Запускаем main.py с соответствующими параметрами в фоне
        task_id = f"{action.value}_{int(time.time())}"
        
        background_tasks.add_task(
            run_pipeline_task,
            action,
            task_id,
            skip_collection=skip_collection,
            run_gap_analysis=run_gap_analysis,
            regions=regions
        )
        
        return PipelineResponse(
            status="started",
            message=f"Pipeline {action.value} started in background",
            command=f"python main.py --skip-collection={skip_collection} --run-gap-analysis={run_gap_analysis}",
            exit_code=None,
            output=f"Task ID: {task_id}. Use /api/pipeline/task/{task_id} to check status."
        )
    
    elif action == PipelineAction.TRAIN_CLUSTERS:
        cmd = ["python", str(base_dir / "scripts" / "train_clusters.py"), "--level", "all"]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Cluster training completed" if returncode == 0 else f"Cluster training failed with code {returncode}",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:]
        )
    
    elif action == PipelineAction.TRAIN_MODEL:
        cmd = ["python", str(base_dir / "main.py"), "--train-model"]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Model training completed" if returncode == 0 else f"Model training failed with code {returncode}",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:]
        )
    
    elif action == PipelineAction.GAP_ANALYSIS:
        cmd = ["python", str(base_dir / "main.py"), "--skip-collection", "--run-gap-analysis"]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Gap analysis completed" if returncode == 0 else f"Gap analysis failed with code {returncode}",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:]
        )
    
    return PipelineResponse(
        status="error",
        message=f"Unknown action: {action}",
        exit_code=-1
    )


@app.get("/api/pipeline/task/{task_id}")
@limiter.limit("60/minute")
async def get_pipeline_task_status(
    request: Request,
    task_id: str
):
    """Получить статус фоновой задачи пайплайна"""
    if task_id not in pipeline_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return pipeline_tasks[task_id]


@app.get("/api/pipeline/tasks")
@limiter.limit("30/minute")
async def list_pipeline_tasks(
    request: Request,
    limit: int = Query(10, ge=1, le=50)
):
    """Список последних задач пайплайна"""
    tasks = list(pipeline_tasks.values())
    tasks.reverse()  # Последние сверху
    return {"tasks": tasks[:limit], "total": len(tasks)}


@app.get("/api/pipeline/status")
@limiter.limit("30/minute")
async def get_pipeline_status(request: Request):
    """Проверяет статус различных компонентов пайплайна"""
    base_dir = Path(__file__).parent.parent
    
    # Проверяем существование ключевых артефактов
    clusters_exist = {
        "junior": (base_dir / "data" / "cache" / "clusters" / "vacancy_clusters_junior.pkl").exists(),
        "middle": (base_dir / "data" / "cache" / "clusters" / "vacancy_clusters_middle.pkl").exists(),
        "senior": (base_dir / "data" / "cache" / "clusters" / "vacancy_clusters_senior.pkl").exists(),
    }
    
    model_exists = (base_dir / "data" / "models" / "ltr_ranker_xgb_regressor.joblib").exists()
    
    recommendations_exist = {
        "base": (base_dir / "data" / "result" / "base" / "full_recommendations_base.json").exists(),
        "dc": (base_dir / "data" / "result" / "dc" / "full_recommendations_dc.json").exists(),
        "top_dc": (base_dir / "data" / "result" / "top_dc" / "full_recommendations_top_dc.json").exists(),
    }
    
    skill_weights_exist = (base_dir / "data" / "processed" / "skill_weights.json").exists()
    
    return {
        "clusters": clusters_exist,
        "clusters_all_ready": all(clusters_exist.values()),
        "ltr_model": model_exists,
        "recommendations": recommendations_exist,
        "recommendations_all_ready": all(recommendations_exist.values()),
        "skill_weights": skill_weights_exist,
        "scripts": {
            "full_rebuild": (base_dir / "scripts" / "full_rebuild.py").exists(),
            "train_clusters": (base_dir / "scripts" / "train_clusters.py").exists(),
        }
    }


@app.post("/api/pipeline/rebuild")
@limiter.limit("2/minute")
async def pipeline_rebuild(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Короткий алиас для полной пересборки (запускается в фоне)"""
    base_dir = Path(__file__).parent.parent
    rebuild_script = base_dir / "scripts" / "full_rebuild.py"
    
    if not rebuild_script.exists():
        raise HTTPException(status_code=404, detail=f"full_rebuild.py not found at {rebuild_script}")
    
    task_id = f"rebuild_{int(time.time())}"
    
    background_tasks.add_task(
        run_pipeline_task,
        PipelineAction.REBUILD,
        task_id
    )
    
    return {
        "status": "started",
        "message": "Full rebuild started in background",
        "task_id": task_id,
        "check_url": f"/api/pipeline/task/{task_id}"
    }


@app.post("/api/pipeline/refresh-cache")
@limiter.limit("5/minute")
async def refresh_cache(request: Request):
    """Очищает кэши и перезапускает пайплайн"""
    base_dir = Path(__file__).parent.parent
    
    # Очищаем кэши
    cache_dirs = [
        base_dir / "data" / "cache" / "embeddings",
        base_dir / "data" / "cache" / "clusters",
    ]
    
    removed = []
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            removed.append(str(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Очищаем parsed_skills.pkl
    parsed_skills = base_dir / "data" / "cache" / "parsed_skills.pkl"
    if parsed_skills.exists():
        parsed_skills.unlink()
        removed.append(str(parsed_skills))
    
    return {
        "status": "success",
        "message": "Cache cleared",
        "removed": removed,
        "next_step": "Run POST /api/pipeline/full-cycle?skip_collection=false to rebuild"
    }


@app.post("/api/pipeline/reload-api")
@limiter.limit("3/minute")
async def reload_api(request: Request):
    """Перезагружает данные API (без перезапуска сервера)"""
    try:
        # Перезагружаем данные в фоне
        asyncio.create_task(reload_api_data())
        return {
            "status": "started",
            "message": "API data reload started. Check /api/status for completion."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def reload_api_data():
    """Фоновая перезагрузка данных API"""
    global evaluator, recommendation_engine, trend_analyzer, student_profiles
    global skill_weights, hybrid_weights, competency_mapping, skill_freq, taxonomy, current_skills_set, basic_vacancies
    
    logger.info("Reloading API data...")
    try:
        # Повторяем инициализацию из startup
        await startup()
        logger.info("API data reloaded successfully")
    except Exception as e:
        logger.error("Ошибка перезагрузки API", error=str(e))


# ============================================
# ВАКАНСИИ
# ============================================

@app.get("/api/vacancies")
@limiter.limit("60/minute")
async def get_vacancies(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Количество вакансий"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    experience: str | None = Query(None, description="Фильтр по опыту: junior, middle, senior"),
    search: str | None = Query(None, description="Поиск по названию"),
    vacancies: list = Depends(get_basic_vacancies)
):
    """Получить список вакансий с фильтрами"""
    
    filtered = vacancies.copy()

    if experience:
        exp_lower = experience.lower()
        filtered = [
            v for v in filtered
            if (isinstance(v.get("experience"), dict) and
                exp_lower in v["experience"].get("id", "").lower()) or
               (isinstance(v.get("experience"), str) and
                exp_lower in v["experience"].lower()) or
               exp_lower in v.get("name", "").lower()
        ]

    if search:
        search_lower = search.lower()
        filtered = [
            v for v in filtered
            if search_lower in v.get("name", "").lower() or
               search_lower in v.get("description", "").lower()
        ]

    total = len(filtered)
    items = filtered[offset:offset + limit]

    formatted_items = []
    for vac in items:
        skills = []
        if "extracted_skills" in vac:
            skills = vac["extracted_skills"][:10]

        exp = "middle"
        if "experience" in vac:
            exp_obj = vac["experience"]
            if isinstance(exp_obj, dict):
                exp_id = exp_obj.get("id", "").lower()
                if "junior" in exp_id or "less1" in exp_id or "no_experience" in exp_id:
                    exp = "junior"
                elif "senior" in exp_id or "morethan10" in exp_id:
                    exp = "senior"

        salary_from = None
        salary_to = None
        salary_currency = "RUR"
        if "salary" in vac and vac["salary"]:
            sal = vac["salary"]
            salary_from = sal.get("from")
            salary_to = sal.get("to")
            salary_currency = sal.get("currency", "RUR")

        employer_name = "Не указано"
        employer_logo = None
        if "employer" in vac and vac["employer"]:
            emp = vac["employer"]
            employer_name = emp.get("name", "Не указано")
            if "logo_urls" in emp and emp["logo_urls"]:
                employer_logo = emp["logo_urls"].get("240") or emp["logo_urls"].get("90")

        formatted_items.append({
            "id": vac.get("id"),
            "name": vac.get("name", "Без названия"),
            "experience": exp,
            "salary_from": salary_from,
            "salary_to": salary_to,
            "salary_currency": salary_currency,
            "employer_name": employer_name,
            "employer_logo": employer_logo,
            "area": vac.get("area", {}).get("name", "Не указано") if isinstance(vac.get("area"), dict) else "Не указано",
            "published_at": vac.get("published_at"),
            "alternate_url": vac.get("alternate_url"),
            "skills": skills,
            "snippet": vac.get("snippet", {})
        })

    return {
        "items": formatted_items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }


@app.get("/api/vacancies/{vacancy_id}")
@limiter.limit("60/minute")
async def get_vacancy_detail(
    request: Request,
    vacancy_id: str,
    vacancies: list = Depends(get_basic_vacancies)
):
    """Получить детальную информацию о вакансии"""
    
    for vac in vacancies:
        if vac.get("id") == vacancy_id:
            skills = []
            if "extracted_skills" in vac:
                skills = vac["extracted_skills"]

            return {
                "id": vac.get("id"),
                "name": vac.get("name"),
                "description": vac.get("description", ""),
                "experience": vac.get("experience"),
                "salary": vac.get("salary"),
                "employer": vac.get("employer"),
                "area": vac.get("area"),
                "published_at": vac.get("published_at"),
                "alternate_url": vac.get("alternate_url"),
                "skills": skills,
                "schedule": vac.get("schedule"),
                "employment": vac.get("employment"),
                "key_skills": vac.get("key_skills", []),
                "snippet": vac.get("snippet")
            }

    raise HTTPException(status_code=404, detail="Вакансия не найдена")


@app.get("/api/vacancies/stats/summary")
@limiter.limit("30/minute")
async def get_vacancies_stats(
    request: Request,
    vacancies: list = Depends(get_basic_vacancies)
):
    """Получить статистику по вакансиям"""
    
    total = len(vacancies)
    junior = 0
    middle = 0
    senior = 0
    salaries = []

    for vac in vacancies:
        exp = "middle"
        if "experience" in vac:
            exp_obj = vac["experience"]
            if isinstance(exp_obj, dict):
                exp_id = exp_obj.get("id", "").lower()
                if "junior" in exp_id or "less1" in exp_id:
                    exp = "junior"
                elif "senior" in exp_id or "morethan10" in exp_id:
                    exp = "senior"

        if exp == "junior":
            junior += 1
        elif exp == "senior":
            senior += 1
        else:
            middle += 1

        if "salary" in vac and vac["salary"]:
            sal = vac["salary"]
            if sal.get("from"):
                salaries.append(sal["from"])
            if sal.get("to"):
                salaries.append(sal["to"])

    avg_salary = sum(salaries) / len(salaries) if salaries else 0

    return {
        "total": total,
        "by_experience": {
            "junior": junior,
            "middle": middle,
            "senior": senior
        },
        "salary": {
            "average": round(avg_salary, 0),
            "min": min(salaries) if salaries else 0,
            "max": max(salaries) if salaries else 0,
            "count": len(salaries)
        }
    }