"""Pipeline: full cycle, rebuild, train, gap analysis, tasks."""

import asyncio
import base64
import os
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path

import requests
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.api_responses import (
    CacheRefreshResponse,
    GapProgressResponse,
    PipelineTaskStatus,
    PipelineTaskListResponse,
)

logger = structlog.get_logger("api")

_areas_cache = None
GAP_PROGRESS_FILE = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "gap_progress.json"
PIPELINE_PROGRESS_FILE = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "pipeline_progress.json"


def _read_progress_file(path: Path) -> dict:
    try:
        if path.exists():
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _read_gap_progress() -> dict:
    return _read_progress_file(GAP_PROGRESS_FILE)


def _read_pipeline_progress() -> dict:
    return _read_progress_file(PIPELINE_PROGRESS_FILE)


def _resolve_area_ids(regions_str: str) -> str:
    """Convert comma-separated region names to HH area IDs."""
    parts = [p.strip() for p in regions_str.split(",") if p.strip()]
    ids = []
    for p in parts:
        if p.isdigit():
            ids.append(p)
        else:
            aid = _resolve_city_name(p)
            if aid:
                ids.append(str(aid))
            else:
                logger.warning("region_not_found", region=p)
                ids.append("0")
    return ",".join(ids) if ids else "0"

def _resolve_city_name(name: str) -> int | None:
    """Look up HH area ID by city name via API."""
    global _areas_cache
    if _areas_cache is None:
        try:
            resp = requests.get("https://api.hh.ru/areas", timeout=10)
            resp.encoding = "utf-8"
            _areas_cache = resp.json()
        except Exception as e:
            logger.error("areas_fetch_failed", error=str(e))
            _areas_cache = []
    name_lower = name.strip().lower()
    def _search(areas):
        for a in areas:
            if a.get("name", "").strip().lower() == name_lower:
                return int(a["id"])
            child = _search(a.get("areas", []))
            if child is not None:
                return child
        return None
    return _search(_areas_cache)


logger = structlog.get_logger("api")

router = APIRouter(tags=["pipeline"])
limiter = Limiter(key_func=get_remote_address)

pipeline_tasks: dict[str, PipelineTaskStatus] = {}


class PipelineAction(str, Enum):
    FULL_CYCLE = "full-cycle"
    REBUILD = "rebuild"
    TRAIN_CLUSTERS = "train-clusters"
    TRAIN_MODEL = "train-model"
    GAP_ANALYSIS = "gap-analysis"


class PipelineResponse(BaseModel):
    status: str
    message: str
    command: str | None = None
    exit_code: int | None = None
    output: str | None = None


async def run_command(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str, str]:
    def _run():
        env = dict(os.environ,
            PYTHONIOENCODING="utf-8",
            HF_HUB_VERBOSITY="error",
            HF_HUB_DISABLE_PROGRESS_BARS="1",
            TRANSFORMERS_VERBOSITY="error",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        kwargs = dict(
            args=cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if timeout:
            kwargs["timeout"] = timeout
        return subprocess.run(**kwargs)
    try:
        proc = await asyncio.to_thread(_run)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def _make_task_status(task_id, status, message, started_at, completed_at=None, output=None, step=0, sub_progress=None):
    return PipelineTaskStatus(
        task_id=task_id, status=status, message=message,
        started_at=started_at, completed_at=completed_at or time.time(),
        output=output, step=step, sub_progress=sub_progress,
    )

async def _rotate_progress(task_id: str, step: int = 4):
    i = 0
    phases_map = {
        1: ["Сбор вакансий с hh.ru...", "Оценка качества...", "Сохранение результатов..."],
        2: ["Кластеризация вакансий..."],
        3: ["Обучение модели ранжирования...", "Проверка качества..."],
        4: ["GAP-анализ: загрузка моделей...", "GAP-анализ: вычисление эмбеддингов...", "GAP-анализ: сравнение профилей...", "GAP-анализ: генерация рекомендаций..."],
    }
    prev_pct = -1
    t = pipeline_tasks.get(task_id)
    while t and t.status == "running":
        if step == 4:
            gp = _read_gap_progress()
            if gp and gp.get("pct", 0) > 0:
                pct = int(gp["pct"])
                if pct > prev_pct:
                    t.sub_progress = pct
                    prev_pct = pct
                t.message = gp.get("message", phases_map[4][i % len(phases_map[4])])
            else:
                t.message = phases_map[4][i % len(phases_map[4])]
                i += 1
        else:
            pp = _read_pipeline_progress()
            if pp and pp.get("pct", 0) > 0:
                pct = int(pp["pct"])
                if pct > prev_pct:
                    t.sub_progress = pct
                    prev_pct = pct
                t.message = pp.get("message", phases_map[step][i % len(phases_map[step])])
            else:
                t.message = phases_map[step][i % len(phases_map[step])]
                i += 1
        await asyncio.sleep(5)


async def run_pipeline_task(action: PipelineAction, task_id: str, **kwargs):
    base_dir = Path(__file__).parent.parent.parent.parent
    started_at = time.time()
    pipeline_tasks[task_id] = _make_task_status(task_id, "running", "Запуск сбора...", started_at, step=0)
    try:
        ret = -1
        out = err = ""
        if action == PipelineAction.REBUILD:
            script = base_dir / "scripts" / "full_rebuild.py"
            pipeline_tasks[task_id].message = "Запуск полной пересборки..."
            ret, out, err = await run_command(["python", str(script)], base_dir)
        elif action == PipelineAction.FULL_CYCLE:
            gap = kwargs.get("run_gap_analysis", True)
            regions_raw = kwargs.get("regions", "113")
            query = kwargs.get("query", "")
            max_pages = kwargs.get("max_pages", 20)
            period = kwargs.get("period", 30)

            regions = _resolve_area_ids(regions_raw) if regions_raw not in ("0", "") else "113"
            logger.info("regions_resolved", raw=regions_raw, resolved=regions)

            progress_rotator = asyncio.ensure_future(_rotate_progress(task_id, step=1))
            pipeline_tasks[task_id].step = 1
            cmd = ["python", str(base_dir / "main.py"), "--it-sector", "--regions", regions, "--excel",
                   "--max-pages", str(max_pages), "--period", str(period)]
            if query:
                enc = base64.b64encode(query.encode("utf-8")).decode("ascii")
                cmd.extend(["--query", f"b64:{enc}"])
            ret, out, err = await run_command(cmd, base_dir)
            progress_rotator.cancel()
            import re
            vac_match = re.search(r"Найдено (\d+) базовых вакансий", out or "")
            if vac_match:
                pipeline_tasks[task_id].message = f"Собрано {vac_match.group(1)} вакансий. Кластеризация..."
            if ret == 0:
                progress_rotator = asyncio.ensure_future(_rotate_progress(task_id, step=2))
                pipeline_tasks[task_id].step = 2
                ret, out, err = await run_command(
                    ["python", str(base_dir / "scripts" / "train_clusters.py"), "--level", "all"], base_dir, timeout=600)
                progress_rotator.cancel()
            if ret == 0:
                progress_rotator = asyncio.ensure_future(_rotate_progress(task_id, step=3))
                model_path = base_dir / "models" / "ltr_ranker_xgb_regressor.joblib"
                if model_path.exists():
                    pipeline_tasks[task_id].message = "Проверка модели ранжирования..."
                else:
                    pipeline_tasks[task_id].message = "Обучение модели ранжирования..."
                pipeline_tasks[task_id].step = 3
                ret, out, err = await run_command(
                    ["python", str(base_dir / "main.py"), "--train-model"], base_dir, timeout=600)
                progress_rotator.cancel()
            if ret == 0 and gap:
                progress_rotator = asyncio.ensure_future(_rotate_progress(task_id, step=4))
                pipeline_tasks[task_id].step = 4
                ret, out, err = await run_command(
                    ["python", str(base_dir / "main.py"), "--skip-collection", "--run-gap-analysis"], base_dir, timeout=600)
                progress_rotator.cancel()
        elif action == PipelineAction.TRAIN_CLUSTERS:
            pipeline_tasks[task_id].message = "Кластеризация вакансий..."
            ret, out, err = await run_command(
                ["python", str(base_dir / "scripts" / "train_clusters.py"), "--level", "all"], base_dir)
        elif action == PipelineAction.TRAIN_MODEL:
            pipeline_tasks[task_id].message = "Обучение модели ранжирования..."
            ret, out, err = await run_command(
                ["python", str(base_dir / "main.py"), "--train-model"], base_dir)
        elif action == PipelineAction.GAP_ANALYSIS:
            gap_progress = asyncio.ensure_future(_rotate_progress(task_id, step=4))
            ret, out, err = await run_command(
                ["python", str(base_dir / "main.py"), "--skip-collection", "--run-gap-analysis"], base_dir)
            gap_progress.cancel()
        else:
            pipeline_tasks[task_id] = _make_task_status(
                task_id, "failed", f"Неизвестное действие: {action}", started_at, output="")
            return
            msg = (err or out)[-500:] if ret != 0 else ""  # show error details only on failure
            if ret == 0:
                if action == PipelineAction.FULL_CYCLE:
                    msg = "Все этапы выполнены. Данные обновлены."
                elif action == PipelineAction.GAP_ANALYSIS:
                    msg = "GAP-анализ завершен. Отчёты готовы."
                elif action == PipelineAction.TRAIN_CLUSTERS:
                    msg = "Кластеризация завершена."
                elif action == PipelineAction.TRAIN_MODEL:
                    msg = "Модель обучена."
                elif action == PipelineAction.REBUILD:
                    msg = "Пересборка завершена."
                else:
                    msg = "Операция завершена."
            status = "completed" if ret == 0 else "failed"
            pipeline_tasks[task_id] = _make_task_status(
                task_id, status, msg[:500], started_at, output=msg)
            if ret == 0 and action == PipelineAction.FULL_CYCLE:
                try:
                    await reload_api_data()
                except Exception as reload_err:
                    logger.error("reload_after_pipeline_failed", error=str(reload_err))
    except Exception as e:
        logger.exception("Pipeline task failed", task_id=task_id)
        pipeline_tasks[task_id] = _make_task_status(
            task_id, "failed", str(e), started_at, output="")


async def reload_api_data():
    logger.info("Reloading API data...")
    try:
        from src.api_pkg.startup import run_startup
        from src.api_pkg import deps as deps_module
        import importlib

        importlib.reload(deps_module)
        app = None
        await run_startup(app)
        logger.info("API data reloaded successfully")
    except Exception as e:
        logger.error("Ошибка перезагрузки API", error=str(e))


@router.post("/api/pipeline/{action}", response_model=PipelineResponse)
@limiter.limit("5/minute")
async def run_pipeline_action_sync(
    request: Request,
    action: PipelineAction,
    background_tasks: BackgroundTasks,
    skip_collection: bool = Query(False, description="Пропустить сбор вакансий"),
    run_gap_analysis: bool = Query(
        True, description="Запустить gap-анализ после обучения"
    ),
    regions: str = Query("0", description="Регионы для сбора вакансий (через запятую)"),
    query: str = Query("", description="Профессия/запрос для поиска вакансий"),
    max_pages: int = Query(20, ge=1, le=100, description="Количество страниц"),
    period: int = Query(30, ge=1, le=365, description="Период поиска в днях"),
):
    base_dir = Path(__file__).parent.parent.parent.parent
    if action == PipelineAction.REBUILD:
        rebuild_script = base_dir / "scripts" / "full_rebuild.py"
        if not rebuild_script.exists():
            raise HTTPException(
                status_code=404, detail=f"full_rebuild.py not found at {rebuild_script}"
            )
        cmd = ["python", str(rebuild_script)]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Full rebuild completed",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:],
        )
    elif action == PipelineAction.FULL_CYCLE:
        task_id = f"{action.value}_{int(time.time())}"
        started_at = time.time()
        pipeline_tasks[task_id] = _make_task_status(task_id, "running", "Запуск сбора...", started_at, step=0)
        background_tasks.add_task(
            run_pipeline_task,
            action,
            task_id,
            skip_collection=skip_collection,
            run_gap_analysis=run_gap_analysis,
            regions=regions,
            query=query,
            max_pages=max_pages,
            period=period,
        )
        return PipelineResponse(
            status="started",
            message=f"Pipeline {action.value} started in background",
            command=f"python main.py --skip-collection={skip_collection} --run-gap-analysis={run_gap_analysis}",
            exit_code=None,
            output=f"Task ID: {task_id}. Use /api/pipeline/task/{task_id} to check status",
        )
    elif action == PipelineAction.TRAIN_CLUSTERS:
        cmd = [
            "python",
            str(base_dir / "scripts" / "train_clusters.py"),
            "--level",
            "all",
        ]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Cluster training completed"
            if returncode == 0
            else f"Cluster training failed with code {returncode}",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:],
        )
    elif action == PipelineAction.TRAIN_MODEL:
        cmd = ["python", str(base_dir / "main.py"), "--train-model"]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Model training completed"
            if returncode == 0
            else f"Model training failed with code {returncode}",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:],
        )
    elif action == PipelineAction.GAP_ANALYSIS:
        cmd = [
            "python",
            str(base_dir / "main.py"),
            "--skip-collection",
            "--run-gap-analysis",
        ]
        returncode, stdout, stderr = await run_command(cmd, base_dir)
        return PipelineResponse(
            status="success" if returncode == 0 else "failed",
            message="Gap analysis completed"
            if returncode == 0
            else f"Gap analysis failed with code {returncode}",
            command=" ".join(cmd),
            exit_code=returncode,
            output=stdout[-1000:] if stdout else stderr[-1000:],
        )
    return PipelineResponse(
        status="error", message=f"Unknown action: {action}", exit_code=-1
    )


@router.get("/api/pipeline/task/{task_id}", response_model=PipelineTaskStatus)
@limiter.limit("60/minute")
async def get_pipeline_task_status(request: Request, task_id: str):
    if task_id not in pipeline_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return pipeline_tasks[task_id]


@router.get("/api/pipeline/tasks", response_model=PipelineTaskListResponse)
@limiter.limit("30/minute")
async def list_pipeline_tasks(request: Request, limit: int = Query(10, ge=1, le=50)):
    tasks = list(pipeline_tasks.values())
    tasks.reverse()
    return {"tasks": tasks[:limit], "total": len(tasks)}


@router.get("/api/pipeline/status", response_model=dict)
@limiter.limit("30/minute")
async def get_pipeline_status(request: Request):
    base_dir = Path(__file__).parent.parent.parent.parent
    clusters_exist = {
        "junior": (
            base_dir / "data" / "cache" / "clusters" / "vacancy_clusters_junior.joblib"
        ).exists(),
        "middle": (
            base_dir / "data" / "cache" / "clusters" / "vacancy_clusters_middle.joblib"
        ).exists(),
        "senior": (
            base_dir / "data" / "cache" / "clusters" / "vacancy_clusters_senior.joblib"
        ).exists(),
    }
    model_exists = (
        base_dir / "data" / "models" / "ltr_ranker_xgb_regressor.joblib"
    ).exists()
    recommendations_exist = {
        "base": (
            base_dir / "data" / "result" / "base" / "full_recommendations_base.json"
        ).exists(),
        "dc": (
            base_dir / "data" / "result" / "dc" / "full_recommendations_dc.json"
        ).exists(),
        "top_dc": (
            base_dir / "data" / "result" / "top_dc" / "full_recommendations_top_dc.json"
        ).exists(),
    }
    skill_weights_exist = (
        base_dir / "data" / "processed" / "skill_weights.json"
    ).exists()
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
        },
    }


@router.post("/api/pipeline/rebuild", response_model=PipelineResponse)
@limiter.limit("2/minute")
async def pipeline_rebuild(request: Request, background_tasks: BackgroundTasks):
    base_dir = Path(__file__).parent.parent.parent.parent
    rebuild_script = base_dir / "scripts" / "full_rebuild.py"
    if not rebuild_script.exists():
        raise HTTPException(
            status_code=404, detail=f"full_rebuild.py not found at {rebuild_script}"
        )
    task_id = f"rebuild_{int(time.time())}"
    background_tasks.add_task(run_pipeline_task, PipelineAction.REBUILD, task_id)
    return PipelineResponse(
        status="started",
        message="Full rebuild started in background",
        command=f"python {rebuild_script}",
        output=f"Task ID: {task_id}. Check /api/pipeline/task/{task_id} for status",
    )


@router.post("/api/pipeline/refresh-cache", response_model=CacheRefreshResponse)
@limiter.limit("5/minute")
async def refresh_cache(request: Request):
    base_dir = Path(__file__).parent.parent.parent.parent
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
    parsed_skills = base_dir / "data" / "cache" / "parsed_skills.joblib"
    if parsed_skills.exists():
        parsed_skills.unlink()
        removed.append(str(parsed_skills))
    return {
        "status": "success",
        "message": "Cache cleared",
        "removed": removed,
        "next_step": "Run POST /api/pipeline/full-cycle?skip_collection=false to rebuild",
    }


@router.post("/api/pipeline/reload-api", response_model=PipelineResponse)
@limiter.limit("3/minute")
async def reload_api(request: Request):
    try:
        asyncio.create_task(reload_api_data())
        return PipelineResponse(
            status="started",
            message="API data reload started. Check /api/status for completion.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/pipeline/gap-progress/{task_id}", response_model=GapProgressResponse)
@limiter.limit("60/minute")
async def get_gap_progress(task_id: str, request: Request):
    gp = _read_gap_progress()
    if gp:
        return GapProgressResponse(
            pct=gp.get("pct", 0),
            message=gp.get("message", ""),
            stage=gp.get("stage", ""),
            exists=True,
        )
    return GapProgressResponse(exists=False)
