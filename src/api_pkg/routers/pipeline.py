"""Pipeline: full cycle, rebuild, train, gap analysis, tasks, WebSocket."""

import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path

import requests
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
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
TASKS_STORE_FILE = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "pipeline_tasks.json"


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
pipeline_procs: dict[str, subprocess.Popen] = {}


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


async def run_command(cmd: list[str], cwd: Path | None = None, timeout: int | None = None, task_id: str | None = None) -> tuple[int, str, str]:
    def _run():
        import tempfile
        env = dict(os.environ,
            PYTHONIOENCODING="utf-8",
            HF_HUB_VERBOSITY="error",
            HF_HUB_DISABLE_PROGRESS_BARS="1",
            TRANSFORMERS_VERBOSITY="error",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        tmp_out = tmp_err = None
        out_path = err_path = None
        try:
            tmp_out = tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8", errors="replace")
            tmp_err = tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8", errors="replace")
            out_path, err_path = tmp_out.name, tmp_err.name
            proc = subprocess.Popen(
                args=cmd,
                stdout=tmp_out,
                stderr=tmp_err,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(cwd) if cwd else None,
                env=env,
            )
            if task_id:
                pipeline_procs[task_id] = proc
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            tmp_out.close()
            tmp_err.close()
            with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                stdout = f.read()
            with open(err_path, "r", encoding="utf-8", errors="replace") as f:
                stderr = f.read()
            rc = proc.returncode
            if rc != 0:
                stderr = (stderr or "") + (f"\nTIMEOUT" if rc == -1 else "")
            return rc, stdout, stderr
        except Exception as e:
            return -1, "", str(e)
        finally:
            for p in (out_path, err_path):
                if p:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
            if task_id and task_id in pipeline_procs:
                del pipeline_procs[task_id]
    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        return -1, "", str(e)


def _save_tasks():
    try:
        TASKS_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {tid: t.model_dump() for tid, t in pipeline_tasks.items()}
        with open(TASKS_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass

def _load_tasks():
    try:
        if TASKS_STORE_FILE.exists():
            with open(TASKS_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            for tid, tdata in data.items():
                t = PipelineTaskStatus.model_validate(tdata)
                status = getattr(t, "status", "")
                started = getattr(t, "started_at", 0) or 0
                if status == "running" and now - started > 3600:
                    t.status = "failed"
                    t.message = "Прерван перезапуском сервера"
                pipeline_tasks[tid] = t
            _save_tasks()
            logger.info("tasks_restored", count=len(data))
    except Exception as e:
        logger.warning("tasks_restore_failed", error=str(e))

_load_tasks()

def _make_task_status(task_id, status, message, started_at, completed_at=None, output=None, step=0, sub_progress=None):
    return PipelineTaskStatus(
        task_id=task_id, status=status, message=message,
        started_at=started_at, completed_at=completed_at or time.time(),
        output=output, step=step, sub_progress=sub_progress,
    )

def _pct_to_step(pct: int) -> int:
    if pct < 18: return 1
    if pct < 60: return 2
    if pct < 70: return 3
    return 4

async def _rotate_progress(task_id: str, step: int = 1):
    i = 0
    phases_map = {
        1: ["Поиск вакансий на hh.ru...", "Загрузка деталей...", "Оценка качества...", "Извлечение навыков..."],
        2: ["Очистка весов...", "Построение уровней..."],
        3: ["Обучение кластеров...", "Обучение LTR-модели..."],
        4: ["GAP-анализ: инициализация...", "GAP-анализ: оценка профилей...", "GAP-анализ: генерация рекомендаций...", "Генерация графиков..."],
    }
    prev_pct = -1
    while True:
        t = pipeline_tasks.get(task_id)
        if not t or t.status != "running":
            break
        pct = -1
        msg = ""
        pp = _read_pipeline_progress()
        if pp and pp.get("pct") is not None:
            pct = int(pp["pct"])
            msg = pp.get("message", "")
        # Interpolate gap_progress sub-pct into pipeline scale (70-92) for step 4
        if _pct_to_step(pct) >= 4:
            gp = _read_gap_progress()
            if gp and gp.get("pct", 0) > 0:
                gp_pct = int(gp["pct"])
                interpolated = 70 + (92 - 70) * gp_pct / 100
                if interpolated > pct:
                    pct = int(interpolated)
                    msg = gp.get("message", msg)
        if pct >= 0:
            if pct > prev_pct:
                t.sub_progress = pct
                prev_pct = pct
                t.step = _pct_to_step(pct)
            elif pct < prev_pct:
                prev_pct = -1
                t.sub_progress = 0
                t.step = _pct_to_step(pct)
            if msg:
                t.message = msg
            else:
                t.message = phases_map.get(t.step, phases_map[1])[i % len(phases_map.get(t.step, phases_map[1]))]
            logs = pp.get("logs", [])
            if logs:
                t.logs = logs
        _save_tasks()
        await asyncio.sleep(5)


async def run_pipeline_task(action: PipelineAction, task_id: str, **kwargs):
    base_dir = Path(__file__).parent.parent.parent.parent
    started_at = time.time()
    pipeline_tasks[task_id] = _make_task_status(task_id, "running", "Запуск сбора...", started_at, step=0)
    _save_tasks()
    try:
        ret = -1
        out = err = ""
        if action == PipelineAction.REBUILD:
            script = base_dir / "scripts" / "full_rebuild.py"
            pipeline_tasks[task_id].message = "Запуск полной пересборки..."
            ret, out, err = await run_command(["python", str(script)], base_dir, task_id=task_id)
        elif action == PipelineAction.FULL_CYCLE:
            gap = kwargs.get("run_gap_analysis", True)
            regions_raw = kwargs.get("regions", "113")
            query = kwargs.get("query", "")
            max_pages = kwargs.get("max_pages", 20)
            period = kwargs.get("period", 30)

            regions = _resolve_area_ids(regions_raw) if regions_raw not in ("0", "") else "113"
            logger.info("regions_resolved", raw=regions_raw, resolved=regions)

            # Clean stale progress files so the rotator doesn't pick up old data
            try:
                for fp in (PIPELINE_PROGRESS_FILE, GAP_PROGRESS_FILE):
                    if fp.exists():
                        fp.unlink()
            except Exception as e:
                logger.warning("clean_progress_failed", error=str(e))

            progress_rotator = asyncio.ensure_future(_rotate_progress(task_id, step=1))
            pipeline_tasks[task_id].step = 1
            cmd = ["python", str(base_dir / "main.py"), "--regions", regions, "--excel",
                   "--max-pages", str(max_pages), "--period", str(period)]
            if not gap:
                cmd.append("--skip-gap-analysis")
            if query:
                enc = base64.b64encode(query.encode("utf-8")).decode("ascii")
                cmd.extend(["--query", f"b64:{enc}"])
            else:
                cmd.append("--it-sector")
            ret, out, err = await run_command(cmd, base_dir, timeout=1200, task_id=task_id)
            progress_rotator.cancel()
        elif action == PipelineAction.TRAIN_CLUSTERS:
            pipeline_tasks[task_id].message = "Кластеризация вакансий..."
            ret, out, err = await run_command(
                ["python", str(base_dir / "scripts" / "train_clusters.py"), "--level", "all"], base_dir, task_id=task_id)
        elif action == PipelineAction.TRAIN_MODEL:
            pipeline_tasks[task_id].message = "Обучение модели ранжирования..."
            ret, out, err = await run_command(
                ["python", str(base_dir / "main.py"), "--train-model"], base_dir, task_id=task_id)
        elif action == PipelineAction.GAP_ANALYSIS:
            gap_progress = asyncio.ensure_future(_rotate_progress(task_id, step=4))
            ret, out, err = await run_command(
                ["python", str(base_dir / "main.py"), "--skip-collection", "--run-gap-analysis"], base_dir, task_id=task_id)
            gap_progress.cancel()
        else:
            pipeline_tasks[task_id] = _make_task_status(
                task_id, "failed", f"Неизвестное действие: {action}", started_at, output="")
            _save_tasks()
            return

        # --- Common completion logic for all known actions ---
        msg = (err or out)[-500:] if ret != 0 else ""
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
        t = pipeline_tasks.get(task_id)
        if t and t.status == "cancelled":
            return
        pipeline_tasks[task_id] = _make_task_status(
            task_id, status, msg[:500], started_at, output=msg)
        _save_tasks()
        if ret == 0 and action == PipelineAction.FULL_CYCLE:
            try:
                await reload_api_data()
            except Exception as reload_err:
                logger.error("reload_after_pipeline_failed", error=str(reload_err))
    except Exception as e:
        t = pipeline_tasks.get(task_id)
        if t and t.status == "cancelled":
            return
        logger.exception("Pipeline task failed", task_id=task_id)
        pipeline_tasks[task_id] = _make_task_status(
            task_id, "failed", str(e), started_at, output="")
        _save_tasks()


async def reload_api_data():
    logger.info("Reloading API data...")
    try:
        from src.api_pkg.startup import run_startup
        await run_startup(None)
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
        req_id = getattr(request.state, "request_id", "unknown")
        task_id = f"{action.value}_{int(time.time())}"
        started_at = time.time()
        logger.info("pipeline_scheduled", request_id=req_id, task_id=task_id)
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


@router.get("/api/pipeline/active", response_model=PipelineTaskStatus | None)
@limiter.limit("30/minute")
async def get_active_pipeline_task(request: Request):
    for t in pipeline_tasks.values():
        if t.status == "running":
            return t
    return None


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


@router.post("/api/pipeline/cancel/{task_id}")
@limiter.limit("10/minute")
async def cancel_pipeline_task(task_id: str, request: Request):
    if task_id not in pipeline_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    t = pipeline_tasks[task_id]
    if t.status != "running":
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not running (status: {t.status})")
    t.status = "cancelled"
    t.message = "Отменено пользователем"
    t.completed_at = time.time()
    _save_tasks()
    proc = pipeline_procs.pop(task_id, None)
    if proc:
        try:
            proc.kill()
        except Exception:
            pass
    return {"status": "cancelled", "message": "Pipeline остановлен"}


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


# === WebSocket: real-time pipeline progress ===
_ws_clients: set[WebSocket] = set()


@router.websocket("/api/pipeline/ws")
async def pipeline_ws(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        prev_data = ""
        while True:
            raw = None
            for fp in (PIPELINE_PROGRESS_FILE, GAP_PROGRESS_FILE):
                if fp.exists():
                    try:
                        raw = json.loads(fp.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        pass
            if raw:
                text = json.dumps(raw, ensure_ascii=False)
                if text != prev_data:
                    prev_data = text
                    await websocket.send_text(text)
            await asyncio.sleep(2)
            _ = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        _ws_clients.discard(websocket)
