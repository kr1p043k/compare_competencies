"""RPD import: upload PDF / collect SFU annotations → KRM → DB seed → analysis.

Endpoints:
    POST /teacher/rpd/upload    multipart (file + dir_code)
    POST /teacher/rpd/collect   dir_code (Yandex Disk only for covered directions)
    GET  /teacher/rpd/sources   directions covered by Yandex Disk collection
    GET  /teacher/rpd/status/{run_id}
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.api_pkg.routers.auth import require_any_role

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["rpd"], dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
limiter = Limiter(key_func=get_remote_address)

_DIR_CODE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$")

# Направления, для которых есть публичные ссылки на аннотации в Yandex Disk
# (совпадает с TARGETS в scripts/sfu_annotations.py).
YANDEX_COVERED = {
    "01.03.01",
    "01.03.02",
    "02.03.02",
    "02.03.02_och",
    "02.03.03",
    "09.03.01_bim",
    "09.03.02",
    "09.03.04",
}

# Код направления в UI/БД -> код для скриптов сбора (ключи TARGETS).
YANDEX_ALIASES = {
    "02.03.02": "02.03.02_och",
}


def _yandex_code(dir_code: str) -> str:
    return YANDEX_ALIASES.get(dir_code, dir_code)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
UPLOAD_RPD_DIR = PROJECT_ROOT / "uploads" / "rpd_pdfs"
REFERENCE_DIR = config.REFERENCE_DIR


class _RpdCancelled(Exception):
    """Pipeline остановлен пользователем через cancel endpoint."""


# run_id -> запущенный процесс (для kill при отмене) и флаг отмены.
_rpd_procs: dict[str, object] = {}
_rpd_cancel: dict[str, threading.Event] = {}


def _cancel_requested(run_id: str | None) -> bool:
    ev = _rpd_cancel.get(run_id or "")
    return ev.is_set() if ev is not None else False


def _validate_dir_code(dir_code: str) -> None:
    if not _DIR_CODE_RE.match(dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")


def _krm_path(dir_code: str) -> Path:
    return REFERENCE_DIR / f"krm_disciplines_{dir_code}.json"


def _cli_env() -> dict:
    return {
        **os.environ,
        # Скрипты лежат в scripts/ и импортируют src.* — без этого
        # `python scripts/*.py` падает с ModuleNotFoundError: No module named 'src'.
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }


def _run_cli_sync(args: list[str], timeout: int, run_id: str | None = None) -> tuple[int, str]:
    """Синхронный запуск (для потоков): работает на любом event loop."""
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
        env=_cli_env(),
    )
    if run_id:
        _rpd_procs[run_id] = proc
    try:
        try:
            out_b, err_b = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out_b, err_b = proc.communicate()
            return -1, "TIMEOUT"
    finally:
        if run_id:
            _rpd_procs.pop(run_id, None)
    out = (out_b or b"").decode("utf-8", errors="ignore")
    err = (err_b or b"").decode("utf-8", errors="ignore")
    return proc.returncode or 0, (out + "\n" + err)[-3000:]


async def _run_cli(args: list[str], timeout: int = 1800, run_id: str | None = None) -> tuple[int, str]:
    """Run a subprocess (python script / module) and capture output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=_cli_env(),
        )
    except NotImplementedError:
        # Windows SelectorEventLoop не умеет async-сабпроцессы —
        # уходим в worker thread с синхронным запуском.
        logger.warning("rpd_cli_thread_fallback", reason="selector_loop")
        code, out = await asyncio.to_thread(_run_cli_sync, args, timeout, run_id)
        if _cancel_requested(run_id):
            raise _RpdCancelled()
        return code, out
    if run_id:
        _rpd_procs[run_id] = proc
    try:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            if proc.returncode is None:
                proc.kill()
            return -1, "TIMEOUT"
    finally:
        if run_id:
            _rpd_procs.pop(run_id, None)
    if _cancel_requested(run_id):
        raise _RpdCancelled()
    out = (stdout or b"").decode("utf-8", errors="ignore")
    err = (stderr or b"").decode("utf-8", errors="ignore")
    return proc.returncode or 0, (out + "\n" + err)[-3000:]


async def _update_run(run_id: str, stats: dict) -> None:
    """Patch pipeline_runs.stats with current stage (no status change)."""
    from src.pipeline.db_writer import _pool
    pool = await _pool()
    await pool.execute(
        "UPDATE pipeline_runs SET stats=$1 WHERE id=$2",
        json.dumps(stats, ensure_ascii=False, default=str),
        run_id,
    )


async def _seed_direction(dir_code: str, run_id: str | None = None) -> tuple[int, str]:
    return await _run_cli([
        sys.executable, "scripts/seed_all_directions.py", "--only", dir_code,
    ], timeout=1800, run_id=run_id)


async def _run_teacher_analysis(dir_code: str, run_id: str | None = None) -> tuple[int, str]:
    return await _run_cli([
        sys.executable, "-m", "src.cli", "teacher-analysis",
        "--direction", dir_code,
    ], timeout=2400, run_id=run_id)


async def _finish_cancelled(run_id: str) -> None:
    from src.pipeline.db_writer import complete_pipeline_run

    try:
        await complete_pipeline_run(run_id, status="cancelled", error="Отменено пользователем",
                                    stats={"stage": "cancelled", "status": "cancelled"})
    except Exception:
        pass
    finally:
        _rpd_cancel.pop(run_id, None)
        _rpd_procs.pop(run_id, None)


# ---------- Upload pipeline ----------


def _parse_pdfs_to_krm(dir_code: str, direction_name: str | None, profile: str | None) -> dict:
    """Parse all PDFs in the per-direction folder and merge into KRM structure."""
    from src.loaders.rpd_loader import RPDLoader

    pdf_dir = UPLOAD_RPD_DIR / dir_code
    loader = RPDLoader(str(pdf_dir))
    parsed = loader.load_all(dir_code=dir_code, direction_name=direction_name, profile=profile)

    # Merge with existing KRM file (keep disciplines already present)
    krm_path = _krm_path(dir_code)
    merged = parsed
    if krm_path.exists():
        try:
            existing = json.loads(krm_path.read_text(encoding="utf-8"))
            sub = next(iter(existing.values()), {}) if isinstance(existing, dict) else {}
            existing_discs = sub.get("disciplines", {})
            if isinstance(existing_discs, dict):
                merged_discs = merged[dir_code]["disciplines"]
                existing_discs.update(merged_discs)
                merged[dir_code]["disciplines"] = existing_discs
                if not merged[dir_code].get("direction_name") or merged[dir_code]["direction_name"] == dir_code:
                    merged[dir_code]["direction_name"] = sub.get("direction_name", dir_code)
                if not merged[dir_code].get("profile"):
                    merged[dir_code]["profile"] = sub.get("profile", "")
        except Exception:
            logger.warning("rpd_merge_existing_failed", dir_code=dir_code)

    krm_path.parent.mkdir(parents=True, exist_ok=True)
    krm_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


async def _upload_pipeline(run_id: str, dir_code: str, fname: str, direction_name: str | None, profile: str | None) -> None:
    from src.pipeline.db_writer import complete_pipeline_run

    _rpd_cancel[run_id] = threading.Event()
    try:
        await _update_run(run_id, {"stage": "parse", "status": "running"})
        merged = await asyncio.to_thread(_parse_pdfs_to_krm, dir_code, direction_name, profile)
        if _cancel_requested(run_id):
            raise _RpdCancelled()
        stats = {
            "stage": "seed",
            "status": "running",
            "disciplines": len(merged.get(dir_code, {}).get("disciplines", {})),
        }
        await _update_run(run_id, stats)

        code, out = await _seed_direction(dir_code, run_id)
        if code != 0:
            raise RuntimeError(f"seed failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "analysis", "status": "running"})
        code, out = await _run_teacher_analysis(dir_code, run_id)
        if code != 0:
            raise RuntimeError(f"teacher-analysis failed: {out[-500:]}")

        await complete_pipeline_run(
            run_id, status="completed",
            stats={"stage": "done", "status": "completed", "file": fname, "dir_code": dir_code},
        )
    except _RpdCancelled:
        logger.info("rpd_upload_pipeline_cancelled", run_id=run_id, dir_code=dir_code)
        await _finish_cancelled(run_id)
        return
    except Exception as exc:
        logger.error("rpd_upload_pipeline_failed", run_id=run_id, dir_code=dir_code,
                     exc_type=type(exc).__name__, exc_repr=repr(exc))
        try:
            await complete_pipeline_run(run_id, status="failed", error=str(exc),
                                        stats={"stage": "error", "status": "failed"})
        except Exception:
            pass
    finally:
        _rpd_cancel.pop(run_id, None)
        _rpd_procs.pop(run_id, None)


@router.post("/teacher/rpd/upload")
@limiter.limit("5/minute")
async def rpd_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    dir_code: Annotated[str, Form()],
    direction_name: Annotated[str, Form()] = "",
    profile: Annotated[str, Form()] = "",
):
    """Загрузка PDF РПД дисциплины."""
    _validate_dir_code(dir_code)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    safe_name = Path(file.filename).name
    target_dir = UPLOAD_RPD_DIR / dir_code
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    target_path.write_bytes(content)
    logger.info("rpd_pdf_saved", dir_code=dir_code, file=safe_name, bytes=len(content))

    from src.pipeline.db_writer import create_pipeline_run
    run_id = await create_pipeline_run("rpd-import")
    await _update_run(run_id, {"stage": "saved", "status": "running", "file": safe_name, "dir_code": dir_code})
    background_tasks.add_task(
        _upload_pipeline, run_id, dir_code, safe_name, direction_name or None, profile or None
    )
    return {"status": "started", "run_id": run_id, "file": safe_name, "dir_code": dir_code}


# ---------- Collect (Yandex Disk) pipeline ----------


async def _collect_pipeline(run_id: str, dir_code: str) -> None:
    from src.pipeline.db_writer import complete_pipeline_run

    _rpd_cancel[run_id] = threading.Event()
    ycode = _yandex_code(dir_code)
    try:
        await _update_run(run_id, {"stage": "collect", "status": "running", "dir_code": dir_code})
        code, out = await _run_cli([
            sys.executable, "scripts/sfu_annotations.py", "collect", ycode,
        ], timeout=1800, run_id=run_id)
        if code != 0:
            raise RuntimeError(f"sfu collect failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "merge", "status": "running"})
        code, out = await _run_cli([
            sys.executable, "scripts/merge_annotations_to_krm.py", "--only", ycode,
        ], timeout=1200, run_id=run_id)
        if code != 0:
            raise RuntimeError(f"merge failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "seed", "status": "running"})
        code, out = await _seed_direction(ycode, run_id)
        if code != 0:
            raise RuntimeError(f"seed failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "analysis", "status": "running"})
        code, out = await _run_teacher_analysis(dir_code, run_id)
        if code != 0:
            raise RuntimeError(f"teacher-analysis failed: {out[-500:]}")

        await complete_pipeline_run(
            run_id, status="completed",
            stats={"stage": "done", "status": "completed", "dir_code": dir_code},
        )
    except _RpdCancelled:
        logger.info("rpd_collect_pipeline_cancelled", run_id=run_id, dir_code=dir_code)
        await _finish_cancelled(run_id)
        return
    except Exception as exc:
        logger.error("rpd_collect_pipeline_failed", run_id=run_id, dir_code=dir_code,
                     exc_type=type(exc).__name__, exc_repr=repr(exc))
        try:
            await complete_pipeline_run(run_id, status="failed", error=str(exc),
                                        stats={"stage": "error", "status": "failed"})
        except Exception:
            pass
    finally:
        _rpd_cancel.pop(run_id, None)
        _rpd_procs.pop(run_id, None)


@router.post("/teacher/rpd/collect")
@limiter.limit("2/minute")
async def rpd_collect(request: Request, background_tasks: BackgroundTasks, dir_code: Annotated[str, Form()] = "09.03.02"):
    """Сбор компетенций из загруженных РПД."""
    _validate_dir_code(dir_code)
    if dir_code not in YANDEX_COVERED:
        raise HTTPException(status_code=400, detail=f"Yandex Disk collection is not available for {dir_code}")

    from src.pipeline.db_writer import create_pipeline_run
    run_id = await create_pipeline_run("rpd-import")
    await _update_run(run_id, {"stage": "collect", "status": "running", "dir_code": dir_code, "source": "yandex"})
    background_tasks.add_task(_collect_pipeline, run_id, dir_code)
    return {"status": "started", "run_id": run_id, "dir_code": dir_code, "source": "yandex"}


@router.post("/teacher/rpd/cancel/{run_id}")
async def rpd_cancel(request: Request, run_id: str):
    """Остановить сбор/загрузку РПД: флаг отмены + kill процесса."""
    from src.pipeline.db_writer import complete_pipeline_run

    ev = _rpd_cancel.get(run_id)
    proc = _rpd_procs.get(run_id)
    if ev is None and proc is None:
        # Задача уже завершилась или неизвестна — смотрим БД.
        from sqlalchemy import select
        from src.database import async_session_factory
        from src.models.krm_models import PipelineRun

        async with async_session_factory() as session:
            run = await session.get(PipelineRun, run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            if run.status in ("completed", "failed", "cancelled"):
                return {"status": run.status, "message": "Задача уже завершена"}
            await complete_pipeline_run(run_id, status="cancelled", error="Отменено пользователем",
                                        stats={"stage": "cancelled", "status": "cancelled"})
            return {"status": "cancelled", "message": "Задача остановлена"}
    if ev is not None:
        ev.set()
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass
    logger.info("rpd_cancel_requested", run_id=run_id)
    return {"status": "cancelling", "message": "Останавливаю задачу..."}


# ---------- Metadata ----------


@router.get("/teacher/rpd/sources")
async def rpd_sources(request: Request):
    """Источники РПД."""
    return {"yandex_covered": sorted(YANDEX_COVERED), "all_directions": sorted(YANDEX_COVERED | {
        p.name[len("krm_disciplines_"):-len(".json")]
        for p in REFERENCE_DIR.glob("krm_disciplines_*.json")
        if "_clean" not in p.name and _DIR_CODE_RE.match(p.name[len("krm_disciplines_"):-len(".json")])
    })}


@router.get("/teacher/rpd/status/{run_id}")
async def rpd_status(run_id: str, request: Request):
    """Статус задачи сбора РПД."""
    from sqlalchemy import select
    from src.database import async_session_factory
    from src.models.krm_models import PipelineRun

    async with async_session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "id": str(run.id),
            "action": run.action,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error": run.error_message,
            "stats": run.stats or {},
        }
