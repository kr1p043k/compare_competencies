"""RPD import: upload PDF / collect SFU annotations → KRM → DB seed → analysis.

Endpoints:
    POST /teacher/rpd/upload    multipart (file + dir_code)
    POST /teacher/rpd/collect   dir_code (Yandex Disk only for covered directions)
    GET  /teacher/rpd/sources   directions covered by Yandex Disk collection
    GET  /teacher/rpd/status/{run_id}
"""
import asyncio
import json
import re
import sys
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
    "02.03.02_och",
    "02.03.03",
    "09.03.01_bim",
    "09.03.02",
    "09.03.04",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
UPLOAD_RPD_DIR = PROJECT_ROOT / "uploads" / "rpd_pdfs"
REFERENCE_DIR = config.REFERENCE_DIR


def _validate_dir_code(dir_code: str) -> None:
    if not _DIR_CODE_RE.match(dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")


def _krm_path(dir_code: str) -> Path:
    return REFERENCE_DIR / f"krm_disciplines_{dir_code}.json"


async def _run_cli(args: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Run a subprocess (python script / module) and capture output."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        if proc.returncode is None:
            proc.kill()
        return -1, "TIMEOUT"
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


async def _run_teacher_analysis(dir_code: str) -> tuple[int, str]:
    return await _run_cli([
        sys.executable, "-m", "src.cli", "teacher-analysis",
        "--direction", dir_code,
    ], timeout=2400)


async def _seed_direction(dir_code: str) -> tuple[int, str]:
    return await _run_cli([
        sys.executable, "scripts/seed_all_directions.py", "--only", dir_code,
    ], timeout=1800)


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

    try:
        await _update_run(run_id, {"stage": "parse", "status": "running"})
        merged = await asyncio.to_thread(_parse_pdfs_to_krm, dir_code, direction_name, profile)
        stats = {
            "stage": "seed",
            "status": "running",
            "disciplines": len(merged.get(dir_code, {}).get("disciplines", {})),
        }
        await _update_run(run_id, stats)

        code, out = await _seed_direction(dir_code)
        if code != 0:
            raise RuntimeError(f"seed failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "analysis", "status": "running"})
        code, out = await _run_teacher_analysis(dir_code)
        if code != 0:
            raise RuntimeError(f"teacher-analysis failed: {out[-500:]}")

        await complete_pipeline_run(
            run_id, status="completed",
            stats={"stage": "done", "status": "completed", "file": fname, "dir_code": dir_code},
        )
    except Exception as exc:
        logger.error("rpd_upload_pipeline_failed", run_id=run_id, dir_code=dir_code, error=str(exc))
        try:
            await complete_pipeline_run(run_id, status="failed", error=str(exc),
                                        stats={"stage": "error", "status": "failed"})
        except Exception:
            pass


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

    try:
        await _update_run(run_id, {"stage": "collect", "status": "running", "dir_code": dir_code})
        code, out = await _run_cli([
            sys.executable, "scripts/sfu_annotations.py", "collect", dir_code,
        ], timeout=1800)
        if code != 0:
            raise RuntimeError(f"sfu collect failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "merge", "status": "running"})
        code, out = await _run_cli([
            sys.executable, "scripts/merge_annotations_to_krm.py", "--only", dir_code,
        ], timeout=1200)
        if code != 0:
            raise RuntimeError(f"merge failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "seed", "status": "running"})
        code, out = await _seed_direction(dir_code)
        if code != 0:
            raise RuntimeError(f"seed failed: {out[-500:]}")

        await _update_run(run_id, {"stage": "analysis", "status": "running"})
        code, out = await _run_teacher_analysis(dir_code)
        if code != 0:
            raise RuntimeError(f"teacher-analysis failed: {out[-500:]}")

        await complete_pipeline_run(
            run_id, status="completed",
            stats={"stage": "done", "status": "completed", "dir_code": dir_code},
        )
    except Exception as exc:
        logger.error("rpd_collect_pipeline_failed", run_id=run_id, dir_code=dir_code, error=str(exc))
        try:
            await complete_pipeline_run(run_id, status="failed", error=str(exc),
                                        stats={"stage": "error", "status": "failed"})
        except Exception:
            pass


@router.post("/teacher/rpd/collect")
@limiter.limit("2/minute")
async def rpd_collect(request: Request, background_tasks: BackgroundTasks, dir_code: Annotated[str, Form()] = "09.03.02"):
    _validate_dir_code(dir_code)
    if dir_code not in YANDEX_COVERED:
        raise HTTPException(status_code=400, detail=f"Yandex Disk collection is not available for {dir_code}")

    from src.pipeline.db_writer import create_pipeline_run
    run_id = await create_pipeline_run("rpd-import")
    await _update_run(run_id, {"stage": "collect", "status": "running", "dir_code": dir_code, "source": "yandex"})
    background_tasks.add_task(_collect_pipeline, run_id, dir_code)
    return {"status": "started", "run_id": run_id, "dir_code": dir_code, "source": "yandex"}


# ---------- Metadata ----------


@router.get("/teacher/rpd/sources")
async def rpd_sources(request: Request):
    return {"yandex_covered": sorted(YANDEX_COVERED), "all_directions": sorted(YANDEX_COVERED | {
        p.name[len("krm_disciplines_"):-len(".json")]
        for p in REFERENCE_DIR.glob("krm_disciplines_*.json")
        if "_clean" not in p.name and _DIR_CODE_RE.match(p.name[len("krm_disciplines_"):-len(".json")])
    })}


@router.get("/teacher/rpd/status/{run_id}")
async def rpd_status(run_id: str, request: Request):
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
