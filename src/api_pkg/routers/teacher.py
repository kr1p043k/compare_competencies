from typing import Any
import structlog
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["teacher"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/teacher/stats")
@limiter.limit("30/minute")
async def teacher_stats(request: Request):
    result_dir = config.DATA_RESULT_DIR
    if not result_dir.exists():
        return {"total_reports": 0, "by_profession": []}

    reports = []
    for f in result_dir.glob("*recommendations_*.json"):
        parts = f.stem.replace("recommendations_", "").split("_")
        prof = parts[0] if parts else "unknown"
        reports.append(prof)

    prof_counts: dict[str, int] = {}
    for p in reports:
        prof_counts[p] = prof_counts.get(p, 0) + 1

    by_profession = [{"profession": k, "reports": v} for k, v in sorted(prof_counts.items(), key=lambda x: -x[1])]
    return {"total_reports": len(reports), "by_profession": by_profession}
