from typing import Any
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api_pkg.student_actions import get_actions, log_action

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["student"])
limiter = Limiter(key_func=get_remote_address)


class LogActionRequest(BaseModel):
    action_type: str
    profession: str = ""
    region: str = ""
    vacancies_found: int = 0
    result_ref: str = ""


@router.get("/api/student/history")
@limiter.limit("30/minute")
async def student_history(request: Request, limit: int = 50):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    entries = get_actions(username=user, limit=limit)
    return {"history": entries, "total": len(entries)}


@router.post("/api/student/log-action")
@limiter.limit("20/minute")
async def student_log_action(request: Request, body: LogActionRequest):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    log_action(
        username=user,
        action_type=body.action_type,
        profession=body.profession,
        region=body.region,
        vacancies_found=body.vacancies_found,
        result_ref=body.result_ref,
    )
    return {"status": "ok"}
