"""n8n webhook router — приём callback'ов от n8n."""

import hmac
import json
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src import Err, Ok
from src.utils import safe_read_json, atomic_write_json

logger = structlog.get_logger("n8n_webhook")

WEBHOOK_STORE = Path(__file__).parent.parent.parent / "data" / "n8n" / "webhooks"

router = APIRouter(tags=["n8n_webhooks"])


class StudentCreatedBody(BaseModel):
    profile_name: str
    skills: list[str]
    target_level: str = "middle"


class PipelineCompletedBody(BaseModel):
    task_id: str
    status: str
    artifacts: dict = {}


class AlertBody(BaseModel):
    type: str
    severity: str = "info"
    message: str
    data: dict = {}


def _verify_n8n_secret(request: Request) -> bool:
    token = request.headers.get("X-N8N-Webhook-Secret", "")
    from src import config
    expected = config.N8N_WEBHOOK_SECRET
    if expected is None:
        logger.warning("n8n_webhook_secret_not_configured")
        return False
    return hmac.compare_digest(token, expected.get_secret_value())


@router.post("/api/n8n/webhook/student-created")
async def webhook_student_created(body: StudentCreatedBody, request: Request):
    if not _verify_n8n_secret(request):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    WEBHOOK_STORE.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "student-created",
        "timestamp": datetime.now().isoformat(),
        "data": body.model_dump(),
    }
    path = WEBHOOK_STORE / f"student_{body.profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_json(record, path)
    logger.info("n8n_webhook_student_created", profile=body.profile_name)
    return {"ok": True, "profile": body.profile_name}


@router.post("/api/n8n/webhook/pipeline-completed")
async def webhook_pipeline_completed(body: PipelineCompletedBody, request: Request):
    if not _verify_n8n_secret(request):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    WEBHOOK_STORE.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "pipeline-completed",
        "timestamp": datetime.now().isoformat(),
        "data": body.model_dump(),
    }
    path = WEBHOOK_STORE / f"pipeline_{body.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_json(record, path)
    logger.info("n8n_webhook_pipeline_completed", task_id=body.task_id, status=body.status)
    return {"ok": True, "task_id": body.task_id}


@router.post("/api/n8n/webhook/alert")
async def webhook_alert(body: AlertBody, request: Request):
    if not _verify_n8n_secret(request):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    WEBHOOK_STORE.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "alert",
        "timestamp": datetime.now().isoformat(),
        "data": body.model_dump(),
    }
    path = WEBHOOK_STORE / f"alert_{body.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_json(record, path)
    logger.info("n8n_webhook_alert", type=body.type, severity=body.severity, message=body.message[:100])
    return {"ok": True}


@router.get("/api/n8n/webhooks")
async def list_webhook_events(request: Request):
    if not _verify_n8n_secret(request):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    WEBHOOK_STORE.mkdir(parents=True, exist_ok=True)
    events = []
    for f in sorted(WEBHOOK_STORE.glob("*.json"), reverse=True)[:50]:
        data = safe_read_json(f)
        if data:
            events.append(data)
    return {"events": events, "total": len(events)}
