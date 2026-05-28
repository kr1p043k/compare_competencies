"""Shared pipeline progress — writes to pipeline_progress.json for frontend polling."""

import json
import time
from pathlib import Path

import structlog

PROGRESS_FILE = Path(__file__).parent.parent.parent / "data" / "cache" / "pipeline_progress.json"
MAX_LOGS = 200
logger = structlog.get_logger("pipeline_progress")


def write(pct: int, message: str, logs: list[str] | None = None):
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {"pct": pct, "message": message, "timestamp": time.time()}
        existing_logs: list[str] = []
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                    existing_logs = prev.get("logs", [])
            except Exception as e:
                logger.warning("progress_file_read_failed", error=str(e))
                existing_logs = []
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        if logs is not None:
            existing_logs = logs
        else:
            if not existing_logs or existing_logs[-1] != entry:
                existing_logs.append(entry)
        data["logs"] = existing_logs[-MAX_LOGS:]
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        logger.warning("progress_write_failed", error=True)
