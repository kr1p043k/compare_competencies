"""Очередь предложений навыков от преподавателей (модерация админа).

Файловое хранилище data/skill_suggestions.json: [{'id','skill','category_hint',
'status','created_by','created_at','decided_at'}]. Без миграций — по скорости
как manual_foundational_skills.json. Нормализация: lower+trim.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from src import config

STATUSES = ("pending", "approved", "rejected")


def _path() -> Path:
    return Path(config.DATA_DIR) / "skill_suggestions.json"


def _norm(skill: str) -> str:
    return " ".join((skill or "").lower().split())


def load_all() -> list[dict]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(items: list[dict]) -> None:
    _path().write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def add(skill: str, category_hint: str, created_by: str) -> dict:
    skill_n = _norm(skill)
    if len(skill_n) < 2 or len(skill_n) > 80:
        raise ValueError("Название навыка: от 2 до 80 символов")
    items = load_all()
    for it in items:
        if it.get("skill") == skill_n and it.get("status") == "pending":
            raise ValueError("Такой навык уже на модерации")
    entry = {
        "id": uuid.uuid4().hex[:12],
        "skill": skill_n,
        "category_hint": (category_hint or "").strip()[:64],
        "status": "pending",
        "created_by": created_by or "",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decided_at": None,
    }
    items.append(entry)
    _save(items)
    return entry


def decide(suggestion_id: str, approve: bool, decided_by: str | None = None) -> dict | None:
    items = load_all()
    for it in items:
        if it.get("id") == suggestion_id and it.get("status") == "pending":
            it["status"] = "approved" if approve else "rejected"
            it["decided_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if decided_by:
                it["decided_by"] = decided_by
            _save(items)
            return it
    return None
