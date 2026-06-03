import json
import hmac
import hashlib
import base64
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["auth"])

USERS_PATH = Path(__file__).parent.parent.parent.parent / "users.json"
SECRET_KEY = "compare-competencies-secret-key-change-in-production"
TOKEN_TTL = 86400 * 7


class LoginRequest(BaseModel):
    username: str
    password: str


def _load_users() -> dict[str, Any]:
    if not USERS_PATH.exists():
        return {}
    with open(USERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _make_token(username: str, role: str) -> str:
    payload = json.dumps({"u": username, "r": role, "t": int(time.time()) + TOKEN_TTL}, separators=(",", ":"))
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=") + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def _decode_token(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64 = parts[0] + "=" * (4 - len(parts[0]) % 4)
        sig_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = base64.urlsafe_b64decode(payload_b64).decode()
        expected_sig = base64.urlsafe_b64decode(sig_b64)
        actual_sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        data = json.loads(payload)
        if data.get("t", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def get_current_user(request: Request) -> dict[str, Any] | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    return _decode_token(token)


def require_role(role: str):
    async def dependency(request: Request):
        user = get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if user.get("r") != role and user.get("r") != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dependency


@router.post("/api/auth/login")
async def login(body: LoginRequest):
    users = _load_users()
    user = users.get(body.username)
    if user is None or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _make_token(body.username, user["role"])
    return {"token": token, "role": user["role"], "name": user["name"], "username": body.username}


@router.get("/api/auth/me")
async def me(request: Request):
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    users = _load_users()
    u = users.get(user["u"], {})
    return {"username": user["u"], "role": user["r"], "name": u.get("name", "")}



