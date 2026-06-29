"""Auth: login/logout against DB users + sessions (asyncpg)."""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src import config
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["auth"])

SECRET_KEY = config.SECRET_KEY
TOKEN_TTL = 86400 * config.TOKEN_TTL_DAYS


class LoginRequest(BaseModel):
    email: str
    password: str


def _make_token(user_id: str, email: str, role: str) -> str:
    payload = json.dumps(
        {"uid": user_id, "u": email, "r": role, "t": int(time.time()) + TOKEN_TTL},
        separators=(",", ":"),
    )
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        + "."
        + base64.urlsafe_b64encode(sig).decode().rstrip("=")
    )


def _decode_token(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64 = parts[0] + "=" * ((4 - len(parts[0]) % 4) % 4)
        sig_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
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


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def get_current_user(request: Request) -> dict[str, Any] | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    data = _decode_token(token)
    if data is None:
        return None
    try:
        pool = get_pool()
        token_hash = _hash_token(token)
        session = await pool.fetchrow(
            "SELECT logged_out_at FROM sessions WHERE token_hash = $1 AND logged_out_at IS NULL",
            token_hash,
        )
        if session is None:
            return None
    except Exception:
        pass
    return data


def require_role(role: str):
    async def dependency(request: Request):
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if user.get("r") != role and user.get("r") != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dependency


def require_any_role(*roles: str):
    """Allow access if user has any of the given roles (admin always allowed)."""
    async def dependency(request: Request):
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if user.get("r") != "admin" and user.get("r") not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dependency


@router.post("/auth/login")
async def login(body: LoginRequest, request: Request):
    try:
        pool = get_pool()
        row = await pool.fetchrow(
            "SELECT id, email, role, full_name, password_hash = crypt($2, password_hash) AS pw_match FROM users WHERE email = $1 AND is_active = true",
            body.email, body.password,
        )
        if row is None or not row["pw_match"]:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = _make_token(str(row["id"]), row["email"], row["role"])

        token_hash = _hash_token(token)
        ip = request.client.host if request.client else "unknown"
        ua = request.headers.get("User-Agent", "")
        await pool.execute(
            """INSERT INTO sessions (user_id, token_hash, ip_address, user_agent)
               VALUES ($1, $2, $3, $4)""",
            row["id"], token_hash, ip, ua,
        )

        logger.info("user_logged_in", email=body.email, role=row["role"])
        return {
            "token": token,
            "role": row["role"],
            "name": row["full_name"],
            "username": row["email"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("login_db_error", error=str(exc))
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")


@router.post("/auth/logout")
async def logout(request: Request):
    user_data = await get_current_user(request)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    auth = request.headers.get("Authorization", "")
    token = auth[7:]
    token_hash = _hash_token(token)

    pool = get_pool()
    await pool.execute(
        "UPDATE sessions SET logged_out_at = NOW() WHERE token_hash = $1 AND logged_out_at IS NULL",
        token_hash,
    )

    logger.info("user_logged_out", email=user_data.get("u"))
    return {"status": "ok", "message": "Logged out"}


@router.get("/auth/me")
async def me(request: Request):
    user_data = await get_current_user(request)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT email, role, full_name FROM users WHERE email = $1",
        user_data["u"],
    )
    return {
        "username": user_data["u"],
        "role": user_data["r"],
        "name": row["full_name"] if row else "",
    }
