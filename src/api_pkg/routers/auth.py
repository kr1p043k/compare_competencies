"""Auth: login/logout against DB users + sessions."""

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
from sqlalchemy import select, text as sa_text

from src import config
from src.database import async_session_factory
from src.models.krm_models import Session as SessionModel
from src.models.krm_models import User

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


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


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
async def login(body: LoginRequest, request: Request):
    async with async_session_factory() as session:
        result = await session.execute(
            select(User).where(User.email == body.email, User.is_active == True)
        )
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Verify password via pgcrypto
        result = await session.execute(
            sa_text("SELECT :input_hash = :stored_hash AS matched"),
            {
                "input_hash": sa_text(f"crypt({body.password!r}, {user.password_hash!r})"),
                "stored_hash": user.password_hash,
            },
        )
        # Simpler approach: run crypt() in SQL
        verify = await session.execute(
            sa_text(f"SELECT crypt(:pw, :hash) = :hash AS matched"),
            {"pw": body.password, "hash": user.password_hash},
        )
        matched = verify.scalar_one()
        if not matched:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = _make_token(str(user.id), user.email, user.role)

        # Create session
        token_hash = _hash_token(token)
        ip = request.client.host if request.client else "unknown"
        ua = request.headers.get("User-Agent", "")
        sess = SessionModel(
            user_id=user.id,
            token_hash=token_hash,
            ip_address=ip,
            user_agent=ua,
        )
        session.add(sess)
        await session.commit()

    logger.info("user_logged_in", email=body.email, role=user.role)
    return {
        "token": token,
        "role": user.role,
        "name": user.full_name,
        "username": user.email,
    }


@router.post("/api/auth/logout")
async def logout(request: Request):
    user_data = get_current_user(request)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    auth = request.headers.get("Authorization", "")
    token = auth[7:]
    token_hash = _hash_token(token)

    async with async_session_factory() as session:
        from sqlalchemy import update as sa_update

        await session.execute(
            sa_update(SessionModel)
            .where(SessionModel.token_hash == token_hash, SessionModel.logged_out_at.is_(None))
            .values(logged_out_at=sa_text("NOW()"))
        )
        await session.commit()

    logger.info("user_logged_out", email=user_data.get("u"))
    return {"status": "ok", "message": "Logged out"}


@router.get("/api/auth/me")
async def me(request: Request):
    user_data = get_current_user(request)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    async with async_session_factory() as session:
        result = await session.execute(
            select(User).where(User.email == user_data["u"])
        )
        user = result.scalar_one_or_none()

    return {
        "username": user_data["u"],
        "role": user_data["r"],
        "name": user.full_name if user else "",
    }
