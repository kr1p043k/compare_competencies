import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api_pkg.rate_limit import limiter
from src.api_pkg.routers.auth import require_any_role, user_error_detail
from src.services.llm_client import LLMClient

logger = structlog.get_logger("api")
router = APIRouter(tags=["llm"])


class ChatRequest(BaseModel):
    message: str = Field(max_length=4000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=2000)


class ChatResponse(BaseModel):
    response: str
    model: str


@router.post(
    "/llm/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_any_role("admin", "teacher", "rop"))],
)
@limiter.limit("10/minute")
async def llm_chat(request: Request, req: ChatRequest):
    """Чат с LLM (лимит 10/мин, до 4000 символов)."""
    client = LLMClient()
    try:
        # LLMClient синхронный (OpenAI SDK), а ручка async: без to_thread
        # один медленный запрос к модели замораживал весь event loop API.
        response = await asyncio.to_thread(
            client.generate,
            prompt=req.message,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return ChatResponse(response=response, model=client.model)
    except Exception as e:
        logger.error("llm_chat_failed", error=str(e))
        raise HTTPException(
            status_code=502,
            detail=await user_error_detail(request, str(e), "Запрос к языковой модели не удался. Попробуйте позже."),
        ) from None
