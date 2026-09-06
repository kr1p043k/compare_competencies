import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.services.llm_client import LLMClient

logger = structlog.get_logger("api")
router = APIRouter(tags=["llm"])
limiter = Limiter(key_func=get_remote_address)


class ChatRequest(BaseModel):
    message: str = Field(max_length=4000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=2000)


class ChatResponse(BaseModel):
    response: str
    model: str


@router.post("/llm/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def llm_chat(request: Request, req: ChatRequest):
    """Чат с LLM (лимит 10/мин, до 4000 символов)."""
    client = LLMClient()
    try:
        response = client.generate(
            prompt=req.message,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return ChatResponse(response=response, model=client.model)
    except Exception as e:
        logger.error("llm_chat_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
