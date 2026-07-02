import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.llm_client import LLMClient

logger = structlog.get_logger("api")
router = APIRouter(tags=["llm"])


class ChatRequest(BaseModel):
    message: str
    temperature: float | None = None
    max_tokens: int | None = None


class ChatResponse(BaseModel):
    response: str
    model: str


@router.post("/llm/chat", response_model=ChatResponse)
async def llm_chat(req: ChatRequest):
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
