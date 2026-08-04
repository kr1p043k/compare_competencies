import structlog
import httpx

from src import config

logger = structlog.get_logger(__name__)


async def send_telegram(chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        logger.warning("telegram_token_not_set")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": False,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            logger.info("telegram_sent", chat_id=chat_id, text_len=len(text))
            return True
        else:
            logger.error("telegram_failed", status=resp.status_code, body=resp.text[:200])
            return False
    except Exception as e:
        logger.error("telegram_error", error=str(e))
        return False
