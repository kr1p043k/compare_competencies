import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from src.api_pkg.rate_limit import limiter
from src.api_pkg.routers.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["subscriptions"])


# ─── Pydantic models ───────────────────────────────────────────────────────

class SubscriptionCreate(BaseModel):
    # topic уходит во внешние API (OpenAlex/arXiv) и в БД, поэтому ограничиваем
    # длину и не даём сохранить пустую строку из одних пробелов.
    topic: str = Field(min_length=1, max_length=200)
    source: str = Field(default="openalex+arxiv", max_length=100)
    telegram_chat_id: str | None = Field(default=None, max_length=64)
    email: str | None = Field(default=None, max_length=254)

    @field_validator("topic")
    @classmethod
    def _topic_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("topic must not be blank")
        return v


class SubscriptionUpdate(BaseModel):
    topic: str | None = None
    source: str | None = None
    telegram_chat_id: str | None = None
    is_active: bool | None = None


# ─── Auth dependency ───────────────────────────────────────────────────────

async def require_auth(request: Request) -> dict:
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


# ─── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/subscriptions")
@limiter.limit("30/minute")
async def list_subscriptions(request: Request, user: dict = Depends(require_auth)):
    """Подписки пользователя."""
    uid = user["uid"]
    from sqlalchemy import select

    from src.database import async_session_factory
    from src.models.krm_models import Subscription

    async with async_session_factory() as session:
        result = await session.execute(
            select(Subscription)
            .where(Subscription.user_id == uid)
            .order_by(Subscription.created_at.desc())
        )
        subs = result.scalars().all()
    return {
        "subscriptions": [
            {
                "id": s.id,
                "topic": s.topic,
                "source": s.source,
                "telegram_chat_id": s.telegram_chat_id,
                "email": s.email,
                "is_active": s.is_active,
                "last_checked_at": s.last_checked_at.isoformat() if s.last_checked_at else None,
                "created_at": s.created_at.isoformat(),
            }
            for s in subs
        ],
        "total": len(subs),
    }


@router.post("/subscriptions")
@limiter.limit("10/minute")
async def create_subscription(request: Request, body: SubscriptionCreate, user: dict = Depends(require_auth)):
    """Создать подписку."""
    uid = user["uid"]
    from src.database import async_session_factory
    from src.models.krm_models import Subscription, _uuid

    sub = Subscription(
        id=_uuid(),
        user_id=uid,
        topic=body.topic.strip(),
        source=body.source,
        telegram_chat_id=body.telegram_chat_id,
        email=body.email,
    )
    async with async_session_factory() as session:
        session.add(sub)
        await session.commit()
        await session.refresh(sub)
    return {"status": "ok", "id": sub.id}


@router.delete("/subscriptions/{sub_id}")
@limiter.limit("10/minute")
async def delete_subscription(sub_id: str, request: Request, user: dict = Depends(require_auth)):
    """Удалить подписку."""
    uid = user["uid"]
    from src.database import async_session_factory
    from src.models.krm_models import Subscription

    async with async_session_factory() as session:
        sub = await session.get(Subscription, sub_id)
        if not sub:
            raise HTTPException(status_code=404, detail="Subscription not found")
        if sub.user_id != uid and user.get("r") != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        await session.delete(sub)
        await session.commit()
    return {"status": "ok"}


@router.get("/notifications")
@limiter.limit("30/minute")
async def list_notifications(
    request: Request,
    # Границы обязательны: без них limit уходит прямо в .limit(), и клиент
    # получает либо ошибку драйвера на отрицательном, либо мега-выборку.
    limit: int = Query(50, ge=1, le=200),
    unread_only: bool = False,
    user: dict = Depends(require_auth),
):
    """Уведомления пользователя."""
    uid = user["uid"]
    from sqlalchemy import desc, select

    from src.database import async_session_factory
    from src.models.krm_models import Notification

    query = (
        select(Notification)
        .where(Notification.user_id == uid)
        .order_by(desc(Notification.created_at))
    )
    if unread_only:
        query = query.where(not Notification.is_read)

    async with async_session_factory() as session:
        result = await session.execute(query.limit(limit))
        notifs = result.scalars().all()
    return {
        "notifications": [
            {
                "id": n.id,
                "subscription_id": n.subscription_id,
                "title": n.title,
                "body": n.body,
                "article_url": n.article_url,
                "article_source": n.article_source,
                "severity": n.severity or "info",
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat(),
            }
            for n in notifs
        ],
        "total": len(notifs),
    }


@router.post("/notifications/{notif_id}/read")
@limiter.limit("30/minute")
async def mark_read(notif_id: str, request: Request, user: dict = Depends(require_auth)):
    """Пометить уведомление прочитанным."""
    uid = user["uid"]
    from src.database import async_session_factory
    from src.models.krm_models import Notification

    async with async_session_factory() as session:
        notif = await session.get(Notification, notif_id)
        if not notif:
            raise HTTPException(status_code=404, detail="Notification not found")
        if notif.user_id != uid and user.get("r") != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        notif.is_read = True
        await session.commit()
    return {"status": "ok"}


@router.get("/notifications/unread-count")
@limiter.limit("30/minute")
async def unread_count(request: Request, user: dict = Depends(require_auth)):
    """Число непрочитанных уведомлений."""
    uid = user["uid"]
    from sqlalchemy import func, select

    from src.database import async_session_factory
    from src.models.krm_models import Notification

    async with async_session_factory() as session:
        result = await session.execute(
            select(func.count(Notification.id))
            .where(Notification.user_id == uid, not Notification.is_read)
        )
        count = result.scalar() or 0
    return {"unread_count": count}
