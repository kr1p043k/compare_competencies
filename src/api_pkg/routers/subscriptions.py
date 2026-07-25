import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select, delete, func, desc

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["subscriptions"])
limiter = Limiter(key_func=get_remote_address)


# ─── Pydantic models ───────────────────────────────────────────────────────

class SubscriptionCreate(BaseModel):
    topic: str
    source: str = "openalex+arxiv"
    telegram_chat_id: str | None = None
    email: str | None = None


class SubscriptionUpdate(BaseModel):
    topic: str | None = None
    source: str | None = None
    telegram_chat_id: str | None = None
    is_active: bool | None = None


# ─── Helpers ───────────────────────────────────────────────────────────────

def _get_user(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


# ─── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/subscriptions")
@limiter.limit("30/minute")
async def list_subscriptions(request: Request):
    user = _get_user(request)
    from src.database import async_session_factory
    from src.models.krm_models import Subscription

    async with async_session_factory() as session:
        result = await session.execute(
            select(Subscription)
            .where(Subscription.user_id == user)
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
async def create_subscription(request: Request, body: SubscriptionCreate):
    user = _get_user(request)
    from src.database import async_session_factory
    from src.models.krm_models import Subscription
    from src.models.krm_models import _uuid

    sub = Subscription(
        id=_uuid(),
        user_id=user,
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
async def delete_subscription(sub_id: str, request: Request):
    user = _get_user(request)
    from src.database import async_session_factory
    from src.models.krm_models import Subscription

    async with async_session_factory() as session:
        sub = await session.get(Subscription, sub_id)
        if not sub:
            raise HTTPException(status_code=404, detail="Subscription not found")
        if sub.user_id != user:
            raise HTTPException(status_code=403, detail="Forbidden")
        await session.delete(sub)
        await session.commit()
    return {"status": "ok"}


@router.get("/notifications")
@limiter.limit("30/minute")
async def list_notifications(request: Request, limit: int = 50, unread_only: bool = False):
    user = _get_user(request)
    from src.database import async_session_factory
    from src.models.krm_models import Notification

    query = (
        select(Notification)
        .where(Notification.user_id == user)
        .order_by(desc(Notification.created_at))
    )
    if unread_only:
        query = query.where(Notification.is_read == False)

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
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat(),
            }
            for n in notifs
        ],
        "total": len(notifs),
    }


@router.post("/notifications/{notif_id}/read")
@limiter.limit("30/minute")
async def mark_read(notif_id: str, request: Request):
    user = _get_user(request)
    from src.database import async_session_factory
    from src.models.krm_models import Notification

    async with async_session_factory() as session:
        notif = await session.get(Notification, notif_id)
        if not notif:
            raise HTTPException(status_code=404, detail="Notification not found")
        if notif.user_id != user:
            raise HTTPException(status_code=403, detail="Forbidden")
        notif.is_read = True
        await session.commit()
    return {"status": "ok"}


@router.get("/notifications/unread-count")
@limiter.limit("30/minute")
async def unread_count(request: Request):
    user = _get_user(request)
    from src.database import async_session_factory
    from src.models.krm_models import Notification
    from sqlalchemy import func

    async with async_session_factory() as session:
        result = await session.execute(
            select(func.count(Notification.id))
            .where(Notification.user_id == user, Notification.is_read == False)
        )
        count = result.scalar() or 0
    return {"unread_count": count}
