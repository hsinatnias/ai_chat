# api/routes_analytics.py
import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from db import get_async_db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

class FeedbackIn(BaseModel):
    message_id: str
    helpful: Optional[bool] = None
    comment: Optional[str] = None

@router.get("/summary")
async def analytics_summary(db: AsyncSession = Depends(get_async_db)):
    """
    Returns basic counts:
      - conversations
      - messages
      - feedback totals (helpful / unhelpful / unknown)
    """
    try:
        q_convs = await db.execute(text("SELECT COUNT(*) AS c FROM conversations"))
        convs = q_convs.scalar() or 0

        q_msgs = await db.execute(text("SELECT COUNT(*) AS c FROM messages"))
        msgs = q_msgs.scalar() or 0

        q_fb = await db.execute(text("""
            SELECT
              SUM(CASE WHEN helpful IS TRUE THEN 1 ELSE 0 END) AS helpful,
              SUM(CASE WHEN helpful IS FALSE THEN 1 ELSE 0 END) AS unhelpful,
              SUM(CASE WHEN helpful IS NULL THEN 1 ELSE 0 END) AS unknown
            FROM message_feedback
        """))
        fb_row = q_fb.first() or (0,0,0)
        helpful = int(fb_row[0] or 0)
        unhelpful = int(fb_row[1] or 0)
        unknown = int(fb_row[2] or 0)

        return {
            "conversations": int(convs),
            "messages": int(msgs),
            "feedback": {"helpful": helpful, "unhelpful": unhelpful, "unknown": unknown}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top_conversations")
async def top_conversations(limit: int = 10, db: AsyncSession = Depends(get_async_db)):
    """
    Returns top conversations by message count.
    Each item: { conversation_id, message_count, last_message_at, last_assistant_snippet }
    """
    try:
        # 1) conversation id + message_count + last_message_at
        q = await db.execute(text("""
            SELECT
              m.conversation_id,
              COUNT(*) as message_count,
              MAX(m.created_at) as last_message_at
            FROM messages m
            GROUP BY m.conversation_id
            ORDER BY message_count DESC
            LIMIT :limit
        """), {"limit": limit})
        rows = q.fetchall()

        results = []
        for r in rows:
            conv_id = r[0]
            message_count = int(r[1] or 0)
            last_message_at = r[2].isoformat() if r[2] else None

            # fetch last assistant message snippet (best-effort)
            q2 = await db.execute(text("""
                SELECT content, created_at FROM messages
                WHERE conversation_id = :conv AND role = 'assistant'
                ORDER BY created_at DESC
                LIMIT 1
            """), {"conv": conv_id})
            last = q2.first()
            snippet = None
            last_assistant_at = None
            if last:
                content = last[0] or ""
                last_assistant_at = last[1].isoformat() if last[1] else None
                # create short snippet
                snippet = content[:300] + ("..." if len(content) > 300 else "")

            results.append({
                "conversation_id": conv_id,
                "message_count": message_count,
                "last_message_at": last_message_at,
                "last_assistant_snippet": snippet,
                "last_assistant_at": last_assistant_at
            })

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def post_feedback(fb: FeedbackIn, db: AsyncSession = Depends(get_async_db)):
    """
    Record feedback for a message.
    body: { message_id, helpful (true/false/null), comment }
    """
    if not fb.message_id:
        raise HTTPException(status_code=400, detail="message_id required")
    try:
        # insert into message_feedback
        await db.execute(text("""
            INSERT INTO message_feedback (id, message_id, helpful, comment, created_at)
            VALUES (:id, :message_id, :helpful, :comment, :created_at)
        """), {
            "id": __import__("uuid").uuid4().hex,
            "message_id": fb.message_id,
            "helpful": fb.helpful,
            "comment": fb.comment,
            "created_at": datetime.datetime.utcnow()
        })
        await db.commit()
        return {"ok": True}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
