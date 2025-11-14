# auth/sessions.py
import os
import uuid
import time
from typing import Optional, Dict
from fastapi import Cookie, HTTPException, Depends, Response
from redis.asyncio import Redis

# Redis URL from .env
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
redis = Redis.from_url(REDIS_URL, decode_responses=True)

SESSION_KEY_PREFIX = os.getenv("SESSION_KEY_PREFIX", "session:")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "session_id")
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24)))  # 24h default

async def create_session(user_id: Optional[str] = "") -> str:
    sid = str(uuid.uuid4())
    key = SESSION_KEY_PREFIX + sid
    await redis.hset(key, mapping={"user_id": user_id, "created": str(int(time.time()))})
    await redis.expire(key, SESSION_TTL)
    return sid

async def get_session(session_id: Optional[str] = Cookie(None)) -> Optional[Dict[str, str]]:
    """
    Returns session mapping or None when absent.
    Note: this is intentionally non-raising so endpoints can choose behavior.
    """
    if not session_id:
        return None
    res = await redis.hgetall(SESSION_KEY_PREFIX + session_id)
    if not res:
        return None
    # refresh TTL
    try:
        await redis.expire(SESSION_KEY_PREFIX + session_id, SESSION_TTL)
    except Exception:
        pass
    # include session_id in returned dict for convenience
    res["session_id"] = session_id
    return res

async def destroy_session(session_id: Optional[str] = None):
    if not session_id:
        return
    try:
        await redis.delete(SESSION_KEY_PREFIX + session_id)
    except Exception:
        pass

async def touch_session(session_id: Optional[str] = None):
    if not session_id:
        return
    try:
        await redis.expire(SESSION_KEY_PREFIX + session_id, SESSION_TTL)
    except Exception:
        pass

async def current_session(session_id: Optional[str] = Cookie(None)):
    """
    Dependency for protected endpoints. Raises 401 when missing/expired.
    Returns a dict like {"session_id": sid, "user_id": "...", "created": "..."}
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="not authenticated")
    s = await get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="session expired")
    # touch ttl
    await touch_session(session_id)
    return {"session_id": session_id, "user_id": s.get("user_id", "")}


def set_session_cookie(response: Response, sid: str):
    """
    Set the session cookie on the given Response object.

    Local dev defaults:
      - secure=False (unless SESSION_COOKIE_SECURE=1 or in production)
      - samesite='Lax' (works for simple POST->redirect flows)
      - httponly=True
      - path="/"
    Control with env vars:
      SESSION_COOKIE_SECURE (0/1 or true/false)
      SESSION_COOKIE_SAMESITE (Lax/Strict/None)
      SESSION_COOKIE_NAME (defaults to session_id)
    """
    # determine secure flag
    env_val = os.getenv("SESSION_COOKIE_SECURE", "").strip().lower()
    if env_val in ("1", "true", "yes"):
        secure_flag = True
    elif env_val in ("0", "false", "no"):
        secure_flag = False
    else:
        # auto-detect: default to False for local dev
        # If you set ENV=production or SESSION_COOKIE_SECURE=1 it'll switch to secure
        if os.getenv("ENV", "").lower().startswith("prod") or os.getenv("ENV", "").lower().startswith("production"):
            secure_flag = True
        else:
            secure_flag = False

    samesite = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")  # Lax is safe default for most flows
    cookie_name = os.getenv("SESSION_COOKIE_NAME", SESSION_COOKIE_NAME)
    max_age = int(os.getenv("SESSION_TTL_SECONDS", str(SESSION_TTL)))

    response.set_cookie(
        key=cookie_name,
        value=sid,
        httponly=True,
        secure=secure_flag,
        samesite=samesite,
        max_age=max_age,
        path="/"
    )
