# helpers.py
import os
import uuid
from passlib.context import CryptContext
from fastapi import Header, HTTPException

# import the one true get_async_db from your db package
from db import get_async_db    # <--- use this, don't re-define it

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)

def gen_user_id() -> str:
    return str(uuid.uuid4())

# admin api key comes from env (set ADMIN_API_KEY in .env or system env)
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "dev-key-change-me")

async def require_admin_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid X-API-Key")
    return True
