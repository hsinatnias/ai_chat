# scripts/create_user.py
import asyncio
import os
import sqlalchemy
from sqlalchemy import text
from db import engine  # your db/__init__.py exports engine
import uuid

def mk_uuid():
    return str(uuid.uuid4())

async def create_user(email: str, name: str = None, password_hash: str = None):
    async with engine.begin() as conn:
        uid = mk_uuid()
        await conn.execute(
            text("INSERT INTO users (id, email, name, password_hash, is_active, created_at) VALUES (:id, :email, :name, :ph, :active, datetime('now'))"),
            {"id": uid, "email": email, "name": name, "ph": password_hash, "active": True}
        )
    print("Created user id:", uid)
    return uid

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_user.py <email> [name]")
        sys.exit(1)
    email = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(create_user(email, name))
