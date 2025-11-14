# database.py
import os
import json
import asyncio
from pathlib import Path
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy.exc import SQLAlchemyError

# import async engine and Base model
from db import engine as async_engine, DATABASE_URL
from db.models import Base

# modules storage file (simple JSON)
MODULES_FILE = os.getenv("MODULES_FILE", "data/modules.json")
os.makedirs(os.path.dirname(MODULES_FILE), exist_ok=True)

def _get_sync_url():
    """
    Convert async DB URL to sync URL for metadata.create_all with a sync engine.
    e.g. sqlite+aiosqlite:///data/app.db -> sqlite:///data/app.db
    """
    url = DATABASE_URL
    if url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite+aiosqlite://", "sqlite://", 1)
    # Postgres example: postgresql+asyncpg:// -> postgresql+psycopg2:// (if you ever use PG sync)
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg2")
    return url

def create_table():
    """
    Synchronous helper to create SQL tables (uses a sync engine under the hood so it can be called
    from a thread via asyncio.to_thread(create_table)).
    """
    sync_url = _get_sync_url()
    # ensure parent path exists for sqlite file
    if sync_url.startswith("sqlite:///"):
        db_path = sync_url.split("sqlite:///")[-1]
        if db_path:
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    # create sync engine and create tables
    try:
        sync_engine = create_sync_engine(sync_url, future=True, echo=False)
        Base.metadata.create_all(bind=sync_engine)
        # enable SQLite foreign keys pragma for sync engine (defensive)
        if sync_url.startswith("sqlite:///"):
            @event.listens_for(Engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, connection_record):
                try:
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()
                except Exception:
                    pass
        print("✅ Database tables created (sync).")
    except SQLAlchemyError as e:
        print("❌ Failed creating tables:", e)
        raise

# --- Modules management (file-backed) ---
def _load_modules():
    try:
        if not os.path.exists(MODULES_FILE):
            return []
        with open(MODULES_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def _save_modules(mods):
    try:
        os.makedirs(os.path.dirname(MODULES_FILE), exist_ok=True)
        with open(MODULES_FILE, "w", encoding="utf-8") as fh:
            json.dump(list(mods), fh, ensure_ascii=False, indent=2)
    except Exception as e:
        raise

def add_module(module_name: str):
    """
    Add a module name (idempotent). Raises on invalid input.
    """
    if not module_name or not str(module_name).strip():
        raise ValueError("invalid module_name")
    mods = _load_modules()
    if module_name in mods:
        return False
    mods.append(module_name)
    _save_modules(mods)
    return True

def get_modules():
    """Return list of modules (strings)."""
    return _load_modules()

def delete_module(module_name: str):
    mods = _load_modules()
    if module_name not in mods:
        return False
    mods = [m for m in mods if m != module_name]
    _save_modules(mods)
    return True

# CLI entry: create tables
if __name__ == "__main__":
    print("Creating DB tables...")
    create_table()
    print("Done.")
