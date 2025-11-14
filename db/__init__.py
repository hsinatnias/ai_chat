# db/__init__.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Default to a local SQLite file for development.
# If you want an absolute path, set DATABASE_URL in your .env like:
# DATABASE_URL=sqlite+aiosqlite:////full/path/to/data/app.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/app.db")

# Create the async engine for SQLAlchemy (works with aiosqlite, asyncpg, etc.)
engine = create_async_engine(DATABASE_URL, future=True, echo=False)

# Async session factory
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_async_db():
    """
    Dependency to provide an async SQLAlchemy session.
    Usage in FastAPI:
        async def endpoint(db: AsyncSession = Depends(get_async_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session
        
       

