# app/database.py
from redis import asyncio as aioredis
from contextlib import contextmanager
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.config import settings
# ─── Lakukan konversi URL langsung di sini ───────────────────────
ASYNC_URL = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

# ─── 2. ASYNC ENGINE & SESSION (UNTUK FASTAPI) ───────────────────
async_engine = create_async_engine(
    ASYNC_URL,  # <-- Gunakan variabel lokal ASYNC_URL di sini
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False,
)

# ─── BASE MODEL ───────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class untuk arsitektur ORM SQLAlchemy 2.0"""
    pass

SessionLocalAsync = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False 
)

async def get_db():
    """Dependency injection untuk endpoint FastAPI."""
    async with SessionLocalAsync() as db:
        try:
            yield db
            # PERHATIAN: Tidak ada lagi await db.commit() di sini!
            # Commit sekarang diurus oleh Service Layer.
        finally:
            await db.close()

# ─── 3. SYNC ENGINE & MANAGER (UNTUK CLI/BACKGROUND SCRIPTS) ─────
sync_engine = create_engine(settings.database_url, pool_pre_ping=True)

class DatabaseManager:
    def __init__(self, engine_instance):
        self.engine = engine_instance 
        self._session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager untuk script CLI (bukan untuk FastAPI)."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

# Inisialisasi global manager untuk diimpor oleh script CLI
db_manager = DatabaseManager(engine_instance=sync_engine)

async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """Dependency injection untuk client Redis Asinkronus di FastAPI."""
    client = aioredis.Redis.from_url(
        settings.redis_url, 
        decode_responses=True
    )
    try:
        yield client
    finally:
        await client.close()