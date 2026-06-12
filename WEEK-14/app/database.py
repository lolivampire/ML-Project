# week-14/scripts/database.py
import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from dotenv import load_dotenv

load_dotenv()

# Baca URL asli dari .env (format: postgresql://...) agar Alembic tetap aman.
SYNC_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://dss_user:secret@localhost:5432/dss_db"
)

# Modifikasi URL secara on-the-fly untuk FastAPI
ASYNC_URL = SYNC_URL.replace("postgresql://", "postgresql+asyncpg://")

# 2. ASYNC ENGINE
engine = create_async_engine(
    ASYNC_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False,
)

# 3. ASYNC SESSION MAKER
# expire_on_commit=False wajib ada di async agar data tidak hilang dari memori setelah di-commit
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False 
)

class Base(DeclarativeBase):
    """Base class untuk arsitektur pengenalan model SQLAlchemy 2.0"""
    pass

# Untuk CLI scripts — sync engine terpisah
sync_engine = create_engine(SYNC_URL, pool_pre_ping=True)
class DatabaseManager:
    def __init__(self, engine_instance, echo: bool = True):
        # Engine bertindak sebagai pool koneksi fisik ke Postgres
        self.engine = engine_instance 
        # Sessionfactory untuk memproduksi unit of work transaksi
        self._session_factory = sessionmaker(
            bind=sync_engine,
            autocommit=False,
            autoflush=False
        )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager untuk memastikan session selalu ditutup setelah dipakai."""
        session = self._session_factory()
        try:
            yield session
            session.commit()  # Auto-commit jika blok kode sukses
        except Exception:
            session.rollback()  # Auto-rollback jika terjadi error di tengah jalan
            raise
        finally:
            session.close()  # Pastikan koneksi dibebaskan kembali ke pool

# Inisialisasi global manager
db_manager = DatabaseManager(engine_instance=engine)

async def get_db():
    """Dependency injection helper untuk aplikasi atau API layer"""
    async with SessionLocal() as db:
        try:
            yield db
            await db.commit()
        except Exception:
            await db.rollback()
            raise