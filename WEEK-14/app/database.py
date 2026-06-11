# week-14/scripts/database.py
import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://dss_user:secret@localhost:5432/dss_db" # Fallback ke DB
)

# engine = connection pool manager
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# SessionLocal = pabrik pembuat session untuk setiap request
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Base(DeclarativeBase):
    """Base class untuk arsitektur pengenalan model SQLAlchemy 2.0"""
    pass

class DatabaseManager:
    def __init__(self, engine_instance, echo: bool = True):
        # Engine bertindak sebagai pool koneksi fisik ke Postgres
        self.engine = engine_instance 
        # Sessionfactory untuk memproduksi unit of work transaksi
        self._session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
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

def get_db() -> Generator[Session, None, None]:
    """Dependency injection helper untuk aplikasi atau API layer"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()