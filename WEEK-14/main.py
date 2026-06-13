"""
main.py (The Entry Point & Lifespan)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
import logging
from fastapi.encoders import jsonable_encoder

from app.database import async_engine 
from app.config import settings
from app.routers import predictions 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Memulai Data Science System API...")
    yield
    logger.info("Mematikan server, membuang sisa koneksi pool database...")
    
    # 2. Ubah juga pemanggilannya di sini
    await async_engine.dispose() 

app = FastAPI(
    title=settings.app_name,
    description="Layered Architecture with Async FastAPI",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(predictions.router)
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "DSS API is running perfectly"}

# ── 1. Handler untuk Error Validasi Pydantic (422) ────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Menyeragamkan output error bawaan Pydantic"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_type": "Validation Error",
            "message": "Data yang dikirim tidak sesuai dengan format yang diminta.",
            "details": jsonable_encoder(exc.errors()) # Menyertakan array detail error tadi
        }
    )

# ── 2. Handler untuk Error Database SQLAlchemy ─────────────────────
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Menangkap error dari PostgreSQL (misal: koneksi putus, integrity error)"""
    # Log error aslinya ke terminal server agar developer bisa investigasi
    logger.error(f"Database error: {str(exc)}")
    
    # Return pesan yang aman ke client (jangan ekspos detail SQL)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": "Database Error",
            "message": "Terjadi gangguan pada sistem penyimpanan data. Silakan coba beberapa saat lagi."
        }
    )

# ── 3. Handler Catch-All untuk Semua Error Python Lainnya ──────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Jaring terakhir untuk menangkap error seperti IndexError, KeyError, dll."""
    logger.error(f"Unhandled server error: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": "Internal Server Error",
            "message": "Terjadi kesalahan internal pada server kami."
        }
    )