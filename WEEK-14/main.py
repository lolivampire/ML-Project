# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from app.routers import predictions
import logging

app = FastAPI(
    title="Data Science System API",
    description="FastAPI + SQLAlchemy 2.0 Integration with Alembic (Async)",
    version="1.0.0"
)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            "details": exc.errors() # Menyertakan array detail error tadi
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