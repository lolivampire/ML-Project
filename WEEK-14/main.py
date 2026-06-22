"""
main.py (The Entry Point & Lifespan)
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from fastapi.encoders import jsonable_encoder
from prometheus_fastapi_instrumentator import Instrumentator

# --- IMPORT OPENTELEMETRY ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from app.database import async_engine 
from app.config import settings
from app.routers import predictions
from app.core.logging_config import setup_global_logging
from app.middleware import MetricsMiddleware
from app.metrics import MODEL_LOADED_STATUS
from app.routers import decision_router 

# --- SETUP OPENTELEMETRY TRACER PUSAT --
# Mendefinisikan nama service agar mudah dicari di Jaeger
resource = Resource.create({"service.name": "data-science-api"})
provider = TracerProvider(resource=resource)
# Mengarahkan tembakan data trace ke container Jaeger di port gRPC 4317
otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Aktifkan Instrumentasi untuk SQLAlchemy Async Engine
# Kita gunakan .sync_engine karena OTel mencegat di level connection pool sinkron internal
SQLAlchemyInstrumentor().instrument(engine=async_engine.sync_engine)

# Deklarasikan logger untuk file ini menggunakan __name__
logger = logging.getLogger(__name__)

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Konfigurasi Logging
    # 1. Jalankan konfigurasi induk
    setup_global_logging()
    # Simulasi memuat model ke dalam memori saat server menyala
    MODEL_LOADED_STATUS.labels(model_name="xgboost_v1").set(1)
    MODEL_LOADED_STATUS.labels(model_name="random_forest_v2").set(1)
    logger.info("Memulai Data Science System API...")
    
    yield

    # Membebaskan memori saat server dimatikan
    MODEL_LOADED_STATUS.labels(model_name="xgboost_v1").set(0)
    MODEL_LOADED_STATUS.labels(model_name="random_forest_v2").set(0)

    logger.info("Mematikan server, membuang sisa koneksi pool database...")
    
    # 2. Ubah juga pemanggilannya di sini
    await async_engine.dispose() 

app = FastAPI(
    title=settings.app_name,
    description="Layered Architecture with Async FastAPI",
    version="2.0.0",
    lifespan=lifespan
)

# --- AKTIFKAN AUTO-INSTRUMENTATION OPENTELEMETRY ---
# Ini akan otomatis mencegat request masuk dan menyuntikkan Span
FastAPIInstrumentor.instrument_app(app)

# Daftarkan Middleware metrik Prometheus yang sudah dirampingkan
app.add_middleware(MetricsMiddleware)


app.include_router(predictions.router)
app.include_router(decision_router.router)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "DSS API is running perfectly"}

# ── 1. Handler untuk Error Validasi Pydantic (422) ────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Menyeragamkan output error bawaan Pydantic"""
    logger.warning(f"Request ditolak karena gagal validasi data dari {request.client.host}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_type": "Validation Error",
            "message": "Data yang dikirim tidak sesuai dengan format yang diminta.",
            "details": jsonable_encoder(exc.errors()) # Menyertakan array detail error
        }
    )

# ── 2. Handler untuk Error Database SQLAlchemy ─────────────────────
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Menangkap error dari PostgreSQL (misal: koneksi putus, integrity error)"""
    # Log error aslinya ke terminal server agar developer bisa investigasi
    logger.error(f"Database error terjadi: {str(exc)}")
    
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
    logger.critical(f"Unhandled server error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": "Internal Server Error",
            "message": "Terjadi kesalahan internal pada server kami."
        }
    )

# Inisialisasi dan ekspos endpoint Prometheus Instrumentator (/metrics)
# Pastikan menggunakan .instrument(app).expose(app) agar tercatat sempurna
Instrumentator().instrument(app).expose(app)