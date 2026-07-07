"""
week-06/app/main.py

Aplikasi FastAPI utama. Bertugas mengelola siklus hidup server (lifespan)
untuk memuat Scikit-Learn Pipeline ke dalam memori sebelum menerima request.
"""
from fastapi.encoders import jsonable_encoder
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
import joblib
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# import routers dan logger
from app.routers import predict
from app.core.logger import get_logger

# ── UTILITY UNTUK PATH ────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR.parent / "scripts" / "models"

logger = get_logger(__name__)

# ── LIFESPAN ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Memulai Server ===")
    
    # Cari semua file .joblib yang namanya diawali dengan 'pipeline_'
    # Ini memastikan kita hanya meload file pipeline, bukan model lawas
    logger.info("Memulai... Mencari file pipeline bundle...")
    pipeline_files = list(MODELS_DIR.glob("pipeline_*.joblib"))
    
    if not pipeline_files:
        logger.error(f"Tidak ada file pipeline bundle .joblib di {MODELS_DIR}")
        raise FileNotFoundError(f"Tidak ada file pipeline .joblib di {MODELS_DIR}")
    
    # Ambil file pipeline yang paling baru
    latest_pipeline_path = max(pipeline_files, key=lambda p: p.stat().st_mtime)
    
    try:
        logger.info(f"Memuat pipeline dari: {latest_pipeline_path.name}...")
        bundle = joblib.load(latest_pipeline_path)
        
        # Validasi struktur bundle (Defensive Programming)
        if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
            raise ValueError("Format bundle tidak valid. Pastikan itu adalah pipeline bundle.")

        # Simpan ke app.state
        app.state.pipeline = bundle["pipeline"]
        app.state.metadata = bundle["metadata"]
        
        # Ekstrak data krusial untuk kemudahan akses di router
        app.state.n_features = bundle["metadata"].get("n_features")
        app.state.classes = bundle["metadata"].get("classes")
        
        logger.info(
            "✅ Pipeline berhasil dimuat. Server siap untuk menerima request.",
            extra={"extra_fields": {"n_features": app.state.n_features, 
                                    "accuracy": bundle['metadata'].get("accuracy")
                                    }
                    }
        )
        
    except Exception as e:
        # Hard fail: Jika pipeline gagal di-load, server batal naik.
        print(f"[startup] Gagal memuat pipeline: {e}")
        raise e

    yield 

    logger.info("Shutting down... cleaning up memory")
    app.state.pipeline = None
    app.state.metadata = None
    app.state.n_features = None
    app.state.classes = None


# ── INISIALISASI APP ──────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Pipeline API",
    description="API untuk inferensi menggunakan Scikit-Learn Pipeline (Scaler + Model).",
    lifespan=lifespan
)

# ── GLOBAL EXCEPTION HANDLERS ─────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Menangkap error 422 dari Pydantic (Skema Input Salah).
    Mengubah format default FastAPI agar lebih bersih dan mencatatnya ke log.
    """
    request_id = getattr(request.state, "request_id", "unknown-id")
    
    # Ambil detail error dari Pydantic
    errors = jsonable_encoder(exc.errors())
    
    logger.warning(
        "Global Handler: Input Validation Failed (422)",
        extra={"extra_fields": {
            "request_id": request_id,
            "errors": errors
        }}
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error_type": "ValidationError",
            "message": "Input tidak sesuai dengan skema yang diharapkan.",
            "detail": errors
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Menangkap semua HTTPException (400, 404, 503, dll) yang kita raise secara manual.
    """
    request_id = getattr(request.state, "request_id", "unknown-id")
    
    # Log sebagai warning untuk error client (4xx), error untuk server (5xx)
    if exc.status_code >= 500:
        logger.error(f"Global Handler: HTTP Exception ({exc.status_code})", extra={"extra_fields": {"request_id": request_id, "detail": exc.detail}})
    else:
        logger.warning(f"Global Handler: HTTP Exception ({exc.status_code})", extra={"extra_fields": {"request_id": request_id, "detail": exc.detail}})

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": "HTTPException",
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def global_catch_all_handler(request: Request, exc: Exception):
    """
    Jaring pengaman terakhir (Catch-All). 
    Menangkap error Python murni yang lolos/lupa di-try-except (500 Internal Server Error).
    """
    request_id = getattr(request.state, "request_id", "unknown-id")
    
    # Catat errornya lengkap dengan Stack Trace
    logger.error(
        "Global Handler: Unhandled Server Crash (500)",
        extra={"extra_fields": {"request_id": request_id, "error_msg": str(exc)}},
        exc_info=True
    )
    
    # Sembunyikan detail teknis (str(exc)) dari user untuk keamanan!
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "InternalServerError",
            "message": "Terjadi kesalahan internal pada server. Harap hubungi administrator."
        }
    )

# ── MIDDLEWARE LOGGING ──────────────────────────────────────────────────────────

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Middleware untuk mencatat metadata setiap request masuk dan keluar.
    input/output data spesifik akan tercatat di level router.
    """
    # 1. Generate ID untuk request ini
    request_id = str(uuid.uuid4())[:8]

    #titipkan ID ini ke request.state agar bisa diakses oleh predict.py
    request.state.request_id = request_id

    start_time = time.perf_counter()

    # 2. log request masuk
    logger.info(
        "Request masuk",
        extra={"extra_fields": {
                                "request_id": request_id,
                                "method": request.method,
                                "path": request.url.path,
                                "client_ip": request.client.host if request.client else "client ip unknown"
                                }
                }
    )

    # 3. Jalankan request ke endpoint (router)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        #jika terjadi crash/error 500 yang tidak tertangkap di endpoint
        logger.error(f"unhandled exception: {e}", extra={"extra_fields": {"request_id": request_id}})
        raise e
    finally:
        # 4. Hitung latency dan log request keluar (pastikan selalu berjalan dengan finally)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "request selesai",
            extra={"extra_fields": {
                                    "request_id": request_id,
                                    "status_code": status_code, 
                                    "latency_ms": latency_ms
                                    }
                    }
        )

    return response

# Registrasi Router
app.include_router(predict.router)

@app.get("/")
async def root():
    return {
        "message": "API Pipeline aktif!",
        "docs": "Kunjungi /docs untuk mencoba."
    }