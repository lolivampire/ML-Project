# app/main.py
from contextlib import asynccontextmanager
import logging

import redis
from fastapi import FastAPI, HTTPException, Request, status

from app.config import settings
from app.database import create_redis_client
from app.routers import predict
from app.services.model_service import ModelService

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info(f"[STARTUP] Risk Scoring API v{settings.app_version} starting...")
    
    # 1. Instansiasi dan load Machine Learning model
    model_service = ModelService(settings.model_path, settings.app_version)
    model_service.load_model()
    
    # 2. Inisialisasi koneksi Redis dengan retry logic eksponensial
    logger.info("[STARTUP] Menyiapkan koneksi ke Redis...")
    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password
    )
    
    # 3. Simpan di app state agar bisa diakses secara global oleh router & endpoint
    app.state.model_service = model_service
    app.state.redis_client = redis_client
    
    yield  # --- APLIKASI BERJALAN ---
    
    # --- SHUTDOWN ---
    logger.info("[SHUTDOWN] Server shutting down. Cleaning up resources...")
    app.state.model_service = None
    
    # Tutup koneksi Redis dengan aman jika sebelumnya berhasil terhubung
    if hasattr(app.state, "redis_client") and app.state.redis_client:
        app.state.redis_client.close()
        logger.info("[SHUTDOWN] Koneksi Redis berhasil ditutup.")


app = FastAPI(
    title="Risk Scoring API",
    description="ML API for risk scoring EPhase 2 Project 1",
    version=settings.app_version,
    lifespan=lifespan
)

# Register router
app.include_router(predict.router)


@app.get("/health", tags=["system"], summary="Check API, model, and Redis status")
async def health_check(request: Request):
    """
    Returns current API status, loaded model version, and Redis connectivity.
    Use this to verify the service is running before sending predictions.
    """
    redis_client = request.app.state.redis_client
    model_service = request.app.state.model_service

    # --- Cek Redis ---
    try:
        redis_client.ping()
        redis_status = "connected"
    except (redis.RedisError, Exception) as e:
        logger.error(f"[HEALTHCHECK] Redis ping gagal: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Unavailable: Gagal terhubung ke Redis"
        )

    # --- Cek Model ---
    # model_service.model adalah attribute yang di-set saat load_model() berhasil
    # Jika None, berarti startup load gagal — prediksi tidak mungkin berjalan
    if model_service is None or model_service.model is None:
        logger.error("[HEALTHCHECK] Model belum ter-load.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Unavailable: Model belum ter-load"
        )

    return {
        "status": "healthy",
        "redis": redis_status,
        "model_version": model_service.version,   # pakai attribute dari ModelService
        "version": settings.app_version
    }