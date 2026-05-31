# app/core/lifespan.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.core.logger import get_json_logger
from app.core.state import app_state
from app.database import create_redis_client
from app.services.model_service import ModelService

logger = get_json_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────
    logger.info("app_starting")
    
    try:
        # 1. Load ML Model
        logger.info("loading_model", extra={"model_path": settings.model_path})
        model_service = ModelService(settings.model_path, settings.app_version)
        model_service.load_model()
        app_state.model_service = model_service
        
        # 2. Connect ke Redis
        logger.info("connecting_redis", extra={"host": settings.redis_host})
        redis_client = create_redis_client(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password
        )
        app_state.redis_client = redis_client
        
        # 3. Tandai siap
        # Simulasi: dependency gagal saat startup
        app_state.ready = True
        logger.info("app_ready")
        
    except Exception as e:
        app_state.startup_error = str(e)
        logger.error("startup_failed", extra={"error": str(e)}, exc_info=True)
        # Jika gagal, aplikasi tetap jalan (alive) tapi siap-siap mengembalikan 503 di /ready

    # ── APP BERJALAN ──────────────────────────────────────────
    yield

    # ── GRACEFUL SHUTDOWN ─────────────────────────────────────
    logger.info("app_shutting_down")
    try:
        # Stop traffic baru masuk
        app_state.ready = False
        
        # Putuskan koneksi database dengan rapi (cegah memory leak)
        if app_state.redis_client:
            app_state.redis_client.close()
            logger.info("redis_disconnected")
            
        # Kosongkan memori model (cegah OOM saat restart cepat)
        app_state.model_service = None
        app_state.redis_client = None
        
        logger.info("app_shutdown_complete")
        
    except Exception as e:
        logger.error("shutdown_error", extra={"error": str(e)}, exc_info=True)