from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import predict
from app.config import settings
from app.services.model_service import ModelService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info(f"[STARTUP] Risk Scoring API v{settings.app_version} starting...")
    
    # Instansiasi dan load model
    model_service = ModelService(settings.model_path, settings.app_version)
    model_service.load_model()
    
    # Simpan di app state agar bisa diakses oleh router
    app.state.model_service = model_service
    
    yield
    
    # --- SHUTDOWN ---
    logger.info("[SHUTDOWN] Server shutting down. Cleaning up resources...")
    app.state.model_service = None

app = FastAPI(
    title="Risk Scoring API",
    description="ML API for risk scoring  EPhase 2 Project 1",
    version=settings.app_version,
    lifespan=lifespan
)

# Register router
app.include_router(predict.router)

@app.get("/health", tags=["system"], summary="Check API and model status")
async def health_check():
    """
    Returns current API status and loaded model version.
    Use this to verify the service is running before sending predictions.
    """
    return {"status": "ok", "version": settings.app_version}

#iniadalahkomentaruntuktestingdockerimage
