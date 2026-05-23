# main.py
# file bersih.
# PERBAIKAN: Menambahkan inisialisasi app.state.prediction_store

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.model.loader import load_model_from_disk
from app.routers import predict
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("Startup: memuat resources ke memori...")
    try:
        app.state.model = load_model_from_disk()
        app.state.prediction_store = []  # DITAMBAHKAN: Memori untuk Repository In-Memory
        logger.info("Resources berhasil dimuat. Server siap.")
    except Exception as e:
        logger.critical(f"FATAL: Gagal memuat resources: {e}")
        raise 

    yield 

    # --- SHUTDOWN ---
    if hasattr(app.state, "model"): del app.state.model
    if hasattr(app.state, "prediction_store"): del app.state.prediction_store
    logger.info("Shutdown: resources di-unload dari memori.")

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)
app.include_router(predict.router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "store_ready": hasattr(app.state, "prediction_store"),
    }