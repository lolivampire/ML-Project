#model/loader.py
import joblib
from fastapi import Request
from pathlib import Path
from app.config import settings

def load_model_from_disk():
    """Load model sekali, cache selamanya (per process)."""
    if not settings.MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan di lokasi: {settings.MODEL_PATH}")
    return joblib.load(settings.MODEL_PATH)

def get_model(request: Request):
    """
    Dependency function untuk FastAPI Depends().
    Mengambil model terpusat yang aman dari state aplikasi.
    """
    if not hasattr(request.app.state, "model"):
        raise RuntimeError("Model belum dimuat ke dalam app.state.")
    return request.app.state.model