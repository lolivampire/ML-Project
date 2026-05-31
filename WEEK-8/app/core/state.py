# app/core/state.py
from dataclasses import dataclass
from typing import Optional
from redis import Redis
from app.services.model_service import ModelService

@dataclass
class AppState:
    """
    Menyimpan state global aplikasi.
    Menggunakan pattern Singleton.
    """
    model_service: Optional[ModelService] = None
    redis_client: Optional[Redis] = None
    ready: bool = False
    startup_error: Optional[str] = None

# Singleton instance
app_state = AppState()