"""
config.py — Centralized configuration loader.

Pattern: Semua environment variables dibaca dan divalidasi DI SINI saja.
Tidak ada pemanggilan os.getenv() yang berserakan di seluruh codebase.
"""
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Menentukan direktori dasar (root) project
# __file__ = lokasi absolut config.py ini
# .parent = folder app/ (asumsi script ini berada di app/config.py)
# .parent.parent = root project (di mana file .env berada)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- Application ---
    app_name: str = "risk-scoring-api"
    app_version: str = "1.0.0"
    # Pydantic otomatis mengonversi string env ("true"/"false") menjadi bool
    debug: bool = False 
    log_level: str = "info"

    # --- Security ---
    # Field(...) menandakan variabel ini WAJIB (Required). 
    # Jika API_KEY tidak ada di .env/sistem, Pydantic akan langsung memicu error
    # saat startup (Fail Fast), menggantikan fungsi manual _require().
    api_key: str = Field(..., description="API Key untuk otentikasi, WAJIB diisi.")

    # --- Redis ---
    redis_host: str = "redis"
    redis_port: int = 6379  # Otomatis di-cast dari string ke integer
    redis_password: str = ""

    # --- Model ---
    # Menggunakan BASE_DIR untuk memastikan path absolut, 
    # tetapi nilainya tetap bisa ditimpa (override) via env var "MODEL_PATH"
    model_path: str = str(BASE_DIR / "app/model/pipeline_pipe_v1_20260415.joblib")

    # --- Pydantic Configuration ---
    model_config = SettingsConfigDict(
        # Membaca .env dulu, lalu ditimpa oleh file spesifik jika ada
        env_file=(
            str(BASE_DIR / ".env"), 
            str(BASE_DIR / f".env.{os.getenv('ENV_STATE', 'dev')}")
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton: dibuat sekali saat module ini diimport.
# Aplikasi Anda akan langsung "Crash/Fail-Fast" di baris ini jika API_KEY tidak ada.
# Cara pakai di file lain: `from config import settings`
settings = Settings()