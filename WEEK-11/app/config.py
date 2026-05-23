# app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict # Sesuaikan import di atas file

# Path(__file__).resolve() adalah lokasi app/config.py itu sendiri
# .parent artinya naik 1 level ke folder app/
# .parent.parent artinya naik 2 level ke root folder WEEK-11/
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # --- APP CONFIG ---
    APP_TITLE: str = "Risk Scoring API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # --- MODEL CONFIG ---
    # Menghasilkan path absolut ke: WEEK-11/app/model/model.pkl
    MODEL_PATH: Path = BASE_DIR / "app" / "model" / "model.pkl"
    MODEL_VERSION: str = "2.0.0"

    # --- BUSINESS LOGIC CONFIG ---
    RISK_LOW_MAX: float = 0.40
    RISK_MEDIUM_MAX: float = 0.70
    
    # Hapus 'class Config:' dan ganti menjadi atribut model_config
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()