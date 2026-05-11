from pathlib import Path
from pydantic_settings import BaseSettings

# __file__ = lokasi absolut config.py ini sendiri
# .parent = folder app/
# .parent.parent = root project (di mana .env berada)
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    MODEL_PATH: str = str(BASE_DIR / "app/model/pipeline_pipe_v1_20260415.joblib")
    APP_VERSION: str = "1.0.0"

    model_config = {
        "env_file": str(BASE_DIR / ".env"),  # path absolut, tidak peduli dari mana dijalankan
        "env_file_encoding": "utf-8",
    }

settings = Settings()