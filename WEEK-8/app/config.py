from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "app/model/pipeline_pipe_v1_20260415.joblib"
    APP_VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"

# Instansiasi settings
settings = Settings()