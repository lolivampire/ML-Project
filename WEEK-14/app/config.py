from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://dss_user:secret@localhost:5432/dss_db"
    redis_url: str = "redis://redis:6379/0"
    app_name: str = "Data Science System API"
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()