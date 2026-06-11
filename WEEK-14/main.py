from fastapi import FastAPI
from app.routers import predictions

from app.database import engine, Base
import app.models

# Membuat semua tabel jika belum ada
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Data Science System API",
    description="FastAPI + SQLAlchemy 2.0 Integration with Alembic",
    version="1.0.0"
)

# Daftarkan router
app.include_router(predictions.router)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "DSS API is running perfectly"}