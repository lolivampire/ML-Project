# app/main.py
from fastapi import FastAPI
from app.config import settings

# Import dari modul yang baru kita refactor
from app.core.lifespan import lifespan
from app.routers import predict, health
from app.middleware.logging_middleware import RequestLoggingMiddleware

# Inisialisasi Aplikasi
app = FastAPI(
    title="Risk Scoring API",
    description="ML API for risk scoring",
    version=settings.app_version,
    lifespan=lifespan
)

# Registrasi Middleware
app.add_middleware(RequestLoggingMiddleware)

# Registrasi Routers
app.include_router(health.router)
app.include_router(predict.router)