# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import router — perhatikan path-nya: dari routers/items.py
from routers.items import router as items_router
from routers.health import router as health_router

app = FastAPI(
    title="ML Engineer API",
    description="Week 05 — Structured FastAPI Project",
    version="0.1.0"
)

# CORS middleware — kalau nanti ada frontend yang akses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Di production: ganti dengan domain spesifik
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(items_router)
app.include_router(health_router)
