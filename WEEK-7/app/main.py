# app/main.py
# Main app hanya bertanggung jawab: setup app dan daftarkan router
# Tidak ada business logic di sini

from fastapi import FastAPI

from app.routers import predict

app = FastAPI(
    title="RISK SCORING API",
    description="""
## Risk Scoring API

API untuk memprediksi risk score user berdasarkan data finansial.

### Cara penggunaan
1. POST ke `/predict/` dengan request body sesuai schema
2. Response berisi `risk_score` (0–1) dan `risk_label`

### Autentikasi
Saat ini belum diimplementasikan. Coming soon di v2.
""",
    version="0.1.0",
    contact={
        "name": "ML Engineer",
        "url": "https://github.com/lolivampire/ML-Project",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        # ← urutan dan deskripsi tiap tag/grup di Swagger
        {"name": "Prediction", "description": "Endpoint untuk prediksi risk scoring"},
        {"name": "Health", "description": "Status check dan monitoring"},
    ],
)

#daftarkan semua router ke app utama
#seperti pasang papan menu di restoran

app.include_router(predict.router)

# @app.get("/health", tags=["System"])
# def health_check():
#     """Health check endpoint, wajib ada """
#     return {"status": "ok", "version": "0.1.0"}
