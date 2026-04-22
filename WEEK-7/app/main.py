# app/main.py
# Main app hanya bertanggung jawab: setup app dan daftarkan router
# Tidak ada business logic di sini

from fastapi import FastAPI

from app.routers import predict

app = FastAPI(
    title="RISK SCORING API",
    description="API untuk inferensi menggunakan Scikit-Learn Pipeline (Scaler + Model).",
    version="0.1.0"
)

#daftarkan semua router ke app utama
# seperti pasang papan menu di restoran

app.include_router(predict.router)

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint, wajib ada """
    return {"status": "ok", "version": "0.1.0"}
