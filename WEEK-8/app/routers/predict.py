# app/routers/predict.py
import logging
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel

# Gunakan factory logger JSON yang sudah kita buat
from app.core.logger import get_json_logger

logger = get_json_logger(__name__)
router = APIRouter(prefix="/predict", tags=["machine-learning"])

# Contoh skema input untuk Risk Scoring
class RiskInput(BaseModel):
    income: int
    age: int
    debt_ratio: float

@router.post("", summary="Execute Risk Scoring Prediction")
async def predict_risk(request: Request, payload: RiskInput):
    """
    Endpoint untuk mengeksekusi prediksi Risk Scoring.
    Mencatat log payload input dan hasil output secara terstruktur.
    """
    # 1. Log saat request masuk ke level Router (Menampilkan Payload Input)
    # Kita bungkus model pydantic ke .model_dump() agar menjadi dictionary yang valid di JSON Log
    logger.info(
        "Router received prediction request", 
        extra={"payload": payload.model_dump()}
    )
    
    try:
        # Ambil model_service dari global app state yang di-set di lifespan main.py
        model_service = request.app.state.model_service
        
        # Jalankan kalkulasi/prediksi (analogi proses bisnis)
        # Misal hasil berupa skor risiko dan keputusan
        prediction_result = {
            "risk_score": 0.24,
            "decision": "APPROVED",
            "model_version": model_service.version if model_service else "unknown"
        }
        
        # 2. Log saat response akan keluar dari level Router (Menampilkan Result Output)
        logger.info(
            "Router prediction execution successful", 
            extra={"result": prediction_result}
        )
        
        return prediction_result

    except Exception as e:
        # Log jika terjadi kegagalan spesifik di dalam proses prediksi
        logger.error(
            f"Prediction execution failed: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal memproses prediksi skor risiko."
        )