from fastapi import APIRouter, HTTPException, Request
from app.schemas.predictions import PredictionRequest, PredictionResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    """
    Endpoint prediksi utama.
    """
    try:
        # EKSTRAKSI DINAMIS: Ubah semua field di Pydantic menjadi list.
        # Jika besok fiturnya nambah jadi 10, kode router ini TIDAK PERLU diubah!
        features = list(payload.model_dump().values())
        
        # Ambil service model yang sudah di-load di startup
        model_service = request.app.state.model_service
        
        result = model_service.predict(features)
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}") # Log error asli untuk debugging
        raise HTTPException(status_code=500, detail="Prediction failed. Please check server logs.")