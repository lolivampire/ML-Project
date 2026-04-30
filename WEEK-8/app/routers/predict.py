from fastapi import APIRouter, HTTPException, Request
from app.schemas.predictions import PredictionRequest, PredictionResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse, summary="Predict binary class from synthetic features")
async def predict(payload: PredictionRequest, request: Request):
    """
    Runs inference on 4 standardized synthetic features using a
    trained scikit-learn pipeline.

    Returns a binary prediction (0 or 1) and confidence probability.
    Model was trained on make_classification synthetic data (500 samples,
    4 features, 3 informative, 1 redundant).
    """
    try:
        # EKSTRAKSI DINAMIS: Ubah semua field di Pydantic menjadi list.
        # Jika besok fiturnya nambah jadi 10, kode router ini TIDAK PERLU diubah!
        # Eksplisit ambil hanya feature fields
        FEATURE_FIELDS = ["feature_1", "feature_2", "feature_3", "feature_4"]
        features = [payload.model_dump()[f] for f in FEATURE_FIELDS]
        
        # Ambil service model yang sudah di-load di startup
        model_service = request.app.state.model_service
        
        result = model_service.predict(features)
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}") # Log error asli untuk debugging
        raise HTTPException(status_code=500, detail="Prediction failed. Please check server logs.")