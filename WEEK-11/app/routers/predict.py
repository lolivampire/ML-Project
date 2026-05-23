# routers/predict.py
# Layering bersih.
# PERBAIKAN: get_prediction_repo sekarang mengambil memori persisten dari app.state.

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import List, Dict, Any
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_services import ModelService
from app.repositories.prediction_repo import PredictionRepository
from app.model.loader import get_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])

# --- Factory Dependency---
def get_prediction_store(request: Request) -> List[Dict[str, Any]]:
    return request.app.state.prediction_store

def get_prediction_repo(store: List[Dict[str, Any]] = Depends(get_prediction_store)) -> PredictionRepository:
    return PredictionRepository(store=store)

def get_model_service(
    request: Request,
    repo: PredictionRepository = Depends(get_prediction_repo),
) -> ModelService:
    model = get_model(request) 
    return ModelService(model=model, repo=repo)

# --- Endpoints Tetap Sama ---
@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    payload: PredictionRequest,
    service: ModelService = Depends(get_model_service),
):
    try:
        features = payload.to_feature_list()  # Terima beres dari ahlinya (Schema)
        result = service.predict(features)    # Kirim langsung ke service
        return result
    except ValueError as e:
        logger.warning(f"Input tidak valid: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Inference gagal: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Terjadi kesalahan saat inference.")

@router.get("/history", summary="Lihat riwayat prediksi")
async def get_history(service: ModelService = Depends(get_model_service)):
    return service.repo.get_all()