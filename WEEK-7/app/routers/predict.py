# app/routers/predict.py
# Hanya urusan HTTP: terima request, panggil service, return response

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_services import process_prediction

# APIRouter = "mini FastAPI" yang bisa di-mount ke main app
# prefix: semua endpoint di file ini otomatis punya prefix /predict
# tags: untuk grouping di Swagger UI
router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):
    """
    endpoint prediksi risiko kredit.
    menerima data pemohon, return skor dan label risiko
    """
    try:
        #panggil serbice -- router tidak perlu tau cara kerjanya
        result = process_prediction(request)
    
    except ValueError as e:
        #kalau service raise ValueError, kembalikan HTTPException (input validation error)
        # router yang akan konversi jadi HTTPException 422
        raise HTTPException(status_code=422, detail=str(e))
    
    #Konversi domain object -> HTTP response schema
    return PredictionResponse(
        risk_score=result.risk_score,
        risk_label=result.risk_label,
        message=result.message
    )