# app/routers/predict.py
# Hanya urusan HTTP: terima request, panggil service, return response

from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_services import process_prediction

# APIRouter = "mini FastAPI" yang bisa di-mount ke main app
# prefix: semua endpoint di file ini otomatis punya prefix /predict
# tags: untuk grouping di Swagger UI
router = APIRouter(
    prefix="/predict",      # endpoint di file ini otomatis punya awalan /predict
    tags=["Prediction"]     # grouping di Swagger UI
    )

@router.post("/",
             response_model=PredictionResponse,
             status_code=status.HTTP_200_OK,
             summary="Prediksi risiko kredit",
             description="""
             **Input yang diharapkan:**
            - `age`: integer, antara 18 - 100
            - `income`: float, dalam satuan USD
            - `credit_score`: integer, antara 300 - 850

            **Catatan:** Model ini dilatih dengan data historis 2022 - 2023.
            Prediksi di luar rentang training data mungkin kurang akurat.
            """,
            # <- description mendukung Markdown - bold, list, link semua bisa
            response_description="Risk score berhasil dihitung", # label untuk response body
            responses={
                #dokumentasi explicit untuk status code selain 200
                400: {"description": "input tidak valid secara business logic"},
                422: {"description": "Validation Error - tipe data salah atau kosong"},
                500: {"description": "Model gagal load atau internal error"}
            },
        )
             
def predict_risk(request: PredictionRequest):
    """
    Endpoint utama prediksi risk scoring.

    Baris pertama docstring ini akan menggantikan summary= jika summary= tidak diset.
    Karena summary= sudah diset di decorator, baris ini tidak tampil di Swagger.
    Tapi tetap berguna sebagai inline documentation untuk developer yang baca source code.
    """
    try:
        #panggil service -- router tidak perlu tau cara kerjanya
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

@router.delete("/", status_code=status.HTTP_204_NO_CONTENT, description="Reset model", 
             summary="Delete id",
             responses={204: {"description": "Model berhasil direset"}})
def delete_id():
    return None

@router.get(
    "/health",
    summary="Health check",
    description="""Cek apakah pipeline sudah ter-load dan metadata-nya.""",
    response_description="Status model dan API",
)
def health_check():
    """Cek apakah pipeline sudah ter-load dan metadata-nya."""
    return {"status": "ready", "model_loaded": True, "version": "0.1.0"}