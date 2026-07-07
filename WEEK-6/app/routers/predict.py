"""
week-06/app/routers/predict.py
"""

import math
from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
import numpy as np



from app.core.logger import get_logger
from app.dependencies.model_loader import get_pipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["prediction"])

# ── LAPIS 1: SCHEMA VALIDATION (Otomatis 422 jika gagal) ──────────────

class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=1, # Pydantic akan melempar 422 jika list kosong
        description="Vektor fitur untuk prediksi."
    )
    @field_validator("features")
    @classmethod
    def validate_nan_inf(cls, value: list[float]) -> list[float]:
        for index, val in enumerate(value):
            if math.isnan(val):
                raise ValueError(f"Fitur pada index {index} bernilai NaN.")
            if math.isinf(val):
                raise ValueError(f"Fitur pada index {index} bernilai Infinity.")
        return value

class PredictResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    input_received: list[float]


# ── ENDPOINT UTAMA ────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=PredictResponse,         # FastAPI generate response schema di docs
    summary="Run model prediction",         # judul singkat di Swagger
    description="""
    Menerima array fitur numerik dan mengembalikan hasil prediksi model.
    
    **Validasi:**
    - Array tidak boleh kosong
    - Semua elemen harus numerik (float/int)
    - Tidak boleh mengandung NaN atau Infinity
    - Nilai harus dalam range yang wajar (0–1000)
    
    **Error codes:**
    - 422: Schema tidak valid (tipe salah, field hilang)
    - 400: Nilai tidak wajar (out of range, wrong count)
    - 503: Model belum loaded
    """,
    responses={
        400: {"description": "Domain validation error"},
        422: {"description": "Input schema validation error"},
        503: {"description": "Model not available"},
    }
)
async def predict(
    request: Request, 
    body: PredictRequest,
    pipeline: Any = Depends(get_pipeline) # Lapis 3 Guard (Otomatis 503 jika model belum siap)
):
    request_id = getattr(request.state, "request_id", "unknown-id")
    features = body.features

    # ── LAPIS 2: DOMAIN / BUSINESS LOGIC VALIDATION (Manual 400) ──────
    
    # A. Validasi Dinamis: Cek panjang input vs metadata model
    expected_features = getattr(request.app.state, "n_features", None)
    if expected_features and len(features) != expected_features:
        logger.warning(
            "Validation Error: Mismatch feature length",
            extra={"extra_fields": {
                "request_id": request_id, 
                "expected": expected_features, 
                "received": len(features)
            }}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Bad Request: Model mengharapkan tepat {expected_features} fitur, diterima {len(features)}."
        )

    # B. Validasi Logika Bisnis (Contoh: Fitur 0 dan 1 tidak boleh negatif)
    if len(features) >= 2 and any(f < 0 for f in features[:2]):
        logger.warning(
            "Validation Error: Negative values detected",
            extra={"extra_fields": {"request_id": request_id, "invalid_values": features[:2]}}
        )
        raise HTTPException(
            status_code=400,
            detail="Bad Request: Fitur index 0 dan 1 (misal: usia/pendapatan) tidak boleh bernilai negatif."
        )

    # ── LAPIS 3: INFERENSI & FAIL-SAFE (Manual 500) ───────────────────
    
    try:
        # Reshape data untuk Scikit-Learn
        X = np.array(features).reshape(1, -1)
        
        # Pipeline otomatis mengeksekusi Scaler -> Model
        prediction_id = int(pipeline.predict(X)[0])
        
        probability = None
        if hasattr(pipeline, "predict_proba"):
            probability = round(float(pipeline.predict_proba(X)[0][prediction_id]), 4)

        # Log Sukses
        logger.info(
            "Prediction successful", 
            extra={"extra_fields": {
                "request_id": request_id,
                "prediction": prediction_id,
                "probability": probability
            }}
        )

        return PredictResponse(
            prediction=prediction_id,
            probability=probability,
            input_received=features
        )

    except Exception as e:
        # Tangkap semua error tidak terduga dari mesin Scikit-Learn
        logger.error(
            "Internal Prediction Error",
            extra={"extra_fields": {"request_id": request_id, "error_msg": str(e)}},
            exc_info=True # Ini akan mencetak Stack Trace lengkap di log!
        )
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan internal saat memproses prediksi. Silakan hubungi administrator."
        )
    
@router.get("/health")
def health_check(request: Request):
    """Cek apakah pipeline sudah ter-load dan metadata-nya."""
    pipeline = getattr(request.app.state, "pipeline", None)
    metadata = getattr(request.app.state, "metadata", {})
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline belum siap.")
        
    return {
        "status": "ready",
        "n_features": getattr(request.app.state, "n_features", None),
        "classes": getattr(request.app.state, "classes", None),
        "trained_at": metadata.get("trained_at"),
        "version": metadata.get("version"),
    }