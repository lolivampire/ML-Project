# app/schemas/prediction.py
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    # Sesuaikan field ini dengan fitur yang dipakai model Week 4 kamu
    # Ini contoh generic — ganti dengan fitur asli
    feature_1: float = Field(..., description="Fitur pertama")
    feature_2: float = Field(..., description="Fitur kedua")
    feature_3: float = Field(..., description="Fitur ketiga")
    feature_4: float = Field(..., description="Fitur keempat")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"feature_1": 1.5, "feature_2": 0.3, "feature_3": 2.1, "feature_4": -1.2}
            ]
        }
    }

class PredictionResponse(BaseModel):
    prediction: int | float
    probability: float | None = None
    model_version: str