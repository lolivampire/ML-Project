# app/schemas/prediction.py
from pydantic import BaseModel, Field, field_validator

class PredictionRequest(BaseModel):    
    features: list[float] = Field(
        ...,
        description="5 numerical features untuk prediksi",
        min_length=5,
        max_length=5,
    )

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: list[float]) -> list[float]:
        import math
        for i, f in enumerate(v):
            if not math.isfinite(f):
                raise ValueError(f"Feature index {i} tidak valid: {f}")
        return v
    
    @field_validator("features")
    @classmethod
    def features_must_be_positive(cls, v: list[float]) -> list[float]:
        for i, f in enumerate(v):
            if f < 0:
                raise ValueError(f"Feature index {i} tidak valid: {f}, feature harus bernilai positif.")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [0.5, 1.2, 3.1, 0.8, 2.4]
                }
            ]
        }
    }
    
    def to_feature_list(self) -> list[float]:
        """Mengembalikan data features dalam bentuk list murni Python."""
        return self.features

class PredictionResponse(BaseModel):
    id: int
    prediction: int
    probability: float
    risk_tier: str
    features: list[float]
    created_at: str