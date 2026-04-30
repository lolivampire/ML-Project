# app/schemas/prediction.py
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    feature_1: float = Field(
        ...,
        description="Informative synthetic feature (standardized, typically -3.0 to 3.0)"
    )
    feature_2: float = Field(
        ...,
        description="Informative synthetic feature (standardized, typically -3.0 to 3.0)"
    )
    feature_3: float = Field(
        ...,
        description="Informative synthetic feature (standardized, typically -3.0 to 3.0)"
    )
    feature_4: float = Field(
        ...,
        description="Redundant synthetic feature derived from features 1-3"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feature_1": 1.5,
                    "feature_2": -0.5,
                    "feature_3": 2.1,
                    "feature_4": 0.8
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    prediction: int | float
    probability: float | None = None
    model_version: str