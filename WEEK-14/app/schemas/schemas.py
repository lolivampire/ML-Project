from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any

# --- SCENARIO SCHEMAS ---
class ScenarioResultBase(BaseModel):
    # Field: Memberikan metadata dan batas dasar
    scenario_type: str = Field(..., description="optimistic, pessimistic, or base")
    risk_score: Optional[float] = None
    output_data: Optional[Dict[str, Any]] = None
    efficiency_index: Optional[float] = None
    model_version: Optional[str] = None

    # field_validator: Logika kustom untuk memvalidasi kolom tertentu
    @field_validator('scenario_type')
    @classmethod
    def validate_scenario_type(cls, value):
        allowed = ['optimistic', 'pessimistic', 'base']
        if value.lower() not in allowed:
            raise ValueError(f"Tipe skenario tidak valid. Harus salah satu dari: {allowed}")
        return value.lower()

    @field_validator('risk_score')
    @classmethod
    def validate_risk_score(cls, value):
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("risk_score tidak masuk akal. Nilai harus berada di antara 0.0 hingga 1.0")
        return value

class ScenarioResultCreate(ScenarioResultBase):
    pass

class ScenarioResultResponse(ScenarioResultBase):
    id: uuid.UUID
    request_id: uuid.UUID
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

# --- ANALYSIS REQUEST SCHEMAS ---
class AnalysisRequestBase(BaseModel):
    # Tambahkan batas minimal dan maksimal panjang karakter
    title: str = Field(..., min_length=5, max_length=150)
    parameters: Optional[Dict[str, Any]] = None
    status: str = "pending"

    @field_validator('status')
    @classmethod
    def validate_status(cls, value):
        allowed = ['pending', 'processing', 'completed', 'failed']
        if value.lower() not in allowed:
            raise ValueError(f"Status tidak dikenali. Gunakan: {allowed}")
        return value.lower()

class AnalysisRequestCreate(AnalysisRequestBase):
    scenarios: List[ScenarioResultCreate] = []

class AnalysisRequestUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=5, max_length=150)
    status: Optional[str] = None

    # Kita bisa me-reuse logika validasi status di sini juga
    @field_validator('status')
    @classmethod
    def validate_status_update(cls, value):
        if value is not None:
            allowed = ['pending', 'processing', 'completed', 'failed']
            if value.lower() not in allowed:
                raise ValueError(f"Status tidak dikenali. Gunakan: {allowed}")
            return value.lower()
        return value

class AnalysisRequestResponse(AnalysisRequestBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    scenarios: List[ScenarioResultResponse] = []
    model_config = ConfigDict(from_attributes=True)