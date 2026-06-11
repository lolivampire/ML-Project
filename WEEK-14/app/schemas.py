from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any

# --- SCENARIO SCHEMAS ---
class ScenarioResultBase(BaseModel):
    scenario_type: str = Field(..., description="optimistic, pessimistic, or base")
    risk_score: Optional[float] = None
    output_data: Optional[Dict[str, Any]] = None
    efficiency_index: Optional[float] = None
    model_version: Optional[str] = None

class ScenarioResultCreate(ScenarioResultBase):
    pass # Hanya data yang diisi user saat POST

class ScenarioResultResponse(ScenarioResultBase):
    id: uuid.UUID
    request_id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True) # Wajib agar Pydantic bisa membaca ORM Object

class AnalysisRequestUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None

# --- ANALYSIS REQUEST SCHEMAS ---
class AnalysisRequestBase(BaseModel):
    title: str
    parameters: Optional[Dict[str, Any]] = None
    status: str = "pending"

class AnalysisRequestCreate(AnalysisRequestBase):
    scenarios: List[ScenarioResultCreate] = [] # User bisa membuat request sekaligus beserta skenarionya

class AnalysisRequestResponse(AnalysisRequestBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    scenarios: List[ScenarioResultResponse] = [] # Eager loading otomatis di-convert ke JSON

    model_config = ConfigDict(from_attributes=True)