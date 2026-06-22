# app/routers/decision_router.py
import uuid
import logging
from typing import Dict, Any
from fastapi import APIRouter, status, BackgroundTasks

from app.schemas.input_schema import DecisionSimulationRequest, DecisionInput

# Inisialisasi logger standar terpusat
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/decision",
    tags=["Decision Support System"]
)

@router.post("/simulate", status_code=status.HTTP_202_ACCEPTED)
async def trigger_decision_simulation(
    payload: DecisionSimulationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Memicu proses simulasi analitik pendukung keputusan berbasis latar belakang.
    
    Validasi Pydantic berjalan secara absolut sebelum memasuki *controller* ini.
    """
    logger.info(f"Menerima analisis keputusan: '{payload.title}' dengan {len(payload.scenarios)} skenario.")
    
    # Rencana D02: Panggil simulation_service secara asinkron
    # background_tasks.add_task(SimulationService.run, payload)
    
    return {
        "success": True,
        "message": "Simulasi keputusan berhasil divalidasi dan masuk antrean pemrosesan.",
        "details": {
            "title": payload.title,
            "scenarios_evaluated": [s.scenario_name for s in payload.scenarios]
        }
    }

@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_decision(payload: DecisionInput) -> Dict[str, Any]:
    """
    Menerima dan memvalidasi data profil entitas untuk analisis keputusan tingkat tinggi.
    
    Menghasilkan UUID unik untuk melacak seluruh siklus hidup *request* melalui sistem 
    observabilitas terdistribusi.
    """
    request_id = uuid.uuid4()
    
    logger.info(
        f"Menerima request analisis keputusan untuk {payload.company_name} | ID: {request_id}",
        extra={
            "request_id": str(request_id),
            "company_name": payload.company_name,
            "budget": payload.budget,
            "growth_target": payload.growth_target,
            "risk_appetite": payload.risk_appetite.value,
            "market_condition": payload.market_condition.value,
            "time_horizon": payload.time_horizon.value,
            "industry_sector": payload.industry_sector.value
        }
    )

    return {
        "request_id": str(request_id),
        "status": "accepted",
        "validated_input": payload.model_dump()
    }