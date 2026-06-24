import uuid
import logging
from fastapi import APIRouter, Depends, status
from typing import Dict, Any

from app.schemas.input_schema import DecisionInput, DecisionResponse
from app.schemas.simulation_schema import SimulationOutput
from app.services.simulation_services import SimulationService
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/decision",
    tags=["Decision Support System"]
)

def get_simulation_service() -> SimulationService:
    return SimulationService()

def get_recommendation_service() -> RecommendationService:
    return RecommendationService()

@router.post("/analyze", response_model=DecisionResponse, status_code=status.HTTP_200_OK)
async def analyze_decision(
    payload: DecisionInput,
    sim_svc: SimulationService = Depends(get_simulation_service),
    rec_svc: RecommendationService = Depends(get_recommendation_service),
) -> DecisionResponse:
    """
    Menerima DecisionInput, menjalankan simulasi 3 skenario,
    dan mengembalikan SimulationOutput terstruktur.
    """
    request_id = uuid.uuid4()
    
    # Observabilitas terpusat
    logger.info(
        f"Menerima request analisis keputusan untuk {payload.company_name} | ID: {request_id}",
        extra={
            "request_id": str(request_id),
            "company_name": payload.company_name,
            "budget": payload.budget,
            "risk_appetite": payload.risk_appetite.value,
        }
    )

    # Orkestrasi Service Layer berantai
    simulation_output = sim_svc.run_simulation(payload)
    recommendation_output = rec_svc.recommend(simulation_output, payload)

    # Pengembalian respons absolut
    return {
        "request_id": str(request_id),
        "status": "completed",
        "data": recommendation_output
    }