import uuid
import logging
from fastapi import APIRouter, Depends, status

from app.schemas.input_schema import DecisionInput
from app.schemas.simulation_schema import SimulationOutput
from app.services.simulation_services import SimulationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/decision",
    tags=["Decision Support System"]
)

def get_simulation_service() -> SimulationService:
    return SimulationService()

@router.post("/analyze", response_model=SimulationOutput, status_code=status.HTTP_200_OK)
async def analyze_decision(
    payload: DecisionInput,
    service: SimulationService = Depends(get_simulation_service),
) -> SimulationOutput:
    """
    Menerima DecisionInput, menjalankan simulasi 3 skenario,
    dan mengembalikan SimulationOutput terstruktur.
    """
    request_id = uuid.uuid4()
    logger.info(
        f"Analisis keputusan untuk {payload.company_name} | ID: {request_id}",
        extra={
            "request_id": str(request_id),
            "company_name": payload.company_name,
            "market_condition": payload.market_condition.value,
        }
    )
    return service.run_simulation(payload)