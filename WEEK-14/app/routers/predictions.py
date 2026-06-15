"""
#app/routers/predictions.py (controller layer)
meneruskan dari client (Pydantic) menuju Service.
"""
# app/routers/predictions.py
import logging
from typing import Optional
from fastapi import APIRouter, Depends, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.analysis_services import AnalysisService
from app.schemas.schemas import AnalysisRequestCreate, AnalysisRequestUpdate, AnalysisRequestResponse
from app.core.logging_config import setup_global_logging

# Konfigurasi Logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.get("/", response_model=list[AnalysisRequestResponse])
async def list_predictions(
    skip: int = 0, limit: int = 100, search: Optional[str] = None, 
    status: Optional[str] = None, db: AsyncSession = Depends(get_db)
):
    return await AnalysisService.get_all(db, skip, limit, search, status)

@router.post("/", response_model=AnalysisRequestResponse, status_code=status.HTTP_201_CREATED)
async def create_prediction(
    payload: AnalysisRequestCreate, 
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):  
    # Catat aktivitas
    logger.info(f"Menerima request analisis baru: '{payload.title}'")

    result = await AnalysisService.create(db, payload)
    # Jalankan background task fire-and-forget
    background_tasks.add_task(AnalysisService.run_heavy_ml_computation, result.id)
    return result

@router.get("/{request_id}", response_model=AnalysisRequestResponse)
async def get_prediction(
    request_id: str,
    db: AsyncSession = Depends(get_db)
    ):
    return await AnalysisService.get_by_id(db, request_id)

@router.patch("/{request_id}", response_model=AnalysisRequestResponse)
async def update_analysis(
    request_id: str, 
    payload: AnalysisRequestUpdate, 
    db: AsyncSession = Depends(get_db)
    ):
    return await AnalysisService.update(db, request_id, payload)

@router.delete("/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    request_id: str, 
    db: AsyncSession = Depends(get_db)
    ):
    await AnalysisService.delete(db, request_id)