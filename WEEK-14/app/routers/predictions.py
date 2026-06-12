import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.database import get_db
from app.models import AnalysisRequest, ScenarioResult
from app.schemas import AnalysisRequestCreate, AnalysisRequestResponse, AnalysisRequestUpdate

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis Requests"],
)

@router.get("/", response_model=list[AnalysisRequestResponse])
async def get_all_requests(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,  # Parameter untuk substring search
    status: Optional[str] = None,  # Parameter untuk exact match filter
    db: AsyncSession = Depends(get_db)
):
    """Mengambil semua Analysis Requests dengan dukungan filter dinamis."""
    # 1. Buat base query
    stmt = select(AnalysisRequest)

    # 2. Tambahkan filter substring secara dinamis jika parameter 'search' diisi
    if search:
        # Tanda % adalah wildcard di SQL (berarti: karakter apa saja sebelum dan sesudah)
        stmt = stmt.where(AnalysisRequest.title.ilike(f"%{search}%"))

    # 3. Tambahkan filter exact match jika parameter 'status' diisi
    if status:
        stmt = stmt.where(AnalysisRequest.status == status)

    stmt = stmt.offset(skip).limit(limit)
    
    # Eksekusi database di-await
    result = await db.execute(stmt)
    requests = result.scalars().all()
    return requests

# ── GET BY ID (ASYNC) ────────────────────────────────────────────
@router.get("/{request_id}", response_model=AnalysisRequestResponse)
async def get_request(request_id: str, db: AsyncSession = Depends(get_db)):
    """Mengambil satu request berdasarkan UUID."""
    try:
        uid = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Format UUID tidak valid")

    # Pencarian database di-await
    db_request = await db.get(AnalysisRequest, uid)
    if not db_request:
        raise HTTPException(status_code=404, detail="Analysis Request tidak ditemukan")
    return db_request

# ── POST CREATE (ASYNC) ──────────────────────────────────────────
@router.post("/", response_model=AnalysisRequestResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis_request(payload: AnalysisRequestCreate, db: AsyncSession = Depends(get_db)):
    """Membuat Request Induk sekaligus Data Skenario Anaknya."""
    db_request = AnalysisRequest(
        title=payload.title,
        parameters=payload.parameters,
        status=payload.status
    )
    
    db.add(db_request)
    await db.flush() # Eksekusi INSERT ke DB untuk mendaptkan UUID induk, tapi belum di-commit!

    # Memasukkan skenario anak
    for sc in payload.scenarios:
        db_scenario = ScenarioResult(
            request_id=db_request.id, # Ambil ID hasil flush di atas
            scenario_type=sc.scenario_type,
            risk_score=sc.risk_score,
            output_data=sc.output_data,
            efficiency_index=sc.efficiency_index,
            model_version=sc.model_version
        )
        db.add(db_scenario)

    return db_request

# ── PATCH update ASYNC ──────────────────────────────────────────────────
@router.patch("/{request_id}", response_model=AnalysisRequestResponse, status_code=status.HTTP_200_OK)
async def update_request(
    request_id: str, 
    payload: AnalysisRequestUpdate, 
    db: AsyncSession = Depends(get_db)
):
    """
    Melakukan update parsial (PATCH) pada title dan/atau status dari sebuah Analysis Request.
    """
    # 1. Validasi UUID (Skenario 422)
    try:
        uid = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Format UUID tidak valid")

    # 2. Cari data di database (Skenario 404)
    db_request = await db.get(AnalysisRequest, uid)
    if not db_request:
        raise HTTPException(status_code=404, detail="Analysis Request tidak ditemukan")

    # 3. Ekstrak hanya field yang dikirim oleh client
    # exclude_unset=True memastikan field yang bernilai 'None' (karena tidak dikirim) diabaikan
    update_data = payload.model_dump(exclude_unset=True)

    # 4. Terapkan perubahan ke objek ORM secara dinamis
    for key, value in update_data.items():
        setattr(db_request, key, value)

    # Return data terupdate (Skenario 200)
    return db_request

# ── DELETE (ASYNC) ──────────────────────────────────────────────────
@router.delete("/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_request(request_id: str, db: AsyncSession = Depends(get_db)):
    """Menghapus request dan otomatis menyapu bersih data anaknya (Cascade)."""
    try:
        uid = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Format UUID tidak valid")

    db_request = await db.get(AnalysisRequest, uid)
    if not db_request:
        raise HTTPException(status_code=404, detail="Analysis Request tidak ditemukan")

    await db.delete(db_request)
    return None

    