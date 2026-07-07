import json
import uuid
import logging
import hashlib
import asyncio  # Ditambahkan untuk handle fungsi sinkronous

from fastapi import APIRouter, Depends, HTTPException, status
from redis import asyncio as aioredis
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, get_redis
from app.models import CompanyModel
from app.schemas.input_schema import DecisionInput, DecisionResponse
from app.services.simulation_services import SimulationService
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/decision", tags=["Decision Support System"])

CACHE_TTL_SECONDS = 300
NEGATIVE_CACHE_TTL_SECONDS = 300
NEGATIVE_CACHE_MARKER = "NOT_FOUND"

def get_simulation_service() -> SimulationService:
    return SimulationService()

def get_recommendation_service() -> RecommendationService:
    return RecommendationService()

def _generate_deterministic_cache_key(payload: DecisionInput) -> str:
    clean_company_name = " ".join(payload.company_name.split()).lower()
    raw_key_string = (
        f"{clean_company_name}_"
        f"{payload.budget}_"
        f"{payload.growth_target}_"
        f"{payload.market_condition.value}_"
        f"{payload.risk_appetite.value}"
    )
    hashed_string = hashlib.md5(raw_key_string.encode()).hexdigest()
    return f"decision:simulation:{hashed_string}"

@router.post("/analyze", response_model=DecisionResponse, status_code=status.HTTP_200_OK)
async def analyze_decision(
    payload: DecisionInput,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    sim_svc: SimulationService = Depends(get_simulation_service),
    rec_svc: RecommendationService = Depends(get_recommendation_service),
):
    request_id = uuid.uuid4()
    cache_key = _generate_deterministic_cache_key(payload)
    normalized_name = payload.company_name.strip().lower()
    
    # Kunci khusus untuk menandai bahwa perusahaan memang tidak ada di DB fisik
    company_blacklist_key = f"decision:company_status:invalid:{normalized_name}"

    logger.info(
        f"[REQUEST START] ID: {request_id} | Company: {payload.company_name}",
        extra={"request_id": str(request_id), "company_name": payload.company_name}
    )

    # ── 1. CACHE READ (Hasil Simulasi) ────────────────────────────────────────
    try:
        cached_data = await redis.get(cache_key)
        if cached_data:
            cached_string = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
            recommendation_data = json.loads(cached_string)
            logger.info(f"[CACHE HIT] Mengembalikan hasil dari Redis | Key: {cache_key}")
            return DecisionResponse(request_id=str(request_id), status="completed", data=recommendation_data)
    except json.JSONDecodeError:
        logger.error(f"[DATA CORRUPTION] Format JSON di cache rusak. Bypass ke database.")
    except Exception as redis_err:
        logger.error(f"[REDIS UNAVAILABLE] Gagal membaca cache: {redis_err}. Bypass.")

    # ── 2. PROTEKSI NEGATIVE CACHE (Deteksi Spam Perusahaan Gaib) ─────────────
    # Diperiksa HANYA saat terjadi simulation cache miss untuk menghemat operasi I/O
    try:
        is_invalid_company = await redis.get(company_blacklist_key)
        if is_invalid_company:
            logger.warning(f"[CACHE HIT - NEGATIVE] Blokir request polimorfik untuk '{normalized_name}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Perusahaan '{payload.company_name}' tidak terdaftar di sistem kami."
            )
    except HTTPException:
        raise
    except Exception as redis_err:
        logger.error(f"[REDIS ERROR] Gagal memeriksa blacklist key: {redis_err}")

    # ── 3. VALIDASI DATABASE POSTGRESQL ───────────────────────────────────────
    logger.info(f"[DB QUERY] Memeriksa eksistensi entitas: {normalized_name}")
    stmt = select(CompanyModel).where(func.lower(CompanyModel.name) == normalized_name)
    result = await db.execute(stmt)
    company_record = result.scalars().first()

    if not company_record:
        logger.warning(f"[RECORD NOT FOUND] Entitas '{payload.company_name}' tidak ada di DB.")
        # Tulis ke key berbasis NAMA PERUSAHAAN, bukan parameter simulasi
        try:
            await redis.set(company_blacklist_key, NEGATIVE_CACHE_MARKER, ex=NEGATIVE_CACHE_TTL_SECONDS)
            logger.info(f"[CACHE WRITE - NEGATIVE] Menyimpan blacklist nama perusahaan: {company_blacklist_key}")
        except Exception as redis_err:
            logger.error(f"[REDIS ERROR] Gagal menulis negative cache: {redis_err}")
            
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Perusahaan '{payload.company_name}' tidak terdaftar di dalam sistem database kami."
        )

    # ── 4. PROSES KOMPUTASI (Non-blocking via Thread Pool) ────────────────────
    logger.info(f"[COMPUTATION START] Memproses simulasi untuk {payload.company_name}")
    
    # Menggunakan asyncio.to_thread agar fungsi sinkronous/CPU-heavy berjalan di thread terpisah
    simulation_output = await asyncio.to_thread(sim_svc.run_simulation, payload)
    recommendation_output = await asyncio.to_thread(rec_svc.recommend, simulation_output, payload)

    if hasattr(recommendation_output, "model_dump"):
        output_dict = recommendation_output.model_dump()
    else:
        output_dict = recommendation_output.dict()

    # ── 5. CACHE WRITE ────────────────────────────────────────────────────────
    try:
        cache_data_string = (
            recommendation_output.model_dump_json() 
            if hasattr(recommendation_output, "model_dump_json") 
            else json.dumps(output_dict)
        )
        await redis.set(cache_key, cache_data_string, ex=CACHE_TTL_SECONDS)
        logger.info(f"[CACHE WRITE] Hasil simulasi disimpan ke Redis | Key: {cache_key}")
    except Exception as redis_err:
        logger.error(f"[REDIS ERROR] Gagal menulis hasil simulasi ke Redis: {redis_err}")

    return DecisionResponse(request_id=str(request_id), status="completed", data=output_dict)