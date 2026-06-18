"""
#app/services/analysis_services.py
Service memvalidasi logika, memeriksa ketersediaan data, melempar HTTP Exceptions, dan mengatur commit.
"""
import asyncio
import logging
import uuid
import random

# --- Tambahan Import OTel ---
from opentelemetry import trace

from app.database import SessionLocalAsync
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.analysis_repo import AnalysisRepository
from app.schemas.schemas import AnalysisRequestCreate, AnalysisRequestUpdate
from app.exceptions import NotFoundException, ValidationException
from app.core.logging_config import setup_global_logging
from app.metrics import PREDICTIONS_TOTAL, PREDICTION_LATENCY

# Konfigurasi Logging
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class AnalysisService:
    @staticmethod
    def _validate_uuid(id_str: str) -> uuid.UUID:
        try:
            return uuid.UUID(id_str)
        except ValueError:
            raise ValidationException("Format UUID tidak valid")

    @staticmethod
    async def get_all(db: AsyncSession, skip: int = 0, limit: int = 100, search: str = None, status: str = None):
        return await AnalysisRepository.get_all(db, skip, limit, search, status)

    @staticmethod
    async def get_by_id(db: AsyncSession, request_id: str):
        uid = AnalysisService._validate_uuid(request_id)
        obj = await AnalysisRepository.get_by_id(db, uid)
        if not obj:
            raise NotFoundException("Analysis Request", request_id)
        return obj

    @staticmethod
    async def create(db: AsyncSession, payload: AnalysisRequestCreate):
        # Pisahkan data induk dan anak
        payload_dict = payload.model_dump(exclude={"scenarios"})
        scenarios_dict = [sc.model_dump() for sc in payload.scenarios]
        
        # Insert lewat repo
        result = await AnalysisRepository.create(db, payload_dict, scenarios_dict)
        await db.commit() # Commit dikelola di Service Layer
        return result

    @staticmethod
    async def update(db: AsyncSession, request_id: str, payload: AnalysisRequestUpdate):
        obj = await AnalysisService.get_by_id(db, request_id) # Reuse method untuk validasi 404
        update_data = payload.model_dump(exclude_unset=True)
        
        result = await AnalysisRepository.update(db, obj, update_data)
        await db.commit()
        return result

    @staticmethod
    async def delete(db: AsyncSession, request_id: str):
        obj = await AnalysisService.get_by_id(db, request_id)
        await AnalysisRepository.delete(db, obj)
        await db.commit()

    @staticmethod
    async def analyze(features: list[float]) -> dict:
        # Span 1: Validasi
        with tracer.start_as_current_span("validate_features") as span:
            await asyncio.sleep(0.05) #mensimulasikan latency,  span duration harus mencerminkan latency asli operasi itu
            count = len(features)
            span.set_attribute("features.count", count)
            
            if count == 0:
                span.set_attribute("error", True)
                span.add_event("Validasi gagal: list features kosong")
                raise ValueError("Features tidak boleh kosong")
            
            span.add_event("Validasi sukses")

        # Span 2: Komputasi
        with tracer.start_as_current_span("compute_stats") as span:
            count = len(features)
            mean_val = sum(features) / count
            max_val = max(features)
            
            span.set_attribute("stats.mean", mean_val)
            span.set_attribute("stats.max", max_val)
            span.add_event(f"Komputasi selesai. Max: {max_val}")
        
        return {"mean": mean_val, "max": max_val}

    @staticmethod
    async def run_heavy_ml_computation(request_id: uuid.UUID):
        """Fungsi yang akan berjalan di latar belakang tanpa memblokir API"""
        current_span = trace.get_current_span()
        current_span.set_attribute("task.request_id", str(request_id))
        current_span.set_attribute("task.type", "ml_computation")
        logger.info(f"Memulai komputasi ML latar belakang untuk ID: {request_id}")

        # Buka koneksi database baru khusus untuk proses latar belakang
        async with SessionLocalAsync() as db:
            try:
                # 1. Ubah status menjadi 'processing'
                obj = await AnalysisRepository.get_by_id(db, request_id)
                if obj:
                    await AnalysisRepository.update(db, obj, {"status": "processing"})
                    await db.commit()

                # Buat data dummy untuk dilempar ke fungsi analyze
                dummy_features = [random.uniform(10.0, 50.0) for _ in range(100)]

                # 2. SIMULASI PROSES BERAT (Misal: Load model, KNN Imputer, Prediksi)
                # Bungkus komputasi berat dengan Stopwatch Histogram
                with PREDICTION_LATENCY.labels(model_name="house_pricing_v1").time():
                    # Panggil fungsi yang dibungkus dengan OTel (Metrik Terperinci)
                    stats = await AnalysisService.analyze(dummy_features)
                    logger.info(f"Statistik komputasi: {stats}")

                    # SIMULASI PROSES BERAT
                    await asyncio.sleep(5)

                # 3. Proses selesai, ubah status menjadi 'completed'
                # disini juga terjadi penyimpanan prediksi
                obj_refresh = await AnalysisRepository.get_by_id(db, request_id)
                if obj_refresh:
                    await AnalysisRepository.update(db, obj_refresh, {"status": "completed"})
                    await db.commit()

                #catat metrics counters sukses
                PREDICTIONS_TOTAL.labels(model_name="house_pricing_v1", status="completed").inc()
                logger.info(f"Selesai komputasi ML latar belakang untuk ID: {request_id}")
                    
            except Exception as e:
                # Jika komputasi gagal, catat statusnya sebagai 'failed'
                obj_fail = await AnalysisRepository.get_by_id(db, request_id)
                if obj_fail:
                    await AnalysisRepository.update(db, obj_fail, {"status": "failed"})
                    await db.commit()
                PREDICTIONS_TOTAL.labels(model_name="house_pricing_v1", status="failed").inc()
                logger.error(f"Background task gagal untuk ID {request_id}: {str(e)}", exc_info=True)