"""
#app/services/analysis_services.py
Service memvalidasi logika, memeriksa ketersediaan data, melempar HTTP Exceptions, dan mengatur commit.
"""
import uuid
import asyncio, logging
from app.database import SessionLocalAsync
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.analysis_repo import AnalysisRepository
from app.schemas.schemas import AnalysisRequestCreate, AnalysisRequestUpdate
from app.exceptions import NotFoundException, ValidationException
from app.core.logging_config import setup_global_logging

# Konfigurasi Logging
logger = logging.getLogger(__name__)

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
    async def run_heavy_ml_computation(request_id: uuid.UUID):
        """Fungsi yang akan berjalan di latar belakang tanpa memblokir API"""
        logger.info(f"Memulai komputasi ML latar belakang untuk ID: {request_id}")

        # Buka koneksi database baru khusus untuk proses latar belakang
        async with SessionLocalAsync() as db:
            try:
                # 1. Ubah status menjadi 'processing'
                obj = await AnalysisRepository.get_by_id(db, request_id)
                if obj:
                    await AnalysisRepository.update(db, obj, {"status": "processing"})
                    await db.commit()

                # 2. SIMULASI PROSES BERAT (Misal: Load model, KNN Imputer, Prediksi)
                # gunakan sleep 15 detik sebagai simulasi
                await asyncio.sleep(15)

                # 3. Proses selesai, ubah status menjadi 'completed'
                # disini juga terjadi penyimpanan prediksi
                obj_refresh = await AnalysisRepository.get_by_id(db, request_id)
                if obj_refresh:
                    await AnalysisRepository.update(db, obj_refresh, {"status": "completed"})
                    await db.commit()

                logger.info(f"Selesai komputasi ML latar belakang untuk ID: {request_id}")
                    
            except Exception as e:
                # Jika komputasi gagal, catat statusnya sebagai 'failed'
                obj_fail = await AnalysisRepository.get_by_id(db, request_id)
                if obj_fail:
                    await AnalysisRepository.update(db, obj_fail, {"status": "failed"})
                    await db.commit()
                logger.error(f"Background task gagal untuk ID {request_id}: {str(e)}", exc_info=True)