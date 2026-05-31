# app/routers/health.py
import shutil
import time
import redis
from fastapi import APIRouter, Response, status
from app.core.state import app_state
from app.core.logger import get_json_logger

# Import settings untuk mengambil versi aplikasi
from app.config import settings
from app.core.state import app_state
from app.core.logger import get_json_logger

logger = get_json_logger(__name__)
router = APIRouter(tags=["health"])

_start_time = time.time()


def _check_disk_space(min_free_mb: int = 100, path: str = "/") -> bool:
    """
    Fungsi internal untuk mengecek apakah sisa ruang disk 
    masih berada di atas ambang batas aman (min_free_mb).
    """
    try:
        # Mengambil informasi total, used, dan free disk (dalam Bytes)
        total, used, free = shutil.disk_usage(path)
        
        # Konversi Bytes ke Megabytes (MB)
        free_mb = free / (1024 * 1024)
        
        # Log peringatan jika kapasitas hampir habis
        if free_mb <= min_free_mb:
            logger.warning(
                "readiness_disk_check_failed", 
                extra={"free_mb": round(free_mb, 2), "path": path}
            )
            return False
            
        return True
    except Exception as e:
        logger.error("disk_check_error", extra={"error": str(e)}, exc_info=True)
        return False


@router.get("/health", summary="Liveness Probe")
async def health_check():
    """
    Liveness probe.
    Selalu 200 selama proses masih bisa menjawab request.
    """
    return {
        "status": "alive",
        "uptime_seconds": round(time.time() - _start_time, 2),
        "version": settings.app_version
    }


@router.get("/ready", summary="Readiness Probe")
async def readiness_check(response: Response):
    """
    Readiness probe.
    Kembalikan 503 jika app belum siap atau ada dependency kritis yang down/penuh.
    """
    # 1. Cek apakah ada error saat lifespan startup
    if not app_state.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "reason": app_state.startup_error or "startup in progress"
        }
        
    # 2. Cek apakah Redis masih terhubung (Runtime Error)
    redis_alive = False
    if app_state.redis_client:
        try:
            app_state.redis_client.ping()
            redis_alive = True
        except redis.RedisError:
            pass

    # 3. Kumpulkan semua hasil pengecekan (termasuk disk space)
    checks = {
        "model_loaded": app_state.model_service is not None,
        "database_connected": redis_alive,
        "disk_space_ok": _check_disk_space(min_free_mb=100)  # ← Memanggil pengecekan disk
    }
    
    # Jika ada satu pun check yang False → Kembalikan 503
    if not all(checks.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        failed = [k for k, v in checks.items() if not v]
        
        logger.warning(
            "readiness_check_failed", 
            extra={"failed_checks": failed, "all_checks": checks}
        )
        return {"status": "not_ready", "checks": checks}

    return {"status": "ready", "checks": checks}