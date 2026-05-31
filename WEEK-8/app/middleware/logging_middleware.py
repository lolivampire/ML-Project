import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Import logger dan context variabel dari core logger yang kita buat sebelumnya
from app.core.logger import get_json_logger, request_id_var

# Inisialisasi logger untuk module ini
logger = get_json_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware untuk melacak setiap HTTP request.
    Menginjeksi Request ID dan mencatat metode, path, status code, serta durasi.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate 8 karakter pertama UUID sebagai Request ID
        req_id = str(uuid.uuid4())[:8]
        
        # Set Request ID ke context variabel (otomatis terbaca oleh RequestIDFilter)
        token = request_id_var.set(req_id)
        
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        # Log saat request pertama kali masuk
        logger.info(f"Incoming request: {method} {path}", extra={
            "method": method,
            "path": path
        })
        
        try:
            # Lanjutkan pemrosesan request ke endpoint tujuan
            response = await call_next(request)
            
            # Hitung durasi dalam milidetik
            process_time = (time.time() - start_time) * 1000
            
            # Log saat request selesai dengan sukses (atau ditangani oleh exception handler bawaan)
            logger.info(
                f"Request completed: {method} {path}",
                extra={
                    "status_code": response.status_code,
                    "duration_ms": round(process_time, 2),
                    "method": method,
                    "path": path
                }
            )
            
            # Sisipkan Request ID ke header balasan agar klien mengetahuinya
            response.headers["X-Request-ID"] = req_id
            return response
            
        except Exception as e:
            # Log jika terjadi kegagalan tak terduga (Unhandled Exception)
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {method} {path} - {str(e)}",
                extra={
                    "status_code": 500,
                    "duration_ms": round(process_time, 2),
                    "method": method,
                    "path": path
                },
                exc_info=True
            )
            raise
        finally:
            # Sangat krusial: Bersihkan context agar ID tidak tumpang tindih dengan request lain
            request_id_var.reset(token)