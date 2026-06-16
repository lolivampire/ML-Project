# app/middleware/request_id.py
import uuid, time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.context import request_id_context_var

logger = logging.getLogger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. Generate UUID baru untuk request ini
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # 2. Simpan UUID ke dalam memori asinkron (ContextVar)
        # Fungsi set() mengembalikan sebuah 'token' untuk proses pembersihan nanti
        token = request_id_context_var.set(request_id)
        
        # Mulai hitung waktu
        start_time = time.perf_counter()

        try:
            # 3. Lanjutkan request ke Router/Service
            response = await call_next(request)

            # Hitung durasi
            process_time = time.perf_counter() - start_time
            duration_ms = round(process_time * 1000, 2)

            # Tampilkan log wakut request selesai
            logger.info(f"{request.method} {request.url.path} completed | duration_ms={duration_ms}")
            
        except Exception:
            logger.error("Unhandled exception in request", exc_info=True)
            raise
            
        finally:
            # 5. Hapus ID dari memori setelah request selesai
            # Ini mencegah kebocoran memori (memory leak)
            request_id_context_var.reset(token)
        
        # Tambahkan X-Request-ID ke response header
        # supaya client bisa korelasi response dengan log.
        response.headers["X-Request-ID"] = request_id
        return response