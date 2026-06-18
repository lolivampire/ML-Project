#app/middleware.py
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace

from app.metrics import HTTP_REQUEST_DURATION

logger = logging.getLogger(__name__)

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
         # Skip instrumentasi untuk /metrics itu sendiri
        if request.url.path == "/metrics":
            return await call_next(request)
        
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            process_time = time.perf_counter() - start_time
            
            # Observe metrik ke Prometheus
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code
            ).observe(process_time)
            
            # Cetak ke log (tetap dipertahankan untuk trace visual)
            duration_ms = round(process_time * 1000, 2)
            logger.info(f"{request.method} {request.url.path} completed | duration_ms={duration_ms}")
            
            # Ambil Trace ID dari OTel dan pasang sebagai Header Response
            span = trace.get_current_span()
            if span and span.is_recording():
                trace_id = format(span.get_span_context().trace_id, "032x")
                response.headers["X-Trace-ID"] = trace_id
                
            return response
        
        except Exception as e:
            # Pastikan metrik tetap tercatat meskipun request berakhir error (500)
            process_time = time.perf_counter() - start_time
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                path=request.url.path,
                status_code=500
            ).observe(process_time)
            raise e
        
        # Blok 'finally' dan 'reset(token)' dihapus karena manajemen memori
        # kini sepenuhnya ditangani oleh arsitektur internal OpenTelemetry.
        # finally:
        #     request_id_context_var.reset(token)