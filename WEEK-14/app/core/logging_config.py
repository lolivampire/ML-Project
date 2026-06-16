# app/core/logging_config.py
"""
konfigurai logging
"""
import logging
import sys

from app.core.context import request_id_context_var

class RequestIDFilter(logging.Filter):
    """Filter untuk menyuntikkan request_id dari ContextVar ke setiap log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        # Menambahkan atribut kustom 'request_id' ke objek log
        record.request_id = request_id_context_var.get()
        return True

def setup_global_logging(level: str = "INFO") -> None:
    """
    Konfigurasi logger standar untuk seluruh aplikasi.
    """
    # Root logger — semua logger di seluruh aplikasi inherit dari sini
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        return

    root_logger.setLevel(logging.DEBUG)  # tangkap semua

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIDFilter())
    root_logger.addHandler(handler)

    # Matikan noise dari library external
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)