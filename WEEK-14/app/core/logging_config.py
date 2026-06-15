# app/core/logging_config.py
"""
konfigurai logging
"""
import logging
import sys

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
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Matikan noise dari library external
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)