"""
week-06/app/core/logger.py
"""

import logging
import json
from datetime import datetime, timezone
import sys

class CustomJSONFormatter(logging.Formatter):
    """
    Formatter khusus untuk mengubah setiap log record menjadi format JSON.
    """
    def format(self, record: logging.LogRecord) -> str:
        # 1. Buat struktur dasar JSON
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # ISO 8601, selalu UTC
            "level": record.levelname,           # "INFO", "ERROR", dll
            "logger": record.name,               # nama logger (biasanya nama modul)
            "message": record.getMessage(),      # pesan log yang sebenarnya
            "module": record.module,             # nama file tanpa .py
            "line": record.lineno,               # nomor baris di source code
        }

        # 2. Ambil data tambahan (extra_fields) jika kita menyuntikkannya
        # Ini akan menangkap request_id, latency, input_features, dll.
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        # 3. Tangkap error (exception traceback) jika ada (Log Level: ERROR)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # 4. Ubah dictionary Python menjadi string JSON
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """
    Fungsi untuk mengambil instance logger yang sudah di-setting ke JSON.
    """
    logger = logging.getLogger(name)
    
    # Mencegah duplikasi log jika fungsi ini dipanggil berkali-kali
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Cetak ke console (terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomJSONFormatter())
        
        logger.addHandler(console_handler)
        
        # Mencegah log ganda dari root logger
        logger.propagate = False 

    return logger