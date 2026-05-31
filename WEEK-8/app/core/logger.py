# app/core/logger.py
import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True

class JSONFormatter(logging.Formatter):
    def __init__(self, environment: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.environment = environment
        
        # Daftar field bawaan internal Python/Uvicorn yang HARUS DIBUANG dari 'extra'
        self.ignored_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process', 'request_id'
        }

    def format(self, record: logging.LogRecord) -> str:
        # 1. Buat struktur yang ringkas dan bersih
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self.environment,
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }

        # 2. Hanya masukkan field yang BENAR-BENAR dikirim lewat extra={...}
        for key, value in record.__dict__.items():
            if key not in self.ignored_fields and not key.startswith('_'):
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)

def get_json_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    app_env = os.getenv("APP_ENV", "development")
    # Baca variabel DEBUG dari env (Ubah string "true"/"false" menjadi boolean Python)
    is_debug = os.getenv("DEBUG", "true").lower() == "true"

    logger.setLevel(logging.DEBUG if is_debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if is_debug else logging.INFO)
    handler.addFilter(RequestIDFilter())

    # SINKRONISASI: Jika DEBUG=true, pakai teks biasa. Jika false, pakai JSON bersih!
    if is_debug:
        fmt = "%(asctime)s | %(levelname)-8s | [%(request_id)s] | %(name)s | %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = JSONFormatter(environment=app_env)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger