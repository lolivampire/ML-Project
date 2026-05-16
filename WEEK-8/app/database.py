# app/database.py
"""
Modul untuk mengelola koneksi database (Redis).
"""

import logging
import time
from typing import Optional

import redis

# Konfigurasi logger khusus untuk modul ini
logger = logging.getLogger(__name__)


def create_redis_client(
    host: str, 
    port: int, 
    password: Optional[str] = None, 
    max_retries: int = 5
) -> redis.Redis:
    """
    Buat Redis client dengan mekanisme exponential backoff.
    
    Kenapa eksponensial? Kalau Redis sedang recover, membanjirinya 
    dengan request konstan (misal tiap 0.1 detik) bisa memperburuk keadaan. 
    Backoff (jeda yang memanjang: 1s, 2s, 4s, 8s...) memberi Redis waktu bernapas.

    Args:
        host (str): Hostname server Redis.
        port (int): Port server Redis.
        password (Optional[str]): Password untuk autentikasi (default: None).
        max_retries (int): Maksimal percobaan koneksi sebelum menyerah.

    Returns:
        redis.Redis: Object client Redis yang siap digunakan.

    Raises:
        RuntimeError: Jika gagal terkoneksi karena network/down setelah batas percobaan.
        ValueError: Jika terjadi kegagalan autentikasi (password salah/ditolak).
    """
    logger.info(f"Mencoba terhubung ke Redis di {host}:{port}...")
    
    # Inisiasi client
    # decode_responses=True agar hasil GET/HGET berupa string Python (bukan bytes)
    client = redis.Redis(
        host=host, 
        port=port, 
        password=password, 
        decode_responses=True
    )
    
    for attempt in range(max_retries):
        try:
            client.ping()  # Mengeksekusi command PING untuk memvalidasi koneksi & auth
            logger.info(f" Berhasil terhubung ke Redis pada percobaan ke-{attempt + 1}")
            return client
            
        except redis.AuthenticationError as e:
            # FAIL-FAST: Jika password salah, retry tidak akan ada gunanya.
            # Langsung hentikan proses agar developer segera memperbaiki .env
            logger.error(" Gagal terhubung ke Redis: Autentikasi ditolak.")
            raise ValueError("Password Redis salah atau tidak valid. Cek REDIS_PASSWORD Anda.") from e
            
        except redis.ConnectionError as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            
            if attempt == max_retries - 1:
                # Habis semua percobaan -> raise error, cegah aplikasi berjalan tanpa database
                logger.error(" Semua percobaan koneksi Redis habis. Menyerah.")
                raise RuntimeError(f"Gagal connect ke Redis setelah {max_retries} percobaan.") from e
            
            logger.warning(
                f" Redis belum siap (Percobaan {attempt + 1}/{max_retries}). "
                f"Mencoba lagi dalam {wait_time} detik..."
            )
            time.sleep(wait_time)

    # Fallback pengaman (secara logis tidak akan pernah tercapai karena raise di dalam loop)
    raise RuntimeError("Terjadi kesalahan sistematis saat memuat client Redis.")