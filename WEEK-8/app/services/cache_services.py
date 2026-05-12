# app/services/cache_service.py
import os
import redis

# Baca dari environment variable — bukan hardcode
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")  # fallback: localhost untuk dev tanpa Docker
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Buat connection pool — lebih efisien dari membuat koneksi baru tiap request
redis_client = redis.Redis(
    host=REDIS_HOST,       # ← di dalam Docker: "redis" (service name)
    port=REDIS_PORT,       # ← 6379
    decode_responses=True  # return str, bukan bytes
)

def set_cache(key: str, value: str, ttl: int = 300) -> None:
    """Simpan value ke Redis dengan TTL dalam detik."""
    redis_client.setex(key, ttl, value)

def get_cache(key: str) -> str | None:
    """Ambil value dari Redis. Return None jika tidak ada."""
    return redis_client.get(key)