"""
cache_aside_demo.py
Week 18 - Day 01: Cache-Aside Pattern Implementation

Mendemonstrasikan pola Cache-Aside (Lazy Loading) menggunakan Redis Hash.
Aplikasi bertanggung jawab penuh untuk membaca/menulis ke cache dan database.
"""

from typing import Any
import redis

# ── DATABASE PALSU (MOCK DB) ─────────────────────────────────────────────
# Representasi data persistent seperti yang ada di MySQL/PostgreSQL
FAKE_DATABASE: dict[str, dict[str, str]] = {
    "101": {"user_id": "101", "name": "Tiara Basori", "email": "tiara@example.com"},
    "102": {"user_id": "102", "name": "Nakano Yotsuba", "email": "yotsuba@example.com"},
    "103": {"user_id": "103", "name": "Sandrone Guillotine", "email": "sandrone@example.com"},
}

# Konfigurasi TTL (Time To Live). 
# Untuk profil user yang jarang berubah, 300 detik (5 menit) adalah nilai yang ideal.
CACHE_TTL_SECONDS = 300


def get_user_profile(client: redis.Redis, user_id: str) -> dict[str, Any] | None:
    """Mengambil profil user menggunakan pola Cache-Aside.

    Args:
        client: Instance koneksi Redis.
        user_id: ID unik user yang ingin dicari.

    Returns:
        Dictionary profil user jika ditemukan, atau None jika tidak ada.
    """
    redis_key = f"user:{user_id}"

    # 1. Cek data di Redis Cache (Hash)
    cached_data = client.hgetall(redis_key)
    
    if cached_data:
        print(f"[CACHE HIT] Profil user {user_id} ditemukan di Redis.")
        return cached_data

    # 2. Jika MISS, ambil data dari Database Utama
    print(f"[CACHE MISS] Profil user {user_id} TIDAK ditemukan di Redis. Mencari di database...")
    user_data = FAKE_DATABASE.get(user_id)

    # 3. Jika di Database juga tidak ada, kembalikan None
    if not user_data:
        print(f"[NOT FOUND] User {user_id} tidak terdaftar di database.")
        return None

    # 4. Jika ditemukan di DB, simpan ke Redis dan atur TTL
    client.hset(redis_key, mapping=user_data)
    client.expire(redis_key, CACHE_TTL_SECONDS)
    print(f"[CACHE WRITE] Sukses menyimpan data user {user_id} ke Redis (TTL: {CACHE_TTL_SECONDS}s).")

    return user_data


def main() -> None:
    """Entry point utama aplikasi.
    
    Berfungsi untuk menginisialisasi client Redis, memvalidasi koneksi jaringan,
    menghapus state lama untuk pengujian, dan mendemonstrasikan skenario 
    Cache-Aside (Cache Miss, Cache Hit, serta penanganan data kosong).
    """
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    try:
        r.ping()
        print("Berhasil terhubung ke Redis server.")
    except redis.exceptions.ConnectionError:
        print("Error: Gagal terhubung ke Redis. Pastikan container Docker Anda berjalan.")
        return

    # Pembersihan key uji coba agar demo bersifat idempotent
    test_user_id = "101"
    invalid_user_id = "999"
    r.delete(f"user:{test_user_id}")

    print("\n--- PANGGILAN PERTAMA (Ekspektasi: CACHE MISS) ---")
    profil_1 = get_user_profile(r, test_user_id)
    print(f"Hasil: {profil_1}")

    print("\n--- PANGGILAN KEDUA (Ekspektasi: CACHE HIT) ---")
    profil_2 = get_user_profile(r, test_user_id)
    print(f"Hasil: {profil_2}")

    print("\n--- PANGGILAN KETIGA (Kasus: User Tidak Eksis) ---")
    profil_3 = get_user_profile(r, invalid_user_id)
    print(f"Hasil: {profil_3}")


if __name__ == "__main__":
    main()