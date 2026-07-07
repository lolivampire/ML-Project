"""
redis_demo.py
Week 18 - Day 01: Redis Fundamentals

Demonstrasi 4 struktur data dasar Redis, TTL, dan pengecekan eviction policy.
Skrip ini dirancang mandiri dan aman dijalankan berulang kali.
"""

import time
from typing import Optional
import redis


def cetak_judul(judul: str) -> None:
    """Helper untuk membuat batasan visual di output terminal."""
    print(f"\n=== {judul.upper()} ===")


def demo_string(client: redis.Redis) -> None:
    """String: Tipe data paling mendasar, menyimpan nilai tunggal (teks/angka)."""
    cetak_judul("Data Type: String")
    
    key = "user:1:name"
    
    # Reset data lama agar demonstrasi tetap akurat
    client.delete(key)
    
    client.set(key, "Tiara Basori")
    nama = client.get(key)
    print(f"Set & Get sukses. {key} -> {nama}")


def demo_hash(client: redis.Redis) -> None:
    """Hash: Cocok untuk menyimpan objek atau record field-value (seperti objek profil)."""
    cetak_judul("Data Type: Hash")
    
    key = "user:1"
    client.delete(key)
    
    payload = {"name": "Budi", "age": "25", "role": "engineer"}
    client.hset(key, mapping=payload)
    
    # Mengambil seluruh isi field di dalam hash
    data = client.hgetall(key)
    print(f"Isi data hash {key}: {data}")
    print(f"Mengambil field spesifik (role): {client.hget(key, 'role')}")


def demo_list(client: redis.Redis) -> None:
    """List: Urutan string berdasarkan waktu penyisipan. 
    Bisa digunakan untuk Queue (FIFO) dengan kombinasi LPUSH dan RPOP.
    """
    cetak_judul("Data Type: List (Queue Simulation)")
    
    key = "queue:jobs"
    client.delete(key)
    
    # Push data dari kiri (LPUSH)
    print("Memasukkan job_A, job_B, job_C ke antrean...")
    client.lpush(key, "job_A", "job_B", "job_C")
    
    # lrange menampilkan data dari indeks 0 sampai terakhir (-1)
    # Hasilnya akan terbalik [job_C, job_B, job_A] karena didorong dari kiri
    semua_job = client.lrange(key, 0, -1)
    print(f"Kondisi list saat ini: {semua_job}")
    
    # Simulasi memproses antrean dari kanan (RPOP) -> FIFO
    job_diproses = client.rpop(key)
    print(f"Memproses data paling awal masuk (FIFO): {job_diproses}")
    print(f"Sisa antrean: {client.lrange(key, 0, -1)}")


def demo_set(client: redis.Redis) -> None:
    """Set: Koleksi string yang unik tanpa urutan tertentu.
    Duplikasi data otomatis diabaikan.
    """
    cetak_judul("Data Type: Set")
    
    key = "online:users"
    client.delete(key)
    
    # user1 dimasukkan dua kali
    client.sadd(key, "user1", "user2", "user1")
    
    online_members = client.smembers(key)
    print(f"Anggota set unik (user1 hanya muncul sekali): {online_members}")
    
    # Cek keanggotaan (Is Member) dengan efisiensi O(1)
    is_online = client.sismember(key, "user1")
    print(f"Apakah user1 sedang online? {is_online}")


def demo_ttl(client: redis.Redis) -> None:
    """TTL (Time To Live): Mengatur batas waktu kedaluwarsa suatu key."""
    cetak_judul("Feature: TTL (Time To Live)")
    
    key = "session:abc123"
    client.delete(key)
    
    # Set data dengan expiry 3 detik
    client.set(key, "active", ex=3)
    
    sisa_waktu = client.ttl(key)
    print(f"Sisa waktu awal key {key}: {sisa_waktu} detik")
    
    # Simulasi menunggu data kedaluwarsa
    print("Menunggu 4 detik...")
    time.sleep(4)
    
    data_sesudah_delay = client.get(key)
    print(f"Mencoba mengambil data setelah delay: {data_sesudah_delay} (Nilai None berarti terhapus)")


def cek_eviction_policy(client: redis.Redis) -> None:
    """Membaca konfigurasi penanganan memori penuh (Eviction Policy) di level server."""
    cetak_judul("Server Config: Eviction Policy")
    
    # config_get mengembalikan dictionary, misal: {'maxmemory-policy': 'noeviction'}
    config: dict = client.config_get("maxmemory-policy")
    policy = config.get("maxmemory-policy")
    
    print(f"Kebijakan memori saat ini: {policy}")
    print("Catatan: Jika penuh, kebijakan ini menentukan apakah Redis akan menghapus data lama atau menolak data baru.")


def main() -> None:
    # Inisialisasi client dengan decode_responses agar output otomatis bertipe data string
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    try:
        r.ping()
        print("Koneksi ke Redis berhasil dibentuk.")
    except redis.exceptions.ConnectionError:
        print("Error: Tidak dapat terhubung ke Redis. Pastikan container 'redis-belajar' sudah berjalan.")
        return

    # Menjalankan seluruh modul demonstrasi
    demo_string(r)
    demo_hash(r)
    demo_list(r)
    demo_set(r)
    demo_ttl(r)
    cek_eviction_policy(r)


if __name__ == "__main__":
    main()