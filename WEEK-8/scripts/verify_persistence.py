# week-10/scripts/verify_persistence.py
"""
Script untuk verifikasi bahwa data Redis survive container restart.
Jalankan setelah docker-compose up.
"""

import argparse
import logging
import sys
import time
from typing import Dict

import redis

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Konfigurasi Default
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6379
MAX_RETRIES = 5
RETRY_DELAY_SEC = 2

# SINGLE SOURCE OF TRUTH: Data test didefinisikan satu kali agar konsisten (DRY Principle)
TEST_DATA: Dict[str, str] = {
    "session:user_001": "active",
    "score:model_v1:request_100": "0.87",
    "counter:predictions_total": "42",
}


def get_redis_client(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> redis.Redis:
    """
    Buat koneksi ke Redis dengan mekanisme retry sederhana.

    Args:
        host (str): Hostname Redis. Default ke localhost.
        port (int): Port Redis. Default ke 6379.

    Returns:
        redis.Redis: Object client Redis yang siap digunakan.

    Raises:
        RuntimeError: Jika gagal terkoneksi setelah batas maksimal percobaan.
    """
    for attempt in range(MAX_RETRIES):
        try:
            client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
            )
            client.ping()  # Validasi koneksi (akan raise error jika gagal)
            logger.info("Berhasil terhubung ke Redis.")
            return client
        except redis.ConnectionError:
            logger.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES}: Redis belum siap, tunggu {RETRY_DELAY_SEC} detik..."
            )
            time.sleep(RETRY_DELAY_SEC)

    logger.error("Gagal terhubung ke Redis setelah semua percobaan.")
    raise RuntimeError("Tidak bisa connect ke Redis.")


def write_test_data(client: redis.Redis, data: Dict[str, str]) -> None:
    """
    Tulis data ke dalam Redis.

    Args:
        client (redis.Redis): Client Redis yang aktif.
        data (Dict[str, str]): Dictionary berisi pasangan key-value untuk disimpan.
    """
    try:
        for key, value in data.items():
            client.set(key, value)
        logger.info(f"Berhasil menulis {len(data)} keys ke Redis.")
    except redis.RedisError as e:
        logger.error(f"Gagal menulis data ke Redis: {e}")
        sys.exit(1)


def read_test_data(client: redis.Redis, expected_data: Dict[str, str]) -> None:
    """
    Baca dan verifikasi data dari Redis cocok dengan yang diharapkan.

    Args:
        client (redis.Redis): Client Redis yang aktif.
        expected_data (Dict[str, str]): Dictionary berisi pasangan key-value yang diharapkan.
    """
    all_ok = True
    for key, expected_value in expected_data.items():
        try:
            actual_value = client.get(key)
            if actual_value is None:
                logger.error(f"Key hilang: {key}")
                all_ok = False
            elif actual_value != expected_value:
                # Memastikan nilainya juga tidak berubah/corrupt
                logger.error(f"Mismatch pada {key}: expected '{expected_value}', got '{actual_value}'")
                all_ok = False
            else:
                logger.info(f"{key} = {actual_value}")
        except redis.RedisError as e:
            logger.error(f"Gagal membaca key {key} dari Redis: {e}")
            all_ok = False

    if all_ok:
        logger.info("SUCCESS: Semua data survive restart dan valid!")
    else:
        logger.error("FAILED: Ada data yang hilang atau tidak valid. Cek konfigurasi volume Docker Anda.")
        sys.exit(1)


def main() -> None:
    """Entry point utama untuk eksekusi script."""
    # Menggunakan argparse untuk CLI yang lebih rapi dan self-documenting
    parser = argparse.ArgumentParser(
        description="Verifikasi persistensi data Redis melintasi container restarts."
    )
    parser.add_argument(
        "mode",
        choices=["write", "read"],
        help="Mode operasi: 'write' untuk memasukkan data, 'read' untuk verifikasi."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Redis host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Redis port (default: {DEFAULT_PORT})"
    )

    args = parser.parse_args()

    try:
        client = get_redis_client(host=args.host, port=args.port)

        if args.mode == "write":
            write_test_data(client, TEST_DATA)
            print("\n" + "="*55)
            print("Langkah selanjutnya:")
            print("1. Restart Redis:  docker-compose restart redis")
            print(f"2. Verifikasi:     python {sys.argv[0]} read")
            print("="*55 + "\n")
        elif args.mode == "read":
            read_test_data(client, TEST_DATA)

    except KeyboardInterrupt:
        logger.info("Operasi dibatalkan oleh pengguna (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()