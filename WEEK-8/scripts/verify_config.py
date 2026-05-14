"""
Script verifikasi konfigurasi.
Tujuan: Memastikan semua variabel environment ter-load dengan benar
tanpa mengekspos kredensial rahasia ke log.
"""

import sys
from pathlib import Path

# Menambahkan root folder (week-10) ke dalam sys.path
# agar bisa melakukan 'import config' jika dijalankan dari dalam folder scripts/
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))


def verify_configuration() -> None:
    print("Memulai verifikasi konfigurasi...\n")
    
    try:
        # Pydantic akan memvalidasi variabel saat config di-import.
        # Jika variabel wajib (seperti API_KEY) hilang, ini akan me-raise Exception.
        from app.config import settings
    except Exception as e:
        print(" Gagal memuat konfigurasi!")
        print(f"API_KEY: MISSING")
        print(f"Detail Pydantic Error: {e}")
        sys.exit(1)

    print("=== Konfigurasi Aktif ===")
    
    # Mengonversi object settings Pydantic menjadi dictionary
    config_dict = settings.model_dump()
    all_valid = True

    for key, value in config_dict.items():
        # Menyembunyikan value asli dari api_key
        if key == "api_key":
            if value: # Jika value ada (tidak string kosong)
                print(f"{key}: SET")
            else:
                all_valid = False
        else:
            # Print semua config lainnya dengan valuenya
            print(f"{key}: {value}")

    print("\n" + "="*25)
    
    # Memastikan script exit dengan code 1 jika API_KEY berisi string kosong
    # (Pydantic mencegah ketiadaan key, tapi bisa jadi nilainya string kosong "")
    if not all_valid:
        print("Verifikasi gagal: Ada variabel wajib yang kosong.")
        sys.exit(1)
        
    print("Verifikasi berhasil. Semua konfigurasi valid dan aman.")


if __name__ == "__main__":
    verify_configuration()