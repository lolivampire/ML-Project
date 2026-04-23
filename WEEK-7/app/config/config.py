"""
config.py
Satu tempat untuk semua konfigurasi aplikasi.
Semua env vars dibaca di sini, bukan di-scatter ke seluruh kode.
"""

import os
from dotenv import load_dotenv

# load_dotenv() membaca file .env dan menyimpannya di os.environ
# override=False artinya jika env var sudah di set oleh sistem
# jangan timpa, penting untuk production
load_dotenv(override=False)

print("Paket dotenv berhasil diimpor dan dijalankan!")

def get_required(key: str) -> str:
    """
    Baca env var yang wajib ada. Crash lebih baik daripada silent error.
    """
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"Env var {key} harus diisi!")
    return value

def get_optional(key: str, default: str="") -> str:
    """
    Baca env var opsional dengan nilai default
    """
    return os.getenv(key, default)


# Konfigurasi database
DB_HOST : str = get_optional("DB_HOST", "localhost")
DB_PORT : str = int(get_optional("DB_PORT", "5432")) #cast ke int
DB_NAME : str = get_required("DB_NAME")
DB_PASSWORD : str = get_required("DB_PASSWORD")

API_KEY : str = get_required("API_KEY")
SECRET_KEY : str = get_required("SECRET_KEY")

DEBUG : bool = get_optional("DEBUG", "False").lower() == "true" #parse ke boolean
PORT : int = int(get_optional("PORT", "8000")) #cast ke int