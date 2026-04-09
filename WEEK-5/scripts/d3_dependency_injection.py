"""
w05d03_dependency_injection.py
Week 05 - Day 03: Dependency Injection Dasar

Demonstrasi tiga tipe dependency:
1. Config dependency (environment/settings)
2. Database session dependency (fake)
3. Auth dependency (header-based)
"""

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Annotated
import uuid

app = FastAPI(title="DI Demo")


# ─────────────────────────────────────────────
# 1. CONFIG DEPENDENCY
#    Simulasi: inject settings ke endpoint
# ─────────────────────────────────────────────

class AppConfig:
    """Konfigurasi aplikasi — di dunia nyata ini dari .env atau os.environ"""
    app_name: str = "ML Risk API"
    max_items: int = 100
    debug: bool = True


def get_config() -> AppConfig:
    """
    Dependency function untuk config.
    FastAPI akan memanggil ini setiap request dan hasilnya di-inject ke endpoint.
    """
    return AppConfig()


# ─────────────────────────────────────────────
# 2. DATABASE SESSION DEPENDENCY
#    Simulasi: inject "koneksi DB" ke endpoint
#    Pola ini PERSIS yang dipakai dengan SQLAlchemy nanti (Week 14)
# ─────────────────────────────────────────────

class FakeDatabase:
    """
    Simulasi database session.
    Di production: ini adalah SQLAlchemy Session atau connection pool.
    """
    def __init__(self):
        # Simulasi data tersimpan di "DB"
        self._store: dict[int, dict] = {
            1: {"id": 1, "name": "Alice", "score": 87.5},
            2: {"id": 2, "name": "Bob",   "score": 62.0},
        }

    def get_user(self, user_id: int) -> dict | None:
        return self._store.get(user_id)

    def close(self):
        # Di SQLAlchemy: session.close()
        pass


def get_db():
    """
    Dependency function untuk database session.

    Kenapa pakai 'yield' bukan 'return'?
    → yield memungkinkan cleanup code SETELAH endpoint selesai.
    → Ini pola Generator — baris setelah yield dijalankan saat request selesai.
    → Ekivalen dengan try/finally: pastikan session selalu ditutup.
    """
    db = FakeDatabase()
    try:
        yield db          # ← endpoint menerima 'db' di sini
    finally:
        db.close()        # ← ini jalan SETELAH endpoint selesai, apapun yang terjadi


# ─────────────────────────────────────────────
# 3. AUTH DEPENDENCY
#    Simulasi: cek API key dari header
# ─────────────────────────────────────────────

VALID_API_KEYS = {"secret-key-123", "another-valid-key"}


def get_current_user(x_api_key: Annotated[str | None, Header()] = None) -> dict:
    """
    Dependency function untuk autentikasi.

    Annotated[str | None, Header()] = cara FastAPI membaca header HTTP.
    Header "X-Api-Key" otomatis di-parse (FastAPI ubah '-' jadi '_').

    Jika key tidak valid → raise HTTPException → endpoint TIDAK jalan.
    Jika valid → return user info → di-inject ke endpoint.
    """
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    # Di production: decode JWT, query user dari DB, dst.
    return {"user_id": 99, "role": "admin", "key": x_api_key}

# ─────────────────────────────────────────────
# 4. REQUEST ID DEPENDENCY
# ─────────────────────────────────────────────

def get_request_id() -> str:
    """
    Dependency untuk generate UUID unik per request.
    Cocok untuk logging atau tracing (mengetahui jejak error di production).
    """
    return str(uuid.uuid4())


def get_admin_user(current_user: Annotated[dict, Depends(get_current_user)])-> dict:
    """
    Endpoint untuk memeriksa apakah current_user memiliki role admin.
    """
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Forbidden:admin only")
    return current_user

# ─────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────

class UserResponse(BaseModel):
    id: int
    name: str
    score: float


class SystemInfoResponse(BaseModel):
    app_name: str
    max_items: int
    debug: bool
    requester_role: str
    request_id: str


# ─────────────────────────────────────────────
# ENDPOINTS — pakai semua dependency di atas
# ─────────────────────────────────────────────

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    db: Annotated[FakeDatabase, Depends(get_db)],           # ← inject DB session
    current_user: Annotated[dict, Depends(get_current_user)] # ← inject auth
):
    """
    Endpoint ini menerima dua dependency:
    - db: FakeDatabase   → otomatis dibuat dan di-close oleh get_db()
    - current_user: dict → otomatis divalidasi oleh get_current_user()

    Perhatikan: endpoint ini TIDAK tahu bagaimana session dibuat
    atau bagaimana auth divalidasi. Dia hanya pakai hasilnya.
    """
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


@app.get("/system/info", response_model=SystemInfoResponse)
def get_system_info(
    config: Annotated[AppConfig, Depends(get_config)],        # ← inject config
    current_user: Annotated[dict, Depends(get_current_user)],  # ← inject auth
    get_id: Annotated[str, Depends(get_request_id)]           # ← inject request ID
):
    """
    Endpoint berbeda, auth dependency SAMA — tidak perlu tulis ulang logika auth.
    FastAPI akan panggil get_current_user() lagi untuk endpoint ini.
    """
    return {
        "app_name": config.app_name,
        "max_items": config.max_items,
        "debug": config.debug,
        "requester_role": current_user["role"],
        "request_id": get_id
    }


@app.get("/public/ping")
def ping():
    """
    Endpoint tanpa dependency apapun.
    Tidak butuh auth, tidak butuh DB.
    """
    return {"status": "ok", "message": "pong"}


@app.get("/admin/stats")
def admin_stats(
    admin: Annotated[dict, Depends(get_admin_user)]  # ← inject dependency-nya
):
    return {"total_users": 2, "accessed_by": admin["user_id"]}

def get_test_config() -> AppConfig:
    """Config khusus testing — nilai berbeda dari production"""
    config = AppConfig()
    config.app_name = "ML Risk API [TEST MODE]"
    config.debug = False
    config.max_items = 5
    return config

# Override: ganti get_config dengan get_test_config
# app.dependency_overrides[get_config] = get_test_config