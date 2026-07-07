# week-05/scripts/w05d05_async_background.py

import asyncio
import time
from fastapi import FastAPI, status, BackgroundTasks
import logging
from pydantic import BaseModel, EmailStr, Field

app = FastAPI()

# ─── SYNC ENDPOINT (blocking) ─────────────────────────────
@app.get("/sync-slow")
def sync_slow():
    # time.sleep MEMBLOKIR seluruh event loop
    # Semua request lain menunggu selama 3 detik ini
    time.sleep(3)
    return {"message": "sync done", "type": "blocking"}


# ─── ASYNC ENDPOINT (non-blocking) ────────────────────────
@app.get("/async-slow")
async def async_slow():
    # asyncio.sleep TIDAK memblokir event loop
    # FastAPI bisa handle request lain selama "menunggu" ini
    await asyncio.sleep(3)
    return {"message": "async done", "type": "non-blocking"}


# ─── ASYNC DENGAN DATABASE SIMULATION ─────────────────────
async def fetch_from_db(user_id: int) -> dict:
    """Simulasi async database call."""
    await asyncio.sleep(0.1)  # simulasi query 100ms
    return {"id": user_id, "name": f"User {user_id}"}


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # await = pause di sini, jangan blokir, kerjakan yang lain
    user = await fetch_from_db(user_id)
    return user


# ─── CONCURRENT ASYNC CALLS ───────────────────────────────
@app.get("/dashboard/{user_id}")
async def get_dashboard(user_id: int):
    # asyncio.gather = jalankan SEMUA sekaligus, bukan satu-satu
    # Total waktu ≈ maks dari semua, bukan jumlah dari semua
    user, orders, notifications = await asyncio.gather(
        fetch_from_db(user_id),           # 100ms
        fetch_orders(user_id),            # 150ms
        fetch_notifications(user_id),     # 80ms
    )
    # Total ≈ 150ms, BUKAN 330ms
    return {
        "user": user,
        "orders": orders,
        "notifications": notifications
    }


async def fetch_orders(user_id: int) -> list:
    await asyncio.sleep(0.15)
    return [{"order_id": 1, "item": "laptop"}]


async def fetch_notifications(user_id: int) -> list:
    await asyncio.sleep(0.08)
    return [{"msg": "Package shipped"}]

# (lanjutan file yang sama: w05d05_async_background.py)



logger = logging.getLogger(__name__)


# ─── SCHEMA ────────────────────────────────────────────────
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3)
    email: str = Field(..., min_length=5)  # simplified
    password: str = Field(..., min_length=8)


class RegisterResponse(BaseModel):
    message: str
    username: str


# ─── BACKGROUND FUNCTIONS ──────────────────────────────────
# Ini fungsi BIASA — tidak perlu async
# Background task bisa sync atau async, keduanya OK
def send_welcome_email(email: str, username: str) -> None:
    """Kirim email welcome. Berjalan SETELAH response dikirim ke client."""
    # Di produksi: pakai SMTP atau service seperti SendGrid
    logger.info(f"Sending welcome email to {email} for user {username}")
    time.sleep(0.5)  # simulasi: kirim email butuh 500ms
    logger.info(f"Welcome email sent to {email}")


def log_registration_analytics(username: str) -> None:
    """Catat event registrasi ke analytics service."""
    logger.info(f"Analytics: new_user_registered | username={username}")
    # Di produksi: kirim ke Mixpanel, Amplitude, dsb


# ─── ENDPOINT DENGAN BACKGROUND TASK ──────────────────────
@app.post("/register", response_model=RegisterResponse, status_code=201)
async def register_user(
    user: UserRegister,
    background_tasks: BackgroundTasks  # ← FastAPI inject otomatis, no Depends()
):
    # 1. Simpan user ke DB (simulasi)
    logger.info(f"Registering user: {user.username}")

    # 2. Daftarkan background tasks
    # add_task(fungsi, *args) — fungsi TIDAK dipanggil sekarang
    background_tasks.add_task(send_welcome_email, user.email, user.username)
    background_tasks.add_task(log_registration_analytics, user.username)

    # 3. Response LANGSUNG dikembalikan ke client
    # Email belum terkirim, tapi client tidak perlu tahu/tunggu
    return RegisterResponse(
        message="Registration successful. Welcome email on its way!",
        username=user.username
    )


# ─── MULTIPLE BACKGROUND TASKS ─────────────────────────────
@app.post("/orders/{order_id}/confirm")
async def confirm_order(
    order_id: int,
    background_tasks: BackgroundTasks
):
    # Tasks dijalankan BERURUTAN setelah response
    # (bukan paralel — jika perlu paralel, pakai asyncio.gather di dalam async task)
    background_tasks.add_task(notify_customer, order_id)
    background_tasks.add_task(update_inventory, order_id)
    background_tasks.add_task(trigger_fulfillment, order_id)

    return {"order_id": order_id, "status": "confirmed"}


async def notify_customer(order_id: int) -> None:
    await asyncio.sleep(0.2)
    logger.info(f"Customer notified for order {order_id}")


async def update_inventory(order_id: int) -> None:
    await asyncio.sleep(0.1)
    logger.info(f"Inventory updated for order {order_id}")


async def trigger_fulfillment(order_id: int) -> None:
    await asyncio.sleep(0.3)
    logger.info(f"Fulfillment triggered for order {order_id}")

# ─────────────────────────────────────────────────────────────────
# 7. EKSPERIMEN SYNC VS ASYNC
# ─────────────────────────────────────────────────────────────────

@app.get("/sync-task")
def sync_task():
    """
    Endpoint Synchronous (Blocking).
    Menggunakan time.sleep(). Selama 2 detik ini, Koki (FastAPI) mematung.
    """
    time.sleep(2)
    return {"type": "sync", "duration": 2}


@app.get("/async-task")
async def async_task():
    """
    Endpoint Asynchronous (Non-blocking).
    Menggunakan await asyncio.sleep(). Selama 2 detik ini, Koki bisa melayani pelanggan lain.
    """
    await asyncio.sleep(2)
    return {"type": "async", "duration": 2}

# ─────────────────────────────────────────────────────────────────
# 8. Background Task: Simulasi Audit Log
# ─────────────────────────────────────────────────────────────────
import time
import logging
from fastapi import BackgroundTasks

# Setup logger sederhana untuk melihat output di terminal
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 8. BACKGROUND TASK: AUDIT LOG SYSTEM
# ─────────────────────────────────────────────────────────────────

# Schema baru khusus untuk audit (menambah field category)
class ItemAuditCreate(BaseModel):
    name: str = Field(..., min_length=2)
    price: float = Field(..., gt=0)
    category: str = Field(..., min_length=1)

# Fungsi yang akan berjalan di background
def record_audit_log(item_name: str, category: str):
    """
    Simulasi fungsi audit yang lambat.
    Fungsi ini berjalan SETELAH response dikirim ke user.
    """
    time.sleep(1)  # Simulasi proses penulisan ke file atau database audit
    print(f"\n[BACKGROUND PROCESS] AUDIT: item '{item_name}' created in category '{category}'")
    logger.info(f"Audit record completed for {item_name}")

@app.post("/items/audit-test", status_code=status.HTTP_201_CREATED)
async def create_item_with_audit(
    item: ItemAuditCreate, 
    background_tasks: BackgroundTasks
):
    """
    Endpoint ini akan mengembalikan response secepat kilat, 
    sementara proses audit 'record_audit_log' berjalan di belakang layar.
    """
    # Simulasi logic simpan data (cepat)
    new_id = 999 
    
    # Daftarkan fungsi ke background tasks
    # Format: background_tasks.add_task(NAMA_FUNGSI, PARAMETER1, PARAMETER2)
    background_tasks.add_task(record_audit_log, item.name, item.category)
    
    # Return langsung ke client tanpa menunggu background task selesai
    return {
        "item_id": new_id,
        "name": item.name,
        "message": "Item created. Audit log is being recorded in the background."
    }

import time
import asyncio

# ─────────────────────────────────────────────────────────────────
# 9. ASYNCIO.GATHER CHALLENGE (PARALLEL EXECUTION)
# ─────────────────────────────────────────────────────────────────

# --- 3 Fungsi Async Tiruan ---

async def get_user_profile(user_id: int) -> dict:
    await asyncio.sleep(0.3)
    return {"id": user_id, "name": f"User {user_id}", "status": "Premium"}

async def get_user_orders(user_id: int) -> list:
    await asyncio.sleep(0.5)
    return [
        {"order_id": 101, "item": "Mechanical Keyboard", "price": 1200000},
        {"order_id": 102, "item": "Mousepad", "price": 150000}
    ]

async def get_user_balance(user_id: int) -> float:
    await asyncio.sleep(0.2)
    return 5500000.0


# --- Endpoint Utama ---

@app.get("/summary/{user_id}")
async def get_user_summary(user_id: int):
    """
    Endpoint ini akan mengambil data dari 3 'sumber' berbeda secara berbarengan.
    Total waktu eksekusi harusnya ~0.5 detik, bukan 1.0 detik.
    """
    # 1. Mulai stopwatch
    start_time = time.time()
    
    # 2. Jalankan semuanya secara berbarengan (paralel)
    # PENTING: Urutan kembalian (return) akan sesuai dengan urutan kamu menaruh fungsinya di gather
    profile, orders, balance = await asyncio.gather(
        get_user_profile(user_id),
        get_user_orders(user_id),
        get_user_balance(user_id)
    )
    
    # 3. Hentikan stopwatch
    end_time = time.time()
    total_duration = round(end_time - start_time, 4) # Bulatkan 4 angka di belakang koma
    
    # 4. Gabungkan dan kembalikan response
    return {
        "execution_time_seconds": total_duration,
        "data": {
            "user": profile,
            "balance": balance,
            "orders": orders
        }
    }


