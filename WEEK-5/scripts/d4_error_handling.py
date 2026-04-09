from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

app = FastAPI(title="Error Handling & Validation API")

# ── Model untuk format error yang konsisten ──────────────────
class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str        # misal: "NOT_FOUND", "OUT_OF_STOCK"
    message: str           # pesan yang bisa dibaca manusia
    detail: str | None = None  # info tambahan opsional

# ── Custom exception class ────────────────────────────────────
class AppException(Exception):
    """
    Gunakan class ini di dalam endpoint/logic bisnis jika ada aturan 
    bisnis yang dilanggar (misal: stok habis, item tidak ditemukan).
    """
    def __init__(self, status_code: int, error_code: str, message: str, detail: str | None = None):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.detail = detail

# ─────────────────────────────────────────────────────────────────
#  PYDANTIC MODELS UNTUK ITEM (Validasi Data)
# ─────────────────────────────────────────────────────────────────

class ItemCreate(BaseModel):
    """
    Schema untuk request body saat membuat/mengupdate item.
    FastAPI otomatis menolak request yang tidak sesuai aturan ini.
    """
    # min_length=2 memastikan nama minimal 2 karakter
    name: str = Field(..., min_length=2, example="Laptop Gaming")
    
    # gt=0 (greater than 0) memastikan harga > 0
    price: float = Field(..., gt=0, example=15000000.0)
    
    # ge=0 (greater than or equal to 0) memastikan stok tidak negatif
    stock: int = Field(..., ge=0, example=10)

# ─────────────────────────────────────────────────────────────────
#  Item response untuk itemCreate
# ─────────────────────────────────────────────────────────────────

class ItemResponse(ItemCreate):
    """
    Schema untuk response data item.
    Dengan melakukan inheritance (pewarisan) dari ItemCreate, 
    kita tidak perlu menulis ulang name, price, dan stock! (Prinsip DRY)
    """
    id: int = Field(..., example=1)

# ── Handler: tangkap AppException, ubah jadi response JSON ───
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error_code=exc.error_code,
            message=exc.message,
            detail=exc.detail,
        ).model_dump()  # ubah Pydantic model jadi dict
    )

# ── Handler: tangkap HTTPException standar FastAPI ────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
        ).model_dump()
    )

# ── Handler: tangkap error validasi tipe data dari FastAPI ───
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            success=False,
            error_code="VALIDATION_ERROR",
            message="Data yang dikirim tidak valid",
            # exc.errors() mengembalikan list/array tentang detail field apa yang salah
            detail=str(exc.errors()) 
        ).model_dump()
    )

# ── Handler: tangkap error Python yang tidak terduga (500) ───
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # PENTING: Di production, kamu harus nge-log error 'exc' ke sistem tracking
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error_code="INTERNAL_SERVER_ERROR",
            message="Terjadi kesalahan pada server",
            detail="Silakan coba beberapa saat lagi."
        ).model_dump()
    )

# ── Fake DB ──────────────────────────────────────────────────
# Tambahkan key "price" pada setiap item
fake_db: dict[int, dict] = {
    1: {"id": 1, "name": "Laptop", "price": 15000000.0, "stock": 10},
    2: {"id": 2, "name": "Mouse", "price": 250000.0, "stock": 0},
}

# ─────────────────────────────────────────────────────────────────
# 6. ENDPOINTS CRUD & BISNIS
# ─────────────────────────────────────────────────────────────────

# --- A. GET ALL ITEMS ---
@app.get("/items", response_model=list[ItemResponse], status_code=status.HTTP_200_OK)
def get_all_items():
    """Mengambil semua daftar item."""
    return list(fake_db.values())

# --- B. GET ITEM BY ID ---
@app.get("/items/{item_id}", response_model=ItemResponse, status_code=status.HTTP_200_OK)
def get_item(item_id: int):
    """Mengambil satu item berdasarkan ID."""
    if item_id not in fake_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id={item_id} not found"
        )
    return fake_db[item_id]

# --- C. POST CREATE ITEM ---
@app.post("/items", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate):
    """
    Membuat item baru. 
    422 Error otomatis di-handle oleh FastAPI + Pydantic jika data tidak valid.
    """    
    new_id = max(fake_db.keys(), default=0) + 1
    # Gabungkan ID baru dengan data dari Pydantic body
    new_item = {"id": new_id, **item.model_dump()}
    fake_db[new_id] = new_item
    return fake_db[new_id]

# --- D. PUT UPDATE ITEM ---
@app.put("/items/{item_id}", response_model=ItemResponse, status_code=status.HTTP_200_OK)  
def update_item(item_id: int, item: ItemCreate):
    """
    Memperbarui item berdasarkan ID. 
    422 Error otomatis di-handle oleh FastAPI + Pydantic jika data tidak valid.
    """
    if item_id not in fake_db:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="ITEM_NOT_FOUND",
            message=f"Gagal memperbarui. Item ID {item_id} tidak ditemukan."
        )
    #update data di database
    updated_item = {"id": item_id, **item.model_dump()}
    fake_db[item_id] = updated_item
    return fake_db[item_id]

# --- E. DELETE ITEM ---
@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    if item_id not in fake_db:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="ITEM_NOT_FOUND",
            message=f"Gagal menghapus. Item ID {item_id} tidak ditemukan."
        )
    del fake_db[item_id]
    return None #wajib tidak return body

# --- POST PURCHASE ITEM(custom business logic) ---
@app.post("/items/{item_id}/purchase", status_code=status.HTTP_200_OK)
def purchase_item(item_id: int, quantity: int = 1):
    """
    Membeli item (mengurangi stok sebanyak 1).
    Akan error 404 jika item tidak ada, atau 400 jika stok habis.
    """
    if item_id not in fake_db:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="ITEM_NOT_FOUND",
            message=f"Gagal membeli. Item ID {item_id} tidak ditemukan."
        )
    item = fake_db[item_id]
    if fake_db[item_id]["stock"] < quantity:
        raise AppException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="STOCK_NOT_ENOUGH",
            message=f"Gagal membeli. Stok item ID {item_id} tidak cukup."
        )
    item["stock"] -= quantity
    return item