# w05d02_pydantic_validation.py
# Week 05 - Day 02: Pydantic Schema & Validation
# Hubungan dengan W05D01: schema ini menggantikan
# parameter loose di routing kemarin. response_model
# sekarang punya pair yang proper: request schema.

from typing import Optional, Literal
from fastapi import FastAPI, status
from pydantic import BaseModel, Field

app = FastAPI(title="W05D02 - Pydantic Validation")


# ── REQUEST SCHEMAS ─────────────────────────────────────────
# Schema terpisah untuk input — tidak pernah campur dengan response

class AddressSchema(BaseModel):
    """Nested model: alamat sebagai objek tersendiri."""
    street: str = Field(..., min_length=5)   # ... = required, tidak ada default
    city: str
    postal_code: str = Field(..., pattern=r"^\d{5}$")  # regex: 5 digit


class UserCreateRequest(BaseModel):
    """
    Schema untuk membuat user baru.
    Setiap field punya kontrak yang jelas — bukan cek manual di endpoint.
    """
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=18, le=120)       # ge=greater or equal, le=less or equal
    email: str = Field(..., min_length=5)
    role: Literal["admin", "user", "viewer"] = "user"  # default "user"
    bio: Optional[str] = Field(None, max_length=200)   # boleh None, max 200 char
    address: Optional[AddressSchema] = None            # nested model, opsional


# ── RESPONSE SCHEMAS ─────────────────────────────────────────
# Ingat dari W05D01: response_model = filter, bukan formatter
# ID dan created_at tidak ada di request — hanya muncul di response

class UserResponse(BaseModel):
    id: int
    name: str
    role: str
    bio: Optional[str] = None
    # Sengaja tidak include email/address — privacy filter via response_model

#schema baru untuk product

class WarehouseLocation(BaseModel):
    """Nested model: alamat sebagai objek tersendiri."""
    warehouse_id: str = Field(..., min_length=5)   # ... = required, tidak ada default
    shelf: str = Field(..., min_length=5)
    quantity: int = Field(..., ge=0)

class ProductCreateRequest(BaseModel):
    """Schema untuk membuat product baru."""
    name: str = Field(..., min_length=3, max_length=100)
    price: float = Field(..., gt=0) # gt=greater than
    stock: int = Field(..., ge=0)
    category: Literal["elektronik", "pakaian", "makanan"]
    description: Optional[str] = Field(None, max_length=500)
    locations: list[WarehouseLocation] = Field(..., min_length=1)


class ProductResponse(BaseModel):
    id: int
    name: str
    price: float
    category: str

# ── FAKE DB ──────────────────────────────────────────────────
# Simulasi database sederhana untuk demo

users_db: list[dict] = []
products_db: list[dict] = []
_user_id_counter = 1
_product_id_counter = 1


# ── ENDPOINTS ────────────────────────────────────────────────

@app.post(
    "/users",
    response_model=UserResponse,          # filter output: hanya field di UserResponse
    status_code=status.HTTP_201_CREATED,  # 201 untuk resource baru
)
def create_user(user: UserCreateRequest):
    """
    Terima data user, validasi via Pydantic, simpan ke fake DB.

    Jika data tidak valid → 422 otomatis dari Pydantic, endpoint ini
    tidak pernah dipanggil. Engineer tidak perlu tulis cek manual.
    """
    global _user_id_counter

    # user.model_dump() → dict dari Pydantic model
    # Pydantic v2: model_dump() | Pydantic v1: dict()
    new_user = {
        "id": _user_id_counter,
        **user.model_dump(),   # spread semua field dari request
    }
    users_db.append(new_user)
    _user_id_counter += 1

    # FastAPI otomatis filter via response_model=UserResponse
    # email dan address tidak akan muncul di response
    return new_user


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    """Ambil user berdasarkan ID. Path parameter user_id otomatis divalidasi int."""
    from fastapi import HTTPException

    user = next((u for u in users_db if u["id"] == user_id), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found",
        )
    return user


@app.get("/users", response_model=list[UserResponse])
def list_users(role: Optional[Literal["admin", "user", "viewer"]] = None):
    """
    List semua user, dengan optional filter by role.
    Query parameter 'role' juga divalidasi Pydantic via Literal.
    """
    if role:
        return [u for u in users_db if u["role"] == role]
    return users_db


# ── ENDPOINTS PRODUCT ────────────────────────────────────────────────

@app.post(
    "/products",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_product(product: ProductCreateRequest):
    global _product_id_counter
    new_product = {
        "id": _product_id_counter,
        **product.model_dump(),
    }
    products_db.append(new_product)
    _product_id_counter += 1
    return new_product