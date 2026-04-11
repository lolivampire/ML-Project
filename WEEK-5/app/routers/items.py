# routers/items.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List

# Import dari dalam project — bukan dari fastapi
from schemas.item import ItemCreate, ItemResponse
from dependencies.common import verify_token, get_pagination

# APIRouter adalah "mini-app" — seperti sub-restoran
# prefix: semua endpoint di file ini otomatis punya awalan /items
# tags: grouping di Swagger UI
router = APIRouter(
    prefix="/items",
    tags=["items"],
)

# Fake database — nanti di W13 diganti PostgreSQL
fake_db: dict[int, dict] = {}
_counter = 0

def log_creation(item_name: str):
    """Background task — berjalan SETELAH response dikirim."""
    print(f"[LOG] Item baru dibuat: {item_name}")

# GET /items — dengan pagination
# Depends(get_pagination): inject pagination params otomatis
@router.get("/", response_model=List[ItemResponse])
async def list_items(
    pagination: dict = Depends(get_pagination),  # inject dari dependencies/
    token: str = Depends(verify_token)           # inject token check
):
    skip = pagination["skip"]
    limit = pagination["limit"]
    items = list(fake_db.values())
    return items[skip : skip + limit]

# POST /items — buat item baru
@router.post("/", response_model=ItemResponse, status_code=201)
async def create_item(
    item: ItemCreate,              # Pydantic validasi otomatis
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    global _counter
    _counter += 1
    new_item = {"id": _counter, **item.model_dump()}
    fake_db[_counter] = new_item

    # Background task dari W05D05 — tetap ada, sekarang di dalam router
    background_tasks.add_task(log_creation, item.name)

    return new_item

# GET /items/{item_id} — ambil satu item
@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item tidak ditemukan")
    return fake_db[item_id]

# DELETE /items/{item_id}
@router.delete("/{item_id}", status_code=204)
async def delete_item(
    item_id: int,
    token: str = Depends(verify_token)
):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item tidak ditemukan")
    del fake_db[item_id]
    # 204 No Content — tidak perlu return apapun