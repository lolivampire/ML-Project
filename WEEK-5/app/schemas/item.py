# schemas/item.py
from pydantic import BaseModel, Field
from typing import Optional

# Request model — apa yang user kirim ke API
class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    description: Optional[str] = None

# Response model — apa yang API kembalikan ke user
# Bedanya: ada 'id' yang di-generate server, bukan dari user
class ItemResponse(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None

    class Config:
        from_attributes = True  # untuk kompatibilitas ORM nanti di W14