# dependencies/common.py
from fastapi import Header, HTTPException

# Dependency yang bisa di-inject ke router manapun
# Ini sama persis dengan yang kamu pelajari di W05D03
# Bedanya sekarang ada di file sendiri — reusable
async def verify_token(x_token: str = Header(...)):
    """Cek apakah request punya token yang valid."""
    if x_token != "secret-token":
        raise HTTPException(
            status_code=401,
            detail="Token tidak valid"
        )
    return x_token

# Dependency untuk pagination — contoh shared logic lain
def get_pagination(skip: int = 0, limit: int = 10):
    """Standard pagination params untuk semua endpoint yang butuh list."""
    if limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit maksimum 100"
        )
    return {"skip": skip, "limit": limit}