"""
w05d01_routing.py
Week 05 - Day 01: FastAPI Setup & Routing Dasar

Demonstrasi: GET, POST, DELETE endpoint.
Path parameter vs query parameter.
"""

from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, status  # FastAPI class + error handler
from pydantic import BaseModel, Field              # Untuk mendefinisikan struktur data request

# ── INISIALISASI APP ──────────────────────────────────────────────────────────
app = FastAPI(
    title="ML API — W05D01",
    description="Latihan routing dasar FastAPI",
    version="0.1.0",
)

# ── FAKE DATABASE (simulasi, belum pakai DB nyata) ────────────────────────────
# Di Phase 1 kamu pakai dict untuk store data sementara — sama konsepnya
fake_db: dict[int, dict] = {
    1: {"name": "breast_cancer_model", "version": "1.0", "accuracy": 0.97},
    2: {"name": "iris_classifier",     "version": "2.1", "accuracy": 0.95},
}

# ── REQUEST BODY MODEL ────────────────────────────────────────────────────────
# Ingat type hints dari W01D01? Pydantic memakai ini untuk validasi otomatis.
# Jika kamu kirim {"name": 123} → FastAPI langsung tolak sebelum masuk ke fungsi
class ModelItem(BaseModel):
    name: str           # wajib ada, harus string
    version: str        # wajib ada, harus string
    # Tambahkan batas ge (>= 0.0) dan le (<= 1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Akurasi antara 0.0 - 1.0")

class ModelResponse(ModelItem):
    id: int

# ── ENDPOINT 1: GET semua model ───────────────────────────────────────────────
@app.get("/models")                         # decorator: HTTP method + path
def get_all_models(                         # nama fungsi bebas, tapi harus deskriptif
    skip: int = 0,                          # query param dengan default value 0
    limit: int = 10,                        # query param dengan default value 10
) -> dict:
    """
    Ambil semua model yang terdaftar.
    Contoh: GET /models?skip=0&limit=5
    """
    items = list(fake_db.values())          # ambil semua value dari dict
    return {"models": items[skip : skip + limit], "total": len(items)}

# ── ENDPOINT 2: GET satu model by ID ─────────────────────────────────────────
@app.get("/models/{model_id}")              # {model_id} = path parameter
def get_model(model_id: int) -> dict:       # FastAPI otomatis convert URL string → int
    """
    Ambil satu model berdasarkan ID.
    Contoh: GET /models/1
    """
    if model_id not in fake_db:             # jika ID tidak ada
        # HTTPException: cara FastAPI kirim error dengan status code yang proper
        raise HTTPException(
            status_code=404,                # 404 = Not Found (kamu sudah tahu ini)
            detail=f"Model dengan ID {model_id} tidak ditemukan.",
        )
    return fake_db[model_id]

# ── ENDPOINT 3: POST tambah model baru ───────────────────────────────────────
@app.post("/models", status_code=201, response_model=ModelResponse)       # 201 = Created (bukan 200)
def create_model(item: ModelItem) -> dict:  # item: Pydantic model → auto-validate
    """
    Tambahkan model baru.
    Contoh body: {"name": "xgb_v2", "version": "1.0", "accuracy": 0.93}
    """
    new_id = max(fake_db.keys(), default=0) + 1  # ID baru = ID terbesar + 1
    fake_db[new_id] = item.model_dump()           # simpan ke "database"
    return {"id": new_id, **fake_db[new_id]}      # kembalikan data + ID baru

# ── ENDPOINT 4: DELETE hapus model ───────────────────────────────────────────
@app.delete("/models/{model_id}", status_code=204)  # 204 = No Content (berhasil, tanpa body)
def delete_model(model_id: int) -> None:
    """
    Hapus model berdasarkan ID.
    Contoh: DELETE /models/1
    """
    if model_id not in fake_db:
        raise HTTPException(status_code=404, detail=f"Model ID {model_id} tidak ada.")
    del fake_db[model_id]                   # hapus dari dict
    # 204 tidak return body — return None sudah cukup

# ── BONUS: Health check endpoint ─────────────────────────────────────────────
# Ini WAJIB ada di setiap API production — untuk monitoring
@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "total_models": len(fake_db)}

class DatasetItem(BaseModel):
    name: str = Field(..., example="titanic_dataset")
    # Literal memastikan input HANYA boleh salah satu dari 3 nilai ini
    source: Literal["kaggle", "sklearn", "custom"] 
    num_rows: int = Field(..., gt=0, description="Jumlah baris harus lebih dari 0")
    has_missing: bool = False # Default ke False jika tidak diisi

# --- FAKE DATABASE UNTUK DATASET ---
fake_db_datasets: dict[int, dict] = {
    1: {"name": "iris_csv", "source": "sklearn", "num_rows": 150, "has_missing": False},
    2: {"name": "titanic_data", "source": "kaggle", "num_rows": 891, "has_missing": True},
}

# --- ENDPOINT DATASET ---

# 1. GET ALL DATASETS (dengan filter source)
@app.get("/datasets")
def get_datasets(source: Optional[str] = None):
    """
    Ambil semua dataset. Bisa difilter berdasarkan source.
    Contoh: /datasets?source=kaggle
    """
    results = list(fake_db_datasets.values())
    
    if source:
        # Filter list berdasarkan source yang dikirim di query param
        results = [d for d in results if d["source"].lower() == source.lower()]
    
    return {"datasets": results, "total": len(results)}

# (bonus) GET DATASET SUMMARY harus sebelum GET SINGLE DATASET agar FastAPI tidak mengira kata "summary" adalah sebuah dataset_id
# FastAPI match route dari atas ke bawah.
# "summary" akan dicoba dulu sebagai literal string,
# SEBELUM dicoba sebagai integer {dataset_id}
@app.get("/datasets/summary")
def get_datasets_summary():
    """
    Mengambil ringkasan statistik dari semua dataset yang ada.
    """
    datasets = list(fake_db_datasets.values())

    # Mengambil list source unik menggunakan Set Comprehension
    # Set otomatis membuang duplikat, lalu kita ubah kembali menjadi List
    unique_sources = list(set([d["source"] for d in datasets]))

    # Menghitung dataset yang memiliki missing values (True)
    missing_count = sum(1 for d in datasets if d["has_missing"])

    return {
        "total_datasets": len(datasets),
        "unique_sources": unique_sources,
        "missing_count": missing_count
        
    }

# 2.GET SINGLE DATASET FROM ID
@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int):
    if dataset_id not in fake_db_datasets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Dataset ID {dataset_id} tidak ditemukan"
        )
    return fake_db_datasets[dataset_id]

# 3.POST CREATE DATASET
@app.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(item: DatasetItem):
    """
    Menambah dataset baru. Body harus sesuai dengan schema DatasetItem.
    """
    new_id = max(fake_db_datasets.keys(), default=0) + 1
    # .model_dump() mengubah objek Pydantic menjadi dictionary Python
    fake_db_datasets[new_id] = item.model_dump()
    return {"id": new_id, **fake_db_datasets[new_id]}

# 4.DELETE DATASET
@app.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int):
    if dataset_id not in fake_db_datasets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Dataset ID {dataset_id} tidak ditemukan"
        )
    del fake_db_datasets[dataset_id]
    return None                                                                 #204 No Content tidak mengirim body