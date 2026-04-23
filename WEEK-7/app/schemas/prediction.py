# app/schemas/prediction.py
# Definisi bentuk request dan response — murni Pydantic, tidak ada logic

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Schema untuk request prediksi risiko."""

    age : int = Field(..., ge=18, le=100, description="Usia harus diantara 18 dan 100 tahun.")
    # ge 18 = minimal 18 tahun
    # le 100 = maksimal 100 tahun
    # '...' = wajib ada (required)

    income : float = Field(..., ge=0, description="Penghasilan tahunan dalam USD.", example=75000.0)
    # ge 0 = minimal 0
    # '...' = wajib ada (required)

    loan_amount : int = Field(..., ge=0, description="Jumlah pinjaman harus lebih besar dari 0.")

class PredictionResponse(BaseModel):
    """Schema untuk response prediksi risiko."""

    risk_score : float = Field(..., ge=0, le=1.0, description="Skor risiko harus diantara 0.0 dan 1.0.", example=0.3)
    #skor resiko dimana 0=aman dan 1=risiko

    risk_label : str = Field(..., description="Label kategori: 'LOW', 'MEDIUM', 'HIGH'.", example="LOW")
    #low, medium, high

    message : str
    #penjelasan human-raeadable

# schema dipisah dari router karena schema bisa dipakai oleh banyak router berbeda. Kalau schema berubah (misal tambah field baru), 
# tinggal edit satu file ini — semua router yang pakai otomatis ikut.