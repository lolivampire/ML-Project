# app/schemas/simulation_schema.py
from pydantic import BaseModel, Field
from typing import Literal

class SimulationResult(BaseModel):
    """Output metrik komputasi untuk satu skenario simulasi tertentu."""
    
    scenario: Literal["optimistic", "realistic", "pessimistic"] = Field(
        ..., description="Kategori skenario simulasi"
    )
    score: float = Field(
        ..., ge=0.0, le=100.0, description="Skor keberhasilan dari 0.0 hingga 100.0"
    )
    projected_revenue: float = Field(
        ..., description="Proyeksi pendapatan akhir dalam satuan mata uang (IDR)"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ..., description="Klasifikasi tingkat risiko spesifik untuk skenario ini"
    )
    recommendation: str = Field(
        ..., description="Teks ringkasan rekomendasi tindakan bisnis"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Tingkat keyakinan model terhadap proyeksi ini (0.0 - 1.0)"
    )

class SimulationOutput(BaseModel):
    """Kumpulan lengkap hasil simulasi yang mencakup tiga kemungkinan skenario."""
    
    optimistic: SimulationResult
    realistic: SimulationResult
    pessimistic: SimulationResult
    summary: str = Field(..., description="Ringkasan eksekutif dari keseluruhan analisis")