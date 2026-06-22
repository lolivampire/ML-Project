# app/schemas/input_schema.py
from enum import Enum
from typing import List, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator

class RiskLevel(str, Enum):
    """Representasi tingkat toleransi risiko dalam pengambilan keputusan bisnis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class IndustrySector(str, Enum):
    """Kategorisasi sektor industri untuk keperluan penyesuaian model analitik."""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    RETAIL = "retail"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    OTHER = "other"

class MarketCondition(str, Enum):
    """Sentimen atau kondisi pasar secara makro."""
    BULL = "bull"
    BEAR = "bear"
    FLAT = "flat"

class TimeHorizon(str, Enum):
    """Jangka waktu target penyelesaian atau proyeksi."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

class ScenarioDetail(BaseModel):
    """
    Skema detail untuk setiap skenario komparasi dalam simulasi.
    """
    scenario_name: str = Field(..., min_length=3, max_length=50, description="Nama identifikasi skenario")
    feature_modifiers: Annotated[List[float], Field(min_length=1, description="List modifikator bobot fitur")]
    estimated_cost: float = Field(..., ge=0.0, description="Estimasi biaya eksekusi skenario tidak boleh negatif")

    @field_validator("feature_modifiers")
    @classmethod
    def validate_modifiers(cls, v: List[float]) -> List[float]:
        """Memvalidasi bahwa setiap modifikator berada dalam ambang batas matematis yang diizinkan."""
        for modifier in v:
            if not (-1.0 <= modifier <= 1.0):
                raise ValueError("Setiap komponen dalam feature_modifiers harus berada di rentang [-1.0, 1.0]")
        return v

class DecisionSimulationRequest(BaseModel):
    """
    Skema utama untuk permintaan simulasi skenario keputusan bisnis.
    """
    title: str = Field(..., min_length=5, max_length=100, description="Judul analisis keputusan")
    total_budget: float = Field(..., gt=0.0, description="Total anggaran simulasi harus lebih besar dari 0")
    target_risk: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Tingkat risiko toleransi keputusan")
    features: Annotated[List[float], Field(min_length=1, description="Kumpulan fitur dasar untuk komputasi")]
    scenarios: Annotated[List[ScenarioDetail], Field(min_length=1, description="Minimal harus ada satu skenario komparasi")]

    @model_validator(mode="after")
    def validate_business_rules(self) -> "DecisionSimulationRequest":
        """Memvalidasi integritas data lintas-kolom, memastikan biaya skenario tidak melebihi anggaran."""
        combined_cost = sum(scenario.estimated_cost for scenario in self.scenarios)
        if combined_cost > self.total_budget:
            raise ValueError(
                f"Total estimasi biaya skenario (Rp{combined_cost:,}) melebihi "
                f"anggaran total yang tersedia (Rp{self.total_budget:,})"
            )
        return self

class DecisionInput(BaseModel):
    """
    Skema input dasar untuk analisis keputusan tahap awal.
    """
    company_name: str = Field(..., min_length=2, max_length=100, description="Nama entitas perusahaan")
    budget: float = Field(..., gt=0.0, description="Total anggaran, harus lebih dari 0")
    growth_target: float = Field(..., gt=0.0, description="Target pertumbuhan (misal: 1.5 untuk 150%)")
    risk_appetite: RiskLevel = Field(..., description="Tingkat toleransi risiko")
    market_condition: MarketCondition = Field(..., description="Kondisi pasar saat ini")
    time_horizon: TimeHorizon = Field(..., description="Target jangka waktu proyeksi")
    industry_sector: IndustrySector = Field(..., description="Sektor industri perusahaan")

    @field_validator("industry_sector", mode="before")
    @classmethod
    def process_and_validate_industry_sector(cls, v: str) -> str:
        """
        Melakukan sanitasi pada input sektor industri sebelum validasi tipe Enum Pydantic berjalan.
        Tugas utama: menghapus spasi putih berlebih dan memvalidasi panjang string dasar.
        """
        if isinstance(v, str):
            v = v.strip()
            if not (2 <= len(v) <= 50):
                raise ValueError("Panjang karakter industry_sector harus berada di antara 2 hingga 50 karakter.")
        return v
    @model_validator(mode="after")
    def validate_budget_risk_ratio(self) -> "DecisionInput":
        """
        Aturan Bisnis: Entitas dengan anggaran di bawah 50.000 tidak diizinkan 
        mengambil profil risiko 'HIGH' untuk melindungi dari kebangkrutan simulasi.
        """
        if self.budget < 50000 and self.risk_appetite == RiskLevel.HIGH:
            raise ValueError(
                f"Anggaran terlalu kecil (Rp{self.budget:,}) untuk profil risiko 'high'. "
                "Silakan tingkatkan anggaran minimum ke Rp50,000 atau turunkan toleransi risiko."
            )
        return self