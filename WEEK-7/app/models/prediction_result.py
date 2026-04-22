# app/models/prediction_result.py
# Representasi internal hasil kalkulasi — bukan HTTP, bukan database (belum)

from  dataclasses import dataclass

@dataclass
class PredictionResult:
    """Representasi hasil prediksi di level domain/business logic."""
    risk_score: float
    risk_label: str
    message: str

    # Catatan: dataclass otomatis generate __init__, __repr__, __eq__
    # Mirip Pydantic tapi tanpa validasi — ini untuk internal use

# Schema adalah kontrak HTTP (apa yang masuk/keluar dari API). Model adalah representasi internal (apa yang service manipulasi). 
# Nanti di W13+, models/ akan berisi SQLAlchemy model yang mapping ke tabel database — kalau sudah tercampur dari awal, refactor akan menyakitkan.