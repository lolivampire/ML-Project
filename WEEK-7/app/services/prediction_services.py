# app/services/prediction_service.py
# Semua logic ada di sini. Tidak import FastAPI. Tidak tahu soal HTTP.

from app.models.prediction_result import PredictionResult
from app.schemas.prediction import PredictionRequest

def calculate_risk_score(age: int, income: float, loan_amount: float) -> float:
    """
    Kalkulasi skor risiko berdasarkan debt-to-income ratio.
    
    Logika sederhana untuk demo:
    - debt_ratio = loan_amount / (income * 12)  ← pinjaman vs pendapatan tahunan
    - Makin tinggi ratio → makin berisiko
    """
    annual_income = income * 12
    if annual_income == 0:
        return 1.0 #edge case: resiko minimal
    debt_ratio = loan_amount / annual_income
    
    # Normalisasi ke range 0.0 - 1.0
    # debt_ratio > 1.0 (pinjaman > pendapatan setahun) = risiko penuh
    risk_score = min(debt_ratio, 1.0)

    return round(risk_score, 4)

def get_risk_label(risk_score: float) -> str:
    """
    Konversi skor numerik ke label dan pesan yang bisa dibaca manusia.
    Return tuple: (label, message)
    """
    if risk_score < 0.3:
        return "LOW", "Profil risiko rendah. Kemungkinan besar disetujui."
    elif risk_score < 0.6:
        return "MEDIUM", "Profil risiko sedang. Harap diperhatikan dan evluasi lanjut."
    else:
        return "HIGH", "Profil risiko tinggi. Kemungkinan ditolak."
    
def process_prediction(request: PredictionRequest) -> PredictionResult:
    """
    Entry point utama service — dipanggil oleh router.
    
    Menerima PredictionRequest, return PredictionResult.
    Perhatikan: tidak ada HTTPException di sini, tidak ada status code.
    """
    risk_score = calculate_risk_score(request.age, request.income, request.loan_amount)
    risk_label, message = get_risk_label(risk_score)
    return PredictionResult(risk_score, risk_label, message)