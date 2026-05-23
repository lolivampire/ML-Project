# tests/test_schema.py
import pytest
from pydantic import ValidationError
from app.schemas.prediction import PredictionRequest

# ── TEST 1: Input valid harus diterima ────────────────────────
def test_valid_input_accepted():
    """Schema harus menerima 5 fitur berbentuk angka/float."""
    data = PredictionRequest(
        features=[0.5, 1.2, 3.1, 0.8, 2.4]
    )
    # Cek apakah DTO method bekerja
    assert len(data.to_feature_list()) == 5
    assert data.features[0] == 0.5

# ── TEST 2: Input invalid harus ditolak (Tipe Salah) ───────────
def test_invalid_input_raises_validation_error():
    """Schema harus raise ValidationError kalau tipe salah (string bukan list of float)."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            features="bukan_list_tapi_string" # ❌ string, bukan list
        )

# ── TEST 3: Panjang List Harus 5 ──────────────────────────────
def test_missing_required_field_raises():
    """Schema harus raise ValidationError kalau jumlah fitur kurang dari 5."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            features=[1.0, 2.0] # Cuma 2 fitur, harusnya 5
        )

# ── TEST 4: Tidak Boleh Ada Infinity (Test Custom Validator) ──
def test_infinite_value_raises():
    """Schema harus menolak nilai Infinity sesuai custom validator."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            features=[1.0, float('inf'), 3.0, 4.0, 5.0] # Ada Infinity
        )

# ── TEST 5 : jika nilai feature = negatif (-5000), maka raise ValidationError ──
def test_negative_value_raises():
    """Schema harus menolak nilai negatif sesuai custom validator."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            features=[1.0, -5000.0, 3.0, 4.0, 5.0] # Ada negatif
        )