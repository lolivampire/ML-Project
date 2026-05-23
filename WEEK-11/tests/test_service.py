# tests/test_service.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from app.services.model_services import ModelService

# ── SETUP: buat service dengan dependency palsu ───────────────
@pytest.fixture
def mock_repo():
    return MagicMock()

@pytest.fixture
def mock_ml_model():
    """Membuat model Scikit-Learn palsu."""
    model = MagicMock()
    # Atur perilaku model.predict
    model.predict.return_value = np.array([1])
    # Atur perilaku model.predict_proba (harus berbentuk array 2D seperti scikit-learn)
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    return model

@pytest.fixture
def service(mock_ml_model, mock_repo):
    """Inject mock_model dan mock_repo ke dalam service."""
    return ModelService(model=mock_ml_model, repo=mock_repo)

# ── TEST 1: Service berhasil return prediksi ──────────────────
def test_predict_returns_result(service, mock_repo):
    """Service harus return hasil prediksi dan menyimpan ke repo."""
    # Instruksikan repo mock
    mock_repo.save.return_value = {
        "id": 1,
        "prediction": 1,
        "probability": 0.8,
        "risk_tier": "high",
        "features": [1.0, 2.0, 3.0, 4.0, 5.0],
        "created_at": "2026-05-21T10:00:00"
    }

    # Data dummy murni list (karena Router melempar list ke Service)
    features_input = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = service.predict(features_input)

    assert "prediction" in result
    assert result["risk_tier"] == "high"

# ── TEST 2: Service raise ValueError kalau input kosong ───────
def test_predict_raises_for_empty_list(service):
    """Service harus raise ValueError kalau list features kosong."""
    with pytest.raises(ValueError, match="Feature list tidak boleh kosong."):
        service.predict([])

# ── TEST 3: Verifikasi Interaksi dengan Repository ────────────
def test_predict_calls_repo_once(service, mock_repo):
    """Service harus memanggil repo.save() tepat 1x dengan data yang benar."""
    features_input = [0.1, 0.2, 0.3, 0.4, 0.5]
    service.predict(features_input)

    # Verifikasi bahwa repo.save dipanggil tepat 1 kali
    mock_repo.save.assert_called_once()

# ── TEST 4 : RuntimeError deskriptif  dari mock_repo.save_prediction raise RuntimeError ────────────
def test_predict_handles_repo_error_and_preserves_cause (service, mock_repo):
    """Service harus menangkap RuntimeError dari repo dan me-raise ulang dengan mempertahankan __cause__."""
    # 1. simulasikan error database pada repo menggunakan side_effect
    true_error = RuntimeError("Koneksi ke database terputus")
    mock_repo.save.side_effect = true_error

    features_input = [1.0, 2.0, 3.0, 4.0, 5.0]

    # 2. Panggil service predict
    with pytest.raises(RuntimeError) as excinfo:
        service.predict(features_input)

    # 3. Verifikasi bahwa RuntimeError dipanggil dengan __cause__ yang sesuai
    assert excinfo.value.__cause__ == true_error
    