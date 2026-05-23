# tests/test_endpoint.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.main import app
from app.routers.predict import get_model_service

client = TestClient(app)

# ── SETUP: MOCK MENGGUNAKAN DEPENDENCY OVERRIDES ──────────────
def override_get_model_service():
    """Fungsi ini akan menggantikan get_model_service asli milik FastAPI selama testing."""
    mock_service = MagicMock()
    mock_service.predict.return_value = {
        "id": 99,
        "prediction": 0,
        "probability": 0.2,
        "risk_tier": "low",
        "features": [1.0, 1.0, 1.0, 1.0, 1.0],
        "created_at": "2026-05-21T10:00:00"
    }
    return mock_service

# Timpa dependency asli dengan mock kita
app.dependency_overrides[get_model_service] = override_get_model_service

# ── TEST 1: POST valid → 200 OK ───────────────────────────────
def test_predict_endpoint_returns_200():
    response = client.post(
        "/predict",
        json={"features": [1.0, 1.0, 1.0, 1.0, 1.0]} # Struktur JSON sesuai schema Pydantic
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 99
    assert data["risk_tier"] == "low"

# ── TEST 2: POST input invalid → 422 Unprocessable Entity ─────
def test_predict_endpoint_rejects_bad_input():
    response = client.post(
        "/predict",
        json={"features": ["a", "b", "c", "d", "e"]} # Huruf, bukan angka
    )
    assert response.status_code == 422

# ── TEST 3 : Verifikasi interaksi mock service ─────
def test_predict_called_service_with_correct_data():
    """
    Pastikan endpoint /predict memanggil service dengan data yang benar. 
    Bukan hanya "statusnya 200" — tapi verifikasi bahwa mock service dipanggil dengan argumen yang tepat.
    using mock_predict.assert_called_once_with(...).
    """
    # 1. Buat mock service secara lokal untuk test ini saja
    mock_service = MagicMock()
    # berikan return value palsu agar router tidak error saat menerima kembalianya
    mock_service.predict.return_value = {
        "id": 1,
        "prediction": 0,
        "probability": 0.1,
        "risk_tier": "low",
        "features": [1.5, 2.5, 3.5, 4.5, 5.5],
        "created_at": "2026-05-21T10:00:00"
    }
    # 2. Timpa dependency dengan fungsi lambda yang menimpa mock    
    app.dependency_overrides[get_model_service] = lambda: mock_service

    # 3. Eksekusi request dari client
    response = client.post(
        "/predict",
        json={"features": [1.5, 2.5, 3.5, 4.5, 5.5]} # Struktur JSON sesuai schema Pydantic
    )

    # 4. Pastikan mock service dipanggil dengan argumen yang benar, hanya sekali
    # dan argumen yang masuk benar-benar berupa list Python biasa, bukan object Pydantic
    mock_service.predict.assert_called_once_with([1.5, 2.5, 3.5, 4.5, 5.5])

    #5. bersikan override dependency
    app.dependency_overrides.clear()