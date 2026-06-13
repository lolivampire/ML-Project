import pytest

# Penanda bahwa fungsi ini adalah fungsi asynchronous
@pytest.mark.asyncio
async def test_create_and_get_prediction(async_client):
    # ─── 1. TEST POST (CREATE) ──────────────────────────────
    payload = {
        "title": "Simulasi E2E Testing",
        "parameters": {"batch_size": 32},
        "status": "pending",
        "scenarios": [
            {
                "scenario_type": "optimistic",
                "risk_score": 0.1,
                "efficiency_index": 0.95
            }
        ]
    }

    # Tembak endpoint POST
    post_response = await async_client.post("/predictions/", json=payload)
    
    # Validasi bahwa server merespons dengan 201 Created
    assert post_response.status_code == 201
    
    # Ambil data JSON dari respons
    created_data = post_response.json()
    
    # Validasi bahwa data yang disimpan sesuai dengan yang dikirim
    assert created_data["title"] == "Simulasi E2E Testing"
    assert "id" in created_data
    
    # Simpan ID yang baru saja digenerate oleh PostgreSQL
    new_request_id = created_data["id"]

    # ─── 2. TEST GET BY ID (READ) ───────────────────────────
    # Tembak endpoint GET menggunakan ID yang baru dibuat
    get_response = await async_client.get(f"/predictions/{new_request_id}")
    
    # Validasi bahwa server merespons dengan 200 OK
    assert get_response.status_code == 200
    
    # Validasi bahwa data yang ditarik sama dengan data yang dibuat
    fetched_data = get_response.json()
    assert fetched_data["id"] == new_request_id
    assert fetched_data["title"] == "Simulasi E2E Testing"

@pytest.mark.asyncio
async def test_create_prediction_validation_error(async_client):
    # ─── 3. TEST PYDANTIC VALIDATION (ERROR 422) ────────────
    # Sengaja mengirim payload yang salah (risk_score > 1.0)
    bad_payload = {
        "title": "Data Rusak",
        "scenarios": [{"scenario_type": "base", "risk_score": 5.0}]
    }

    response = await async_client.post("/predictions/", json=bad_payload)
    
    # Validasi bahwa server menolak dengan 422 Unprocessable Entity
    assert response.status_code == 422
    assert response.json()["success"] is False