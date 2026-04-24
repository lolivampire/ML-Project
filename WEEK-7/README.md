# 🔍 Risk Scoring API

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

API untuk memprediksi risk score pelanggan berdasarkan data demografis
dan riwayat transaksi. Dibangun dengan FastAPI dan model Random Forest
yang di-serialize menggunakan joblib.

---

## Architecture

Client (curl / Postman)
│
▼
FastAPI app (main.py)
│
├── routers/predict.py   ← routing & validation
├── services/model.py    ← business logic
└── models/rf_model.pkl  ← trained model (loaded at startup)
---

## Installation

```bash
# 1. Clone repo
git clone https://github.com/lolivampire/ML-Project.git
cd ML-Project

# 2. Buat dan aktifkan virtual environment (Windows)
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Masuk ke bagian WEEK-7
cd WEEK-7

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup environment variables
cp .env.example .env
# Edit .env dan isi nilai yang diperlukan
```

---

## How to Run

```bash
uvicorn app.main:app --reload --port 8000
```

Setelah server berjalan, akses:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

---

## API Reference

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/health` | Status server dan model |
| POST | `/predict` | Prediksi risk score |
| GET | `/docs` | Swagger UI |

### POST `/predict`

**Request Body:**
```json
{
  "age": 35,
  "income": 75000,
  "transaction_count": 12
}
```

**Response (200 OK):**
```json
{
  "risk_score": 0.23,
  "risk_label": "low",
  "confidence": 0.91
}
```

**Error Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ]
}
```

---

## Project Structure
week-07/
├── app/
│   ├── main.py                             ← FastAPI app + lifespan
│   ├── routers/
│   │   └── predict.py                      ← terima request, panggil service, return response
│   ├── config/
│   │   └── config.py                       ← semua konfigurasi aplikasi
│   ├── services/
│   │   └── prediction_services.py          ← logika bisnis
│   ├── schemas/
│   │   └── prediction.py                   ← Pydantic models
│   └── models/
│       └── prediction_result.py            ← Domain result model.
├── .env                                    ← env variables (gitignored — jangan di-push)
├── .env.example                            ← template env variables
├── requirements.txt                        ← dependencies
└── README.md

---

## Notes

- Model di-load sekali saat startup, bukan per-request
- Input di-validate ketat via Pydantic sebelum masuk ke model
- Semua error di-handle dan mengembalikan pesan yang informatif