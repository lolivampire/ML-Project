# Week 06 — Integrasikan ML Model ke FastAPI

**Phase 2: API & Deploy** · 6 hari · Difficulty ★★☆

---

## Tujuan Minggu Ini

Membangun ML API yang production-ready: model ter-serialize dengan benar, ter-load saat startup, preprocessing berjalan dalam pipeline, setiap request ter-log secara terstruktur, dan semua edge case ditolak dengan error yang jelas — bukan crash jadi 500.

---

## Output

| File | Deskripsi |
|------|-----------|
| `app/main.py` | Entry point FastAPI: lifespan startup, global exception handlers |
| `app/routers/predict.py` | Endpoint `/predict` dan `/predict/health` |
| `app/schemas/predict.py` | Pydantic schema: `PredictionRequest`, `PredictionResponse` |
| `app/services/model_loader.py` | Load model sekali saat startup, simpan di cache |
| `models/pipeline.joblib` | Sklearn Pipeline (preprocessing + model) yang ter-serialize |
| `tests/test_api.py` | Integration test suite: 8 skenario, httpx client |

---

## Struktur Project

```
week-06/
├── app/
│   ├── main.py
│   ├── routers/
│   │   └── predict.py
│   ├── schemas/
│   │   └── predict.py
│   └── services/
│       └── model_loader.py
├── models/
│   └── pipeline.joblib
├── tests/
│   └── test_api.py
└── requirements.txt
```

---

## Cara Menjalankan

```bash
# 1. Aktifkan virtual environment
venv\Scripts\Activate.ps1        # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train dan simpan model (jika belum ada)
python train_model.py

# 4. Jalankan server
uvicorn app.main:app --reload

# 5. Jalankan test suite (terminal terpisah)
python tests/test_api.py
```

Swagger UI tersedia di: `http://127.0.0.1:8000/docs`

---

## Endpoint

### `GET /predict/health`

Mengecek status server dan model.

**Response 200:**
```json
{
  "status": "ready",
  "n_features": 4,
  "classes": [0, 1],
  "trained_at": "2026-04-15T16:53:43.863401",
  "version": "pipe_v1"
}
```

---

### `POST /predict/`

Menerima array fitur numerik, mengembalikan hasil prediksi model.

**Request body:**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response 200:**
```json
{
  "prediction": 1,
  "probability": 0.9585,
  "input_received": [5.1, 3.5, 1.4, 0.2]
}
```

**Error responses:**

| Status | Kondisi |
|--------|---------|
| `422` | Schema tidak valid: tipe salah, field hilang, array kosong, NaN, Infinity |
| `400` | Nilai tidak wajar: negatif pada fitur 0/1, jumlah fitur tidak sesuai |
| `503` | Model belum ter-load saat request masuk |
| `500` | Internal server error (detail tidak di-expose ke client) |

---

## Konsep Kunci

### Dua Lapisan Validasi

```
Request body
     │
     ▼
┌─────────────┐     gagal → 422 (Unprocessable Entity)
│   Pydantic  │     — tipe salah, field hilang
│  Validation │     — array kosong (min_length=1)
│             │     — NaN/Infinity (@field_validator)
└──────┬──────┘
       │ lolos
       ▼
┌─────────────┐     gagal → 400 (Bad Request)
│   Domain    │     — nilai negatif pada fitur usia/pendapatan
│    Logic    │     — jumlah fitur tidak sesuai ekspektasi model
└──────┬──────┘
       │ lolos
       ▼
  Preprocessing → Predict → Response 200
```

**Filosofi pemisahan:** 422 berarti *"perbaiki format request"* — kesalahan di level schema. 400 berarti *"perbaiki nilai yang dikirim"* — kesalahan di level bisnis/domain. Keduanya adalah kesalahan client, bukan server error.

---

### Model Loading via Lifespan

Model di-load **sekali** saat server startup menggunakan `@asynccontextmanager lifespan`, bukan di dalam endpoint. Ini mencegah overhead load model di setiap request.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()    # jalankan sekali saat startup
    yield           # server berjalan di sini
```

Jika model `None` saat endpoint dipanggil → return `503 Service Unavailable`, bukan biarkan crash jadi `500`.

---

### jsonable_encoder untuk Pydantic Errors

`exc.errors()` dari Pydantic bisa mengandung objek `ValueError` di dalam `ctx["error"]` — tidak bisa di-serialize langsung ke JSON. `jsonable_encoder()` menelusuri struktur secara rekursif dan mengkonversi objek non-serializable ke string yang aman.

```python
# SALAH — bisa throw TypeError saat NaN ditangkap @field_validator
errors = exc.errors()

# BENAR
from fastapi.encoders import jsonable_encoder
errors = jsonable_encoder(exc.errors())
```

---

### Urutan except yang Benar

```python
try:
    pred = pipe.predict(X)
except HTTPException:
    raise                    # WAJIB di atas — jangan flatten 4xx ke 500
except Exception as e:
    logger.error(str(e))
    raise HTTPException(500, "Internal server error")
```

---

### time.perf_counter() di Luar try

```python
t0 = time.perf_counter()    # di luar try — guaranteed ada di memori saat exception
try:
    ...
except Exception:
    ...
latency = time.perf_counter() - t0
```

---

## Test Suite

`tests/test_api.py` menjalankan 8 skenario integration test:

```
✓ Health check                          GET  /predict/health      → 200
✓ Happy path — input valid              POST /predict/            → 200
✓ Empty features                        POST /predict/            → 422
✓ Wrong type (string in array)          POST /predict/            → 422
✓ Missing required field                POST /predict/            → 422
✓ NaN in features (field_validator)     POST /predict/            → 422
✓ Negative value (domain logic)         POST /predict/            → 400
✓ Wrong feature count (domain logic)    POST /predict/            → 400

Hasil: 8/8 pass
```

---

## Dependencies

```
fastapi
uvicorn[standard]
scikit-learn
joblib
httpx
pydantic
python-multipart
```

---

## Topik Harian

| Hari | Topik | Output |
|------|-------|--------|
| D01 | Model serialization: joblib & pickle | `models/pipeline.joblib` |
| D02 | Load model saat startup (lifespan) | `app/services/model_loader.py` |
| D03 | Preprocessing pipeline dalam API | Pipeline sklearn di dalam joblib |
| D04 | Structured logging (bukan print) | JSON log per request: input, output, latency |
| D05 | Input validation & edge cases | Dua lapisan: Pydantic (422) + domain logic (400) |
| D06 | ML API local testing | `tests/test_api.py` — 8/8 pass |

---

## Selanjutnya

**Week 07 — Environment Management & Clean Structure**

Virtual environment & pip · Environment variables & `.env` · Folder structure MVC-inspired · API documentation & Swagger · README profesional · Review + polish Project 1 prep.

---

*ML Engineer Journey — Phase 2: API & Deploy*
*github.com/lolivampire/ML-Project*