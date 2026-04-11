# Week 05 — FastAPI Fundamentals

**Phase 2: API & Deploy** | ML Engineer Journey — 24 Weeks

---

## Gambaran Minggu Ini

Week 05 membangun fondasi pengembangan API menggunakan FastAPI. Dimulai dari setup routing dasar hingga refactor ke struktur folder yang scalable, minggu ini mencakup semua komponen inti yang dibutuhkan sebelum model ML diintegrasikan ke dalam API di Week 06.

---

## Topik Harian

| Hari | Topik | Output |
|------|-------|--------|
| D01 | FastAPI setup & routing dasar | 3 endpoint: GET, POST, DELETE |
| D02 | Pydantic schema & validation | Request/response model dengan field constraints |
| D03 | Dependency injection dasar | `Depends()` untuk DB session & config |
| D04 | HTTP status codes & error handling | `HTTPException`, custom exception handler |
| D05 | Async endpoint & background tasks | `async/await`, `asyncio.gather()`, `BackgroundTasks` |
| D06 | Structured FastAPI project | Refactor ke `routers/`, `schemas/`, `dependencies/` |

---

## Struktur Project

```
week-05/
└── app/
    ├── main.py                  # Entry point — setup, middleware, include_router
    ├── routers/
    │   ├── __init__.py
    │   ├── items.py             # Endpoint definitions untuk resource items
    │   └── health.py            # Health check endpoint
    ├── schemas/
    │   ├── __init__.py
    │   └── item.py              # Pydantic models: ItemCreate, ItemResponse
    └── dependencies/
        ├── __init__.py
        └── common.py            # Shared Depends(): verify_token, get_pagination
```

---

## Cara Menjalankan

```bash
# Dari root project
cd week-05
uvicorn app.main:app --reload
```

API tersedia di `http://localhost:8000`
Dokumentasi Swagger di `http://localhost:8000/docs`

---

## Endpoint

### Items

| Method | Path | Auth | Deskripsi |
|--------|------|------|-----------|
| `GET` | `/items/` | ✓ | List semua item (dengan pagination) |
| `GET` | `/items/{id}` | — | Ambil satu item berdasarkan ID |
| `POST` | `/items/` | ✓ | Buat item baru |
| `DELETE` | `/items/{id}` | ✓ | Hapus item |

### Health

| Method | Path | Auth | Deskripsi |
|--------|------|------|-----------|
| `GET` | `/health/` | — | Status API dan versi |

**Auth**: endpoint bertanda ✓ membutuhkan header `x-token: secret-token`

---

## Contoh Request

```bash
# Buat item baru
curl -X POST http://localhost:8000/items/ \
  -H "Content-Type: application/json" \
  -H "x-token: secret-token" \
  -d '{"name": "Laptop", "price": 15000000}'

# List item dengan pagination
curl "http://localhost:8000/items/?skip=0&limit=5" \
  -H "x-token: secret-token"

# Health check
curl http://localhost:8000/health/
```

---

## Konsep Kunci

**Pydantic validation** — `ItemCreate` dan `ItemResponse` dipisah karena user tidak boleh set `id` sendiri. `response_model` memfilter field yang tidak seharusnya keluar ke client.

**Dependency injection** — `verify_token` dan `get_pagination` ada di `dependencies/common.py` agar bisa di-inject ke router manapun tanpa duplikasi. Satu perubahan mekanisme token cukup edit satu file.

**Structured project** — `main.py` yang ideal tidak mengandung business logic apapun. Menambah resource baru cukup dua baris di `main.py`: satu import, satu `include_router`.

**Async vs background tasks** — `async def` untuk I/O non-blocking, `BackgroundTasks` untuk pekerjaan yang berjalan setelah response dikirim ke client.

---

## Scripts

| File | Deskripsi |
|------|-----------|
| `app/main.py` | Entry point FastAPI |
| `app/routers/items.py` | CRUD endpoint untuk items |
| `app/routers/health.py` | Health check endpoint |
| `app/schemas/item.py` | Pydantic request/response models |
| `app/dependencies/common.py` | Shared dependency functions |

---

## Catatan

- `__init__.py` wajib ada di setiap subfolder agar Python memperlakukan folder sebagai package yang bisa di-import
- Data disimpan di `fake_db` (dict in-memory) — akan diganti PostgreSQL di Week 13
- CORS middleware dikonfigurasi dengan `allow_origins=["*"]` untuk development; di production harus diganti dengan domain spesifik

---

*Week 05 selesai — Next: Week 06 — Integrasikan ML Model ke FastAPI*