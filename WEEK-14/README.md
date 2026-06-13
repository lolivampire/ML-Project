# Week 14 — Database Integration with SQLAlchemy & FastAPI

> **ML Engineer Journey · Phase 4: Database & Observability**
> 9–13 Jun 2026 · 6 sessions

---

## Overview

Week 14 adalah minggu di mana semua potongan backend dari phase sebelumnya akhirnya punya fondasi permanen — **database**. Dimulai dari mendefinisikan schema sebagai Python class (ORM), menjalankan query CRUD, mengelola perubahan schema dengan migration, mengintegrasikan semuanya ke FastAPI, hingga memigrasikan ke async I/O untuk concurrency production-grade. Puncaknya: satu backend utuh berlapis dengan separation of concerns yang ketat.

```
PostgreSQL ← Alembic migrations ← SQLAlchemy ORM ← FastAPI (async) ← Layered Architecture
```

---

## Tech Stack

| Dependency | Versi | Fungsi |
|---|---|---|
| `fastapi` | latest | Web framework |
| `sqlalchemy` | 2.x | ORM + async session |
| `asyncpg` | latest | Async PostgreSQL driver |
| `alembic` | latest | Database migration |
| `pydantic-settings` | latest | Environment config |
| `uvicorn` | latest | ASGI server |
| `pytest` + `pytest-asyncio` | latest | Testing |

---

## Project Structure

```
WEEK-14/
├── main.py                         # Entry point: lifespan, app init, router register
├── .env                            # DATABASE_URL, APP_NAME, DEBUG
├── requirements.txt
├── Dockerfile
├── alembic.ini
├── alembic/
│   ├── env.py                      # Alembic config — target_metadata wajib diset
│   └── versions/                   # Migration scripts (auto-generated)
├── app/
│   ├── config.py                   # Pydantic BaseSettings — baca .env otomatis
│   ├── database.py                 # Async engine + AsyncSessionLocal + get_db()
│   ├── exceptions.py               # NotFoundException, ValidationException
│   ├── models/
│   │   ├── __init__.py             # Import semua model agar Alembic detect
│   │   └── analysis.py             # SQLAlchemy ORM model
│   ├── schemas/
│   │   └── analysis.py             # Pydantic Request/Response schemas
│   ├── repositories/
│   │   └── analysis_repo.py        # Semua query DB — CRUD only, no business logic
│   ├── services/
│   │   └── analysis_service.py     # Business logic, domain validation
│   ├── routers/
│   │   └── predictions.py          # HTTP endpoints — routing only
│   └── scripts/                    # CLI scripts (pakai DatabaseManager sync)
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Fixtures: test DB, async client
    └── test_routers.py             # End-to-end endpoint tests
```

---

## Sessions

### W14D01 — SQLAlchemy ORM: Model Definition

Schema PostgreSQL sebagai blueprint gedung: tabel adalah ruangan, kolom adalah dinding, foreign key adalah koridor. SQLAlchemy ORM menerjemahkan blueprint itu ke Python class.

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, DateTime, func
import uuid

class Base(DeclarativeBase):
    pass

class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

**Konsep kunci:**
- `Mapped[T]` = type hint sekaligus deklarasi kolom (SQLAlchemy 2.x)
- `server_default=func.now()` = nilai default di-generate **di Postgres**, bukan Python
- `relationship()` + `lazy="selectin"` = eager load otomatis, mencegah `DetachedInstanceError`

---

### W14D02 — CRUD Operations via ORM

Session SQLAlchemy seperti buku agenda dan database seperti lemari arsip. Kamu tulis di agenda dulu (`add()`), kirim draft ke kasir (`flush()`), baru cap stempel permanen (`commit()`).

**State lifecycle object:**

| State | Cara masuk | Ada di DB? |
|---|---|---|
| Transient | `obj = Model(...)` | Tidak |
| Pending | `session.add(obj)` | Tidak |
| Persistent | `flush()` / `commit()` | Ya |
| Detached | `session.close()` | Ya (tapi stale) |

**Jebakan kritis:** `model_dump(exclude_unset=True)` wajib untuk PATCH — tanpa ini, field yang tidak dikirim client akan di-overwrite dengan `None`.

---

### W14D03 — Alembic: Database Migration

Alembic adalah Git-nya schema database. Setiap perubahan schema (tambah kolom, ubah tipe, hapus tabel) menjadi migration file yang bisa di-commit, di-review, dan dijalankan ulang di environment manapun.

```bash
# Inisialisasi
alembic init alembic

# Generate migration dari perubahan model
alembic revision --autogenerate -m "add analysis table"

# Apply ke database
alembic upgrade head

# Cek posisi saat ini
alembic current

# Rollback 1 step
alembic downgrade -1
```

**Konsep kunci:**
- Alembic menyimpan state di tabel `alembic_version` di dalam DB sendiri
- `env.py` wajib punya `target_metadata = Base.metadata` agar autogenerate berfungsi
- Import semua model di `models/__init__.py` agar semua tabel ter-detect

---

### W14D04 — FastAPI + SQLAlchemy Integration

Session-per-request pattern: session hidup hanya selama satu HTTP request cycle. Dibuat saat request masuk, di-inject via `Depends(get_db)`, ditutup di `finally` setelah response dikirim.

```python
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db       # inject ke endpoint
    finally:
        db.close()     # SELALU tutup — finally absolut, bukan hanya saat error
```

**Kenapa `finally` bukan `except`?** `except` hanya jalan saat error. Kalau request sukses 100%, `except` tidak pernah dipanggil → `db.close()` tidak jalan → connection leak. `finally` absolut — jalan baik saat sukses maupun error.

---

### W14D05 — Async SQLAlchemy & Connection Pooling

Analogi pelayan restoran: sync = berdiri diam di dapur menunggu masak selesai sebelum layani meja lain. Async = titip pesanan ke dapur, langsung layani meja berikutnya, kembali saat dapur panggil "siap!".

```python
# Async engine — URL prefix wajib berubah
create_async_engine("postgresql+asyncpg://...", pool_size=10, max_overflow=20)

# Query pattern berubah total
result = await db.execute(select(Model))   # bukan db.query(Model)
items  = result.scalars().all()

# Session wajib expire_on_commit=False
AsyncSession(engine, expire_on_commit=False)  # lazy load di async = MissingGreenlet error
```

**Connection Pool formula:**
```
Total koneksi ke DB = (pool_size + max_overflow) × jumlah instance
```
Hitung sebelum deploy. PostgreSQL default max_connections = 100.

| Setting | Fungsi | Rekomendasi |
|---|---|---|
| `pool_size` | Koneksi permanen siaga | 5–20 |
| `max_overflow` | Koneksi darurat sementara | 10–30 |
| `pool_pre_ping=True` | Test koneksi sebelum pakai | **Wajib di prod** |
| `pool_recycle` | Recycle koneksi setelah N detik | 1800 (30 menit) |

---

### W14D06 — Project 3 Backend: Layered Architecture

Semua fondasi dirakit jadi satu sistem production-grade dengan separation of concerns ketat.

**Request flow:**
```
HTTP Request → Router → Service → Repository → Database
                                                    ↓
HTTP Response ← Router ← Service ← Repository ←───┘
```

**Tanggung jawab tiap layer:**

| Layer | Boleh | TIDAK Boleh |
|---|---|---|
| Router | HTTP routing, status code, response format | Business logic, query DB |
| Service | Business rules, domain validation, raise exceptions | Query DB langsung |
| Repository | Query DB, CRUD, filter | HTTPException, business rules |
| Model | Definisi tabel ORM | Logic apapun |
| Schema | Validasi I/O Pydantic | ORM, DB logic |

---

## Core Patterns — Quick Reference

### Async CRUD Pattern

```python
# CREATE
obj = Model(**data)
db.add(obj)
await db.flush()       # generate UUID di Postgres — SEBELUM refresh
await db.refresh(obj)  # sync Python object dengan nilai aktual DB

# READ ALL
result = await db.execute(select(Model))
items  = result.scalars().all()

# READ BY PK (efisien — cek identity map dulu)
obj = await db.get(Model, pk)

# UPDATE (via object tracking)
for key, val in patch_data.items():
    setattr(obj, key, val)
await db.flush()
await db.refresh(obj)

# DELETE
await db.delete(obj)
await db.flush()
```

### Async get_db() dengan commit/rollback

```python
async def get_db():
    async with AsyncSessionLocal() as db:
        try:
            yield db
            await db.commit()     # commit hanya jika tidak ada exception
        except:
            await db.rollback()   # rollback di except, BUKAN finally
            raise
```

**Kenapa rollback di `except`, bukan `finally`?** `finally` selalu jalan — termasuk saat sukses. Rollback setelah commit sukses akan membatalkan transaksi yang baru saja berhasil. `except` hanya jalan saat error — tepat kapan rollback diperlukan.

### Dynamic Filter Pattern

```python
query = select(Analysis)
if name:
    query = query.where(Analysis.name.ilike(f"%{name}%"))
if min_score is not None:
    query = query.where(Analysis.score >= min_score)
result = await db.execute(query)
```

---

## Environment Variables

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname
APP_NAME=Project 3 API
DEBUG=false
```

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Setup database (pastikan PostgreSQL running)
alembic upgrade head

# Run server
uvicorn main:app --reload

# Run tests
pytest tests/ -v
```

---

## Test Results

```
platform win32 -- Python 3.14.3, pytest-9.0.3
asyncio: mode=Mode.STRICT

tests/test_routers.py::test_create_and_get_prediction         PASSED
tests/test_routers.py::test_create_prediction_validation_error PASSED

2 passed in 15.21s
```

---

## API Endpoints

| Method | Endpoint | Description | Status |
|---|---|---|---|
| `GET` | `/health` | Health check | 200 |
| `GET` | `/analyses/` | List all analyses | 200 |
| `GET` | `/analyses/{id}` | Get by ID | 200 / 404 |
| `POST` | `/analyses/` | Create new | 201 |
| `PATCH` | `/analyses/{id}` | Partial update | 200 / 404 |
| `DELETE` | `/analyses/{id}` | Delete | 204 / 404 |

Swagger UI tersedia di `/docs` saat server berjalan.

---

## Key Concepts Mastered

- **ORM as blueprint** — tabel → class, baris → instance, FK → relationship
- **flush() → refresh() order** — flush() dulu agar UUID ter-generate Postgres, baru refresh() sync
- **finally vs except** — `finally` untuk `db.close()`, `except` untuk `rollback()`
- **expire_on_commit=False** — wajib di AsyncSession, lazy load setelah commit = MissingGreenlet
- **exclude_unset=True** — PATCH semantics: hanya field yang dikirim client yang diproses
- **get_by_id() sebelum update** — validasi keberadaan + cara kerja ORM object tracking
- **pool_pre_ping=True** — wajib di production, cegah stale connection
- **Separation of concerns** — setiap layer punya satu tanggung jawab, tidak boleh dilanggar

---

*Week 14 complete. Next: Week 15 — Observability, Structured Logging & Monitoring.*