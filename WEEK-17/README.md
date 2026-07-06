# Week 17 — System Design & Scale

> Phase 5: System Design & Scale | ML Engineer Journey — 24 Weeks
> 6 hari: Scaling → Load Balancing → Database Bottleneck → CAP Theorem → Instagram Case Study → Review & Compile

##  Ringkasan Minggu Ini

Minggu ini bukan lima topik yang berdiri sendiri — ini satu rantai evolusi sistem yang berurutan. Setiap solusi pada satu hari memunculkan masalah baru yang diselesaikan di hari berikutnya:

```
Server tunggal kehabisan kapasitas
        │
        ▼
Scaling (vertical vs horizontal)
        │
        ▼
Load Balancer (distribusi trafik ke banyak server)
        │
        ▼
Database Bottleneck (semua server menulis ke 1 DB yang sama)
        │
        ▼
CAP Theorem (partition antar node hidup → pilih CP atau AP)
        │
        ▼
Hybrid Fan-out (studi kasus Instagram-scale)
```

Dokumen referensi lengkap (analogi, tabel perbandingan, diagram, kode, key takeaways) tersedia di [`W17D06_System_Design_Reference.pdf`](./W17D06_System_Design_Reference.pdf).

---

##  Breakdown Harian

| Hari | Topik | Goal |
|---|---|---|
| **D01** | Scaling: vertical vs horizontal | Memahami trade-off scale up vs scale out dengan contoh nyata |
| **D02** | Load balancing: konsep & algoritma | Round-robin, least-connection, IP-hash — kapan masing-masing dipakai |
| **D03** | Database bottleneck patterns | N+1 query, connection pool exhaustion, slow query |
| **D04** | CAP theorem — intuisi, bukan hafalan | Consistency vs Availability saat partition terjadi |
| **D05** | System design case study: Instagram-scale | Rekonstruksi arsitektur nyata + hybrid fan-out |
| **D06** | System design notes & diagram (review day) | Compile seluruh materi jadi 1 dokumen referensi interview-prep |

---

## 🧠 Konsep Kunci

### 1. Scaling
- **Vertical scaling** — tambah kapasitas satu mesin (CPU/RAM). Sederhana, tapi ada batas fisik hardware.
- **Horizontal scaling** — tambah jumlah instance/mesin. Nyaris tak terbatas, tapi butuh koordinasi (load balancer, state management).

### 2. Load Balancing
| Algoritma | Cara Kerja | Cocok Untuk |
|---|---|---|
| Round-robin | Giliran bergilir antar server | Kapasitas server seragam |
| Least-connection | Ke server dengan koneksi aktif paling sedikit | Durasi request tidak seragam |
| IP-hash | Hash IP client → server yang sama tiap kali | Butuh session affinity (sticky session) |

### 3. Database Bottleneck Patterns
- **N+1 query** — satu query awal memicu N query tambahan per baris, alih-alih satu query gabungan (JOIN).
- **Connection pool exhaustion** — koneksi ke DB habis karena request simultan melebihi kapasitas pool.
- **Solusi umum** — indexing, connection pooling, caching (Redis), read replica.

### 4. CAP Theorem
Sistem terdistribusi hanya bisa menjamin dua dari tiga: **C**onsistency, **A**vailability, **P**artition tolerance. Karena partition tidak bisa dihindari sepenuhnya, pilihan nyata ada di antara **CP** dan **AP**.

| Pilihan | Sikap Sistem Saat Partition | Contoh |
|---|---|---|
| CP | Tolak layani sampai data tersinkron | Autentikasi, transaksi finansial |
| AP | Tetap layani meski data sedikit basi | Cache feed sosial media |

> **Catatan presisi:** matinya load balancer (SPOF) **bukan** masalah CAP theorem. SPOF menghentikan akses sebelum request menyentuh lapisan data sama sekali — murni masalah redundansi infrastruktur, bukan dilema trade-off data.

### 5. Hybrid Fan-out (Instagram Case Study)
| Strategi | Beban Posting | Beban Baca Feed | Masalah |
|---|---|---|---|
| Push (fan-out on write) | Berat | Ringan | Celebrity problem — 1 post → jutaan write |
| Pull (fan-out on read) | Ringan | Berat | Lambat untuk user dengan banyak following |
| **Hybrid** (dipakai Instagram) | Push untuk user biasa, skip untuk selebriti | Pull khusus feed selebriti | Solusi nyata, bukan kompromi teoretis |

---

##  Kode: `FeedService`

File: [`scripts/instagram_case_study.py`](./scripts/instagram_case_study.py)

Simulasi hybrid fan-out dengan penerapan **Dependency Injection**:
- `celebrity_threshold` di-inject via constructor `FeedService`, bukan konstanta global — memungkinkan pengujian jalur "celebrity" tanpa membuat 1 juta data dummy.
- `FeedCacheRepository` dipisah sebagai dependency tersendiri dari business logic (Separation of Concerns, lanjutan dari W11).
- `User` adalah `frozen=True` dataclass — pure data carrier, tidak lagi memegang kebijakan bisnis (`is_celebrity` dipindah ke domain service).

```python
class FeedService:
    def __init__(
        self,
        cache_repo: FeedCacheRepository,
        celebrity_threshold: int = 1_000_000,
    ) -> None:
        self._cache_repo = cache_repo
        self._celebrity_threshold = celebrity_threshold

    def handle_new_post(self, author: User, follower_ids: Sequence[int]) -> str:
        if self._is_celebrity(author):
            return self._handle_celebrity_post(author, follower_ids)
        return self._handle_regular_post(author, follower_ids)
```

Jalankan:
```bash
python scripts/instagram_case_study.py
```

---

##  Struktur Folder

```
week-17/
├── README.md                              ← file ini
├── W17D06_System_Design_Reference.pdf      ← dokumen referensi lengkap untuk interview prep
└── scripts/
    └── instagram_case_study.py             ← FeedService + hybrid fan-out simulation
```

---

##  Status

- [x] Scaling vertical vs horizontal
- [x] Load balancing algorithms
- [x] Database bottleneck patterns
- [x] CAP theorem (dengan pembeda presisi vs SPOF)
- [x] Instagram case study + FeedService implementation
- [x] Refactor FeedService dengan Dependency Injection
- [x] Dokumen referensi Week 17 di-compile

**Next:** Week 18 — Caching & Performance (Redis, cache-aside pattern, API response caching)

---
*ML Engineer Journey — [github.com/lolivampire/ML-Project](https://github.com/lolivampire/ML-Project)*