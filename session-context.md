# 📍 Session Context — ML Engineer Journey

## Status Terakhir
- **Week**: 10
- **Day**: 03
- **Phase**: 3 — Docker & Clean Architecture
- **Tanggal**: 13 Mei 2026

## ✅ Yang Sudah Selesai
- W10D01: docker-compose.yml fundamentals untuk Risk Scoring API ✅
- W10D02: Multi-service Compose + Networking (api + redis) ✅
- W10D03: Volume Mounting & Data Persistence ✅

## 🧠 Yang Sudah Dipahami
- Compose membuat default network otomatis — service name = DNS hostname
- Dua compose file = dua network terpisah = tidak bisa saling resolve via nama
- Bind mount (./logs) = akses langsung host, cocok dev
- Named volume (api-logs) = Docker-managed, portable, cocok production
- restart: unless-stopped vs always — beda perilaku setelah manual stop + machine restart
- docker-compose down vs down -v — volume aman vs volume terhapus
- Path(__file__).resolve() — path absolut yang tidak peduli CWD, robust di semua environment
- Docker embedded DNS (127.0.0.11) me-resolve service name ke IP internal container
- REDIS_HOST=redis — service name sebagai hostname, bukan hardcoded IP
- depends_on hanya menjamin container started, bukan healthy (readiness = W10D05)
- redis:7-alpine ~30MB vs redis:7 ~110MB — image optimization langsung applicable
- --appendonly yes — AOF persistence: data survive container restart
- --remove-orphans — bersihkan container sisa service yang sudah dihapus dari Compose
- os.getenv fallback pattern — satu kode bisa jalan di Docker maupun lokal
- Bind mount tidak terpengaruh docker-compose down -v — host filesystem di luar wewenang Docker
- Named volume BISA dihapus: down -v, docker volume rm, docker system prune --volumes
- Docker Compose prefix volume name dengan nama project (folder name by default)
- name: di docker-compose.yml = project name explicit, volume name tidak berubah meski folder direname
- AOF persistence perlu diaktifkan eksplisit — tanpa ini Redis bisa kehilangan data saat crash
- Single source of truth pattern untuk test data (DRY) — write dan read pakai dict yang sama
- argparse lebih proper dari sys.argv manual untuk CLI script
- Value mismatch check — verifikasi tidak cukup cek key ada, tapi juga cek value tidak corrupt
- ports di Redis sebaiknya dihapus di production — unnecessary attack surface
- Backup named volume: docker run --rm -v nama:/data -v $(pwd):/backup alpine tar czf ...

## ⚠️ Yang Masih Blur
-

## 📁 Output Files
- `week-10/docker-compose.yml` (multi-service + volumes) ✅
- `week-10/scripts/verify_persistence.py` ✅
- `W10D03_Volume_Mounting_Data_Persistence.pdf` ✅

## ❓ Pertanyaan Pending
-

## 📌 Next Session
- **Week**: 10
- **Day**: 04
- **Topik**: Environment Variables & Secrets Management
- **Preview**: Pisahkan konfigurasi dev dan production menggunakan
  environment variables yang proper — .env files, docker-compose override,
  dan cara handling secrets agar tidak pernah masuk ke image atau Git history