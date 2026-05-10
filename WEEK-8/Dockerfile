# ── BASE IMAGE ─────────────────────────────────────────────
# python:3.11-slim bukan python:3.11
# Alasannya sudah dipahami di W09D05: selisih ~870MB
# Slim = Debian minimal, cukup untuk run Python app
FROM python:3.11-slim

# ── WORKING DIRECTORY ──────────────────────────────────────
# Semua perintah setelah ini dieksekusi relatif ke /app
# Best practice: jangan kerja di root (/)
WORKDIR /app

# ── INSTALL SYSTEM DEPENDENCIES ────────────────────────────
# Kalau app butuh library C (misal: psycopg2, atau beberapa
# sklearn dependencies), install di sini.
# WAJIB: apt-get clean DAN rm -rf /var/lib/apt/lists/*
# harus dalam 1 RUN yang sama — W09D05 sudah dibahas kenapa.
# Kalau tidak butuh system deps, hapus blok ini seluruhnya.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── COPY REQUIREMENTS DULU — LAYER CACHING ─────────────────
# Ini adalah pola paling penting untuk efisiensi build.
# Kalau hanya kode app yang berubah (bukan requirements.txt),
# layer install pip ini di-skip — langsung pakai cache.
# Tapi kalau COPY . . dulu, setiap perubahan kode apapun
# akan trigger reinstall semua dependency.
COPY requirements.txt .

# ── INSTALL PYTHON DEPENDENCIES ────────────────────────────
# --no-cache-dir: buang cache pip setelah install
# Tidak ada gunanya simpan cache di dalam image — dead weight.
# --upgrade pip: pastikan pip versi terbaru sebelum install
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── COPY SOURCE CODE ───────────────────────────────────────
# Baru sekarang copy semua kode.
# Perubahan kode tidak trigger reinstall dependency (lihat atas).
# File yang tidak ingin di-copy ditangani oleh .dockerignore
COPY . .

# ── EXPOSE PORT ────────────────────────────────────────────
# Deklarasi informatif — port mana yang app listen.
# Tidak otomatis publish port; tetap perlu -p 8000:8000 saat run.
EXPOSE 8000

# ── CMD: EXEC FORM, BUKAN SHELL FORM ───────────────────────
# ["uvicorn", ...] = exec form → proses berjalan langsung
#   sebagai PID 1, signal (SIGTERM) diterima dengan benar.
# "uvicorn main:app" = shell form → bungkus dalam /bin/sh -c,
#   signal tidak diteruskan dengan baik ke aplikasi.
# --host 0.0.0.0: wajib agar container bisa diakses dari luar
# --port 8000: eksplisit, tidak bergantung default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]