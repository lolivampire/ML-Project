# ================================================================
# STAGE 1: builder
# Tujuan: install semua dependencies, compile apa yang perlu
# Stage ini TIDAK akan masuk ke production image
# ================================================================
FROM python:3.11-slim AS builder

# Install build dependencies — hanya dibutuhkan untuk compile
# packages seperti numpy, pandas, scikit-learn yang punya C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements dulu — manfaatkan layer caching (W09D04)
# Jika requirements.txt tidak berubah, layer ini di-cache
COPY requirements.txt .

# Install ke folder khusus: --prefix=/install
# Semua package pergi ke /install/lib/python3.11/site-packages/
# Ini yang akan kita copy ke production stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ================================================================
# STAGE 2: production
# Tujuan: image final yang bersih dan ringan
# Hanya berisi: Python runtime + compiled packages + app code
# ================================================================
FROM python:3.11-slim AS production

# Set environment variables — best practice untuk Python di Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy HANYA hasil install dari builder stage
# gcc, build tools, cache — semua tertinggal di builder, tidak ikut
COPY --from=builder /install /usr/local

# Copy source code aplikasi
# Ini di layer paling bawah supaya rebuild cepat saat kode berubah
COPY app/ ./app/

# Jalankan sebagai non-root user — security best practice
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Gunakan exec form (array), bukan shell form
# Exec form: signal langsung ke uvicorn (bisa Ctrl+C dengan benar)
# Shell form: signal ke shell dulu, bisa ada delay
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]