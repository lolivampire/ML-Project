# Gunakan image Python resmi versi ringan
FROM python:3.11-slim

# Mencegah Python menulis file .pyc ke disk (menghemat ruang)
ENV PYTHONDONTWRITEBYTECODE=1
# Mencegah Python mem-buffer stdout dan stderr (log langsung muncul di terminal)
ENV PYTHONUNBUFFERED=1

# Tentukan direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements terlebih dahulu untuk memanfaatkan layer caching Docker
COPY requirements.txt .

# Install dependencies tanpa menyimpan cache instalasi
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam direktori kerja kontainer
COPY . .

# Ekspos port yang akan digunakan oleh FastAPI
EXPOSE 8000

# Perintah utama untuk menyalakan server menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]