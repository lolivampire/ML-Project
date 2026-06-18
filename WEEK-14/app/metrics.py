#app/metrics.py
"""
Mndefinisikan instrumen pengukuran metrik kustom (Counter, Histogram, dan Gauge) secara terpusat.
Dan juga menambahkan satu Histogram khusus untuk middleware.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary


# 1. Counter: Total prediksi
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total request prediksi yang diproses",
    ["model_name", "status"]
)

# 2. Histogram: Durasi komputasi ML spesifik
# atur bucket-nya menyesuaikan ekspektasi waktu eksekusi model (dalam detik)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Waktu komputasi yang dibutuhkan model machine learning",
    ["model_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 30.0]
)

# 3. Gauge: Status keaktifan model
MODEL_LOADED_STATUS = Gauge(
    "model_status",
    "Status keaktifan model machine learning",
    ["model_name"]
)

# 4. Histogram: Durasi endpoint HTTP (Untuk Middleware)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Durasi HTTP request dari masuk hingga keluar",
    ["method", "path", "status_code"]
)