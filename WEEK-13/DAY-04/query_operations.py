"""
DAY-04/query_operations.py
W13D04 — SQL DQL: SELECT, JOIN, GROUP BY, dan Analisis Performa

Pola penarikan data menggunakan psycopg2 dengan auto-deserialization.
"""

import psycopg2
from psycopg2.extras import register_json

# ── CONFIG ──────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mldb",
    "user": "mluser",
    "password": "mlpassword"
}

# ── 1. SELECT DENGAN FILTER & LIMIT ──────────────────────────────────────────

def get_top_confident_predictions(conn) -> list[dict]:
    """
    Mengambil prediksi, urutkan dari confidence tertinggi ke terendah.
    Membatasi hanya 10 data teratas (Top 10).
    """
    sql = """
        SELECT id, prediction, confidence, created_at
        FROM predictions
        WHERE confidence IS NOT NULL
        ORDER BY confidence DESC
        LIMIT 10;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        
    return [
        {"prediction_id": str(r[0]), "prediction": r[1], "confidence": float(r[2]), "created_at": r[3].isoformat()}
        for r in rows
    ]

# ── 2. INNER JOIN (Hubungan Pasti Induk-Anak) ─────────────────────────────────
def get_high_confidence_prediction(conn, min_confidence: float) -> list[dict]:
    """
    Menggabungkan tabel models dan predictions.
    Hanya mengambil data yang memiliki pasangan (Inner Join) dan lolos filter.
    """
    sql = """
        SELECT models.name, predictions.prediction, predictions.confidence
        FROM models
        INNER JOIN predictions
        ON models.id = predictions.model_id
        WHERE predictions.confidence > %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (min_confidence,))
        rows = cur.fetchall()

    return [
        {"model_name": r[0], 
         "prediction": r[1], 
         "confidence": float(r[2])}
        for r in rows
    ]

# ── 3. LEFT JOIN + GROUP BY (Analisis Aggregasi Komprehensif) ─────────────────
def get_model_performance_summary(conn) -> list[dict]:
    """
    Menampilkan SEMUA model, termasuk yang belum memiliki prediksi (Left Join).
    Menghitung total prediksi dan rata-rata confidence score per model.
    """
    sql = """
        SELECT 
            m.name, 
            m.version, 
            COUNT(p.id) AS total_predictions,
            AVG(p.confidence) AS avg_confidence
        FROM models m
        LEFT JOIN predictions p ON m.id = p.model_id
        GROUP BY m.id, m.name, m.version
        ORDER BY total_predictions DESC;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    return [
        {
            "model_name": r[0],
            "model_version": r[1],
            "total_predictions": r[2],
            "avg_confidence": float(r[3]) if r[3] is not None else None
        }
        for r in rows
    ]

# ── 4. GROUP BY + HAVING (Truncate Tanggal & Filter Agregat) ──────────────────
def get_high_volumes_days(conn, min_volume: int = 3) -> list[dict]:
    """
    Menghitung total prediksi per model_id per hari.
    Fungsi HAVING menyaring hasil setelah aggregasi selesai dihitung.
    """
    sql = """
        SELECT 
            model_id, 
            DATE(created_at) AS prediction_date, 
            COUNT(id) AS daily_count
        FROM predictions
        GROUP BY model_id, DATE(created_at)
        HAVING COUNT(id) >= %s
        ORDER BY prediction_date DESC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (min_volume,))
        rows = cur.fetchall()

    return [
        {
        "model_id": r[0], 
        "date": str(r[1]), 
        "total_predictions": r[2]
         }
        for r in rows
    ]

def run_performance_diagnostics(conn) -> None:
    with conn.cursor() as cur:
        # Buat index dulu
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_confidence
            ON predictions(confidence);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_model_id
            ON predictions(model_id);
        """)
        conn.commit()
        print("Index created.\n")

        # Update statistik tabel agar planner lebih akurat
        cur.execute("ANALYZE predictions;")
        conn.commit()

        # Jalankan EXPLAIN ANALYZE setelah index ada
        sql = """
            EXPLAIN ANALYZE
            SELECT m.name, p.prediction, p.confidence, p.created_at
            FROM predictions p
            INNER JOIN models m ON p.model_id = m.id
            WHERE p.confidence > 0.8;
        """
        cur.execute(sql)
        plan = cur.fetchall()
        for line in plan:
            print(line[0])

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main()-> None:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_json(conn) # Aktifkan auto-deserialize agar output JSON aman
        print("=============================================")
        print("SQL DQL OPERATIONAL TESTING")
        print("==================================================\n")
        # Test 1: Select + Filter
        print("[1] TESTING: SELECT WITH FILTER & LIMIT 10")
        print(get_top_confident_predictions(conn), "\n")
        # Test 2: Inner Join
        print("[2] TESTING: INNER JOIN (confidence > 0.8)")
        print(get_high_confidence_prediction(conn, 0.8), "\n")
        # Test 3: Left Join + Group By
        print("[3] TESTING: LEFT JOIN + GROUP BY (Analisis Aggregasi Komprehensif)")
        print(get_model_performance_summary(conn), "\n")
        # Test 4: Group By + Having
        print("[4] TESTING: GROUP BY + HAVING (Truncate Tanggal & Filter Agregat)")
        print(get_high_volumes_days(conn,3), "\n")
        # Test 5: Performance Diagnostics
        print("[5] TESTING: PERFORMANCE DIAGNOSTICS")
        run_performance_diagnostics(conn)
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    main()