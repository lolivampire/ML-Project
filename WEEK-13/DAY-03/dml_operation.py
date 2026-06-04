"""
week-13/day-03/dml_operations.py
W13D03 — SQL DML dengan Auto-Serialization JSON

Menggunakan psycopg2.extras.register_json untuk otomatisasi
konversi Python dict -> PostgreSQL JSONB.
"""

import psycopg2
from psycopg2.extras import register_json, Json



# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mldb",
    "user": "mluser",
    "password": "mlpassword"
}


# ── INSERT ─────────────────────────────────────────────────────────────────────

def insert_prediction(conn, model_name: str, model_version: str,
                      input_data: dict, result: str,
                      confidence: float) -> dict:
    """
    Insert satu prediction. input_data di-pass sebagai DICT MENTAH.
    psycopg2 akan otomatis mengonversinya ke string JSON di latar belakang.
    """
    with conn.cursor() as cur:
        # 1. Cari atau buat model_id di tabel models
        cur.execute("""
            INSERT INTO models (name, version) 
            VALUES (%s, %s)
            ON CONFLICT (name, version) 
            DO UPDATE SET name = EXCLUDED.name
            RETURNING id;
        """, (model_name, model_version))
        model_id = cur.fetchone()[0]

        # 2. INSERT ke tabel predictions (Tanpa json.dumps!)
        sql = """
            INSERT INTO predictions
                (model_id, input_data, prediction, confidence, latency_ms)
            VALUES
                (%s, %s, %s, %s, %s)
            RETURNING id, created_at
        """
        # input_data langsung dimasukkan sebagai objek dict Python
        params = (model_id, Json(input_data), result, confidence, 45)
        
        cur.execute(sql, params)
        row = cur.fetchone()
        conn.commit()
        
    return {"id": str(row[0]), "created_at": row[1].isoformat()}


# ── UPDATE ─────────────────────────────────────────────────────────────────────

def update_confidence(conn, prediction_id: str, new_confidence: float) -> dict | None:
    sql = """
        UPDATE predictions
        SET confidence = %s
        WHERE id = %s
        RETURNING id, confidence
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_confidence, prediction_id))
        row = cur.fetchone()
        conn.commit()
    
    if row is None:
        return None
    
    return {"id": str(row[0]), "confidence": float(row[1])}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # KUNCI UTAMA: Daftarkan auto-serializer JSON ke koneksi ini
        register_json(conn)
        
        print("=============================================")
        print("Connected to PostgreSQL (Auto-Serialize Active)")
        print("=============================================\n")
        
        # Jalankan pengujian
        result = insert_prediction(
            conn,
            model_name="risk_scorer",
            model_version="2.0.0",
            input_data={"age": 35, "income": 45000, "loan_amount": 12000}, # Tetap Dict
            result="medium_risk",
            confidence=0.74
        )
        print(f"INSERT result: {result}")
        
        updated = update_confidence(conn, result["id"], 0.81)
        print(f"UPDATE result: {updated}")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("\nConnection closed.")


if __name__ == "__main__":
    main()