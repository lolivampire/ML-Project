"""
test_connection.py
W13D01 — Verifikasi koneksi Python ke PostgreSQL

Menggunakan psycopg2 untuk connect dan run query sederhana.
"""

import psycopg2
from psycopg2 import OperationalError


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "superrahasia",
}


def get_connection() -> psycopg2.extensions.connection:
    """Buat koneksi ke PostgreSQL.
    
    Returns:
        psycopg2 connection object
        
    Raises:
        OperationalError: jika koneksi gagal
    """
    # psycopg2.connect() membaca dict config dan membuka TCP connection
    # ke PostgreSQL di host:port yang ditentukan
    conn = psycopg2.connect(**DB_CONFIG)
    return conn


def run_diagnostics(conn: psycopg2.extensions.connection) -> None:
    """Jalankan query diagnostik untuk verifikasi koneksi."""
    # cursor adalah 'kursor' yang menunjuk posisi eksekusi query
    # with statement otomatis menutup cursor setelah selesai
    with conn.cursor() as cur:
        
        # Query 1: verifikasi user dan database aktif
        cur.execute("SELECT current_user, current_database();")
        user, db = cur.fetchone()  # fetchone() ambil satu baris hasil
        print(f"Connected as : {user}")
        print(f"Database     : {db}")
        
        # Query 2: verifikasi versi PostgreSQL
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        # Ambil hanya bagian pertama (nama + versi), sisanya verbose
        print(f"PG Version   : {version.split(',')[0]}")
        
        # Query 3: cek tabel yang sudah ada di schema public
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()  # fetchall() ambil semua baris sekaligus
        
        if tables:
            print(f"Tables found : {[t[0] for t in tables]}")
        else:
            print("Tables found : (none yet)")


def main() -> None:
    """Entry point — connect, diagnose, tutup koneksi."""
    print("=" * 45)
    print("PostgreSQL Connection Diagnostics")
    print("=" * 45)
    
    try:
        conn = get_connection()
        run_diagnostics(conn)
        
    except OperationalError as e:
        # OperationalError muncul jika: host tidak bisa dijangkau,
        # password salah, atau database tidak exist
        print(f"Connection failed: {e}")
        return
    
    finally:
        # finally selalu dieksekusi — pastikan koneksi SELALU ditutup
        # koneksi yang tidak ditutup = resource leak
        if 'conn' in locals() and conn:
            conn.close()
            print("-" * 45)
            print("Connection closed.")
    
    print("=" * 45)


if __name__ == "__main__":
    main()