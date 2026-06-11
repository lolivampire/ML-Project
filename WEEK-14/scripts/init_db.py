# week-14/scripts/init_db.py
from database import db_manager, Base
import models  # Memaksa python mendaftarkan metadata model ke Base

def init_db() -> None:
    """Membaca model Python dan membuat tabel fisiknya di PostgreSQL jika belum ada."""
    print("Connecting to database and verifying metadata...")
    print("Creating tables if not exists...")
    
    # create_all membaca semua class yang mewarisi 'Base'
    Base.metadata.create_all(bind=db_manager.engine)
    
    print("Database initialization successfully completed.")

if __name__ == "__main__":
    init_db()