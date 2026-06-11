# alembic/env.py
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from dotenv import load_dotenv

# 1. SETUP PATH ABSOLUT UNTUK .ENV DAN SCRIPTS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

print(f" DEBUG: Alembic mencari .env di jalur: {ENV_PATH}")
print(f" DEBUG: Apakah file .env fisik ada di sana? {os.path.exists(ENV_PATH)}")
# -------------------------------------

sys.path.append(os.path.join(BASE_DIR, "scripts"))

sys.path.append(os.path.join(BASE_DIR, "scripts"))

# 2. IMPORT METADATA DARI MODEL PYTHON
import database
import models

# 3. INTERPRET CONFIG FILE UNTUK LOGGING BROWSER TERMINAL
if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

# VARIABEL JUARA: Ini yang dicari oleh fitur --autogenerate
target_metadata = database.Base.metadata

# 4. KONFIGURASI DATABASE URL DENGAN FALLBACK AMAN
db_url = os.getenv("DATABASE_URL")
if not db_url:
    db_url = "postgresql://dss_user:secret@localhost:5432/dss_db"
    print(" WARNING: DATABASE_URL tidak ditemukan di .env, menggunakan default fallback.")

config = context.config
config.set_main_option("sqlalchemy.url", db_url)

def include_object(object, name, type_, reflected, compare_to):
    """Menyaring objek database agar tabel Week-13 tidak ikut di-drop oleh Alembic."""
    ignored_tables = ["users", "simulation_requests", "simulation_scenarios", "recommendations"]
    if type_ == "table" and name in ignored_tables:
        return False
    return True

# 5. FUNGSI EKSEKUSI MIGRASI OFFLINE
def run_migrations_offline() -> None:
    """Menjalankan migrasi dalam mode 'offline' (generate skrip SQL mentah)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object
    )
    
    with context.begin_transaction():
        context.run_migrations()


# 6. FUNGSI EKSEKUSI MIGRASI ONLINE (LANGSUNG TEMBAK KE DOCKER)
def run_migrations_online() -> None:
    """Menjalankan migrasi dalam mode 'online' (koneksi aktif ke DB)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, # (atau url=url untuk mode offline)
            target_metadata=target_metadata,
            include_object=include_object
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()