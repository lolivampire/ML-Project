# alembic/versions/xxxx_add_trigram_index_to_posts_content.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = None # Sesuaikan dengan revision sebelumnya di production nanti itu wajib diisi dengan revision ID sebelumnya, atau Alembic akan bingung soal urutan migration
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Aktifkan ekstensi pg_trgm di PostgreSQL (Jika belum ada)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    
    # 2. Buat GIN Index menggunakan operator class gin_trgm_ops
    # Ini akan mengindeks trigram (kombinasi 3 karakter) dari teks,
    # memungkinkan pencarian LIKE '%keyword%' berjalan dalam waktu O(log N).
    op.create_index(
        'ix_posts_content_trgm',
        'posts',
        [text("content gin_trgm_ops")],
        postgresql_using='gin'
    )

def downgrade() -> None:
    op.drop_index('ix_posts_content_trgm', table_name='posts')
    # Opsional: Hapus ekstensi jika tidak ada tabel lain yang menggunakannya
    # op.execute("DROP EXTENSION IF EXISTS pg_trgm;")