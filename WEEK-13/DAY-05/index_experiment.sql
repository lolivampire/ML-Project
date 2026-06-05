-- =============================================================================
-- EXPERIMEN OPTIMASI INDEX: 200.000 DATA SIMULASI (FIXED VERSION)
-- =============================================================================

-- Matikan pager agar tidak terhenti di tengah jalan
\set pager off

-- Hapus tabel lama jika ada agar tes bisa di-run berkali-kali secara bersih
DROP TABLE IF EXISTS ml_predictions;

-- Hapus indeks lama yang berpotensi bentrok namanya
DROP INDEX IF EXISTS idx_ml_predictions_user_id;
DROP INDEX IF EXISTS idx_ml_predictions_created_at;
DROP INDEX IF EXISTS idx_ml_predictions_model_status;
DROP INDEX IF EXISTS idx_ml_predictions_pending;

\echo '\n=== [1] Membuat Tabel & Mengisi 200.000 Data... ==='
CREATE TABLE ml_predictions (
    id          SERIAL PRIMARY KEY,
    user_id     INT NOT NULL,
    model_name  VARCHAR(50),
    score       NUMERIC(5,4),
    status      VARCHAR(20) DEFAULT 'pending',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO ml_predictions (user_id, model_name, score, status)
SELECT
    (random() * 10000)::INT,
    CASE (random()*3)::INT
        WHEN 0 THEN 'xgboost_v1'
        WHEN 1 THEN 'rf_v2'
        ELSE 'logistic_v1'
    END,
    random()::NUMERIC(5,4),
    CASE (random()*2)::INT
        WHEN 0 THEN 'pending'
        WHEN 1 THEN 'done'
        ELSE 'failed'
    END
FROM generate_series(1, 200000);

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [2] Memperbarui Statistik Tabel (ANALYZE)... ==='
ANALYZE ml_predictions;

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [3] Query SEBELUM Ada Index (Mencari user_id)... ==='
EXPLAIN ANALYZE
SELECT * FROM ml_predictions WHERE user_id = 4242;

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [4] Membuat Index B-Tree (Gunakan Nama Unik Khusus)... ==='
CREATE INDEX idx_ml_predictions_user_id ON ml_predictions (user_id);
CREATE INDEX idx_ml_predictions_created_at ON ml_predictions (created_at);

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [5] Query SETELAH Ada Index (Mencari user_id)... ==='
EXPLAIN ANALYZE
SELECT * FROM ml_predictions WHERE user_id = 4242;

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [6] Range Query Menggunakan Timestamp... ==='
-- Paksa planner menonaktifkan seq scan sementara khusus untuk melihat visualisasi index scan pada timestamp
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT user_id, score FROM ml_predictions
WHERE created_at >= NOW() - INTERVAL '1 hour';

-- Kembalikan settingan seq scan ke bawaan
SET enable_seqscan = on;

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [7] Membuat Composite Index (model_name + status)... ==='
CREATE INDEX idx_ml_predictions_model_status ON ml_predictions (model_name, status);

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [8] Menampilkan Daftar Seluruh Index di Tabel Ini... ==='
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'ml_predictions';

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [9] Mengukur Ukuran Fisik Tabel vs Seluruh Index... ==='
SELECT
    pg_size_pretty(pg_table_size('ml_predictions'))   AS table_size,
    pg_size_pretty(pg_indexes_size('ml_predictions')) AS indexes_size;

-- ─────────────────────────────────────────────────────────────────────────────
\echo '\n=== [Varian] Membuat Partial Index Khusus Baris "pending"... ==='
CREATE INDEX idx_ml_predictions_pending
    ON ml_predictions (created_at)
    WHERE status = 'pending';