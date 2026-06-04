-- ============================================================
-- W13D02: Schema untuk ML Prediction System
-- File: week-13/day-02/schema.sql
-- ============================================================

-- Hapus tabel lama jika ada (urutan penting: child dulu, baru parent)
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS models;

-- ────────────────────────────────────────────────
-- TABEL 1: models
-- Katalog semua ML model yang pernah di-deploy
-- ────────────────────────────────────────────────
CREATE TABLE models (
    -- SERIAL = auto-increment integer (1, 2, 3, ...)
    -- PRIMARY KEY otomatis NOT NULL + UNIQUE
    id          SERIAL PRIMARY KEY,

    -- nama model, wajib ada, tidak boleh duplikat
    name        TEXT NOT NULL,

    -- versi model: "1.0.0", "2.1.3", dst
    version     TEXT NOT NULL,

    -- UNIQUE composite: kombinasi name+version harus unik
    -- boleh ada model "fraud_detector" v1.0 dan v2.0,
    -- tapi tidak boleh dua "fraud_detector" v1.0
    CONSTRAINT uq_model_name_version UNIQUE (name, version),

    -- kapan model ini didaftarkan, default sekarang
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ────────────────────────────────────────────────
-- TABEL 2: predictions
-- Setiap prediction request yang masuk ke API
-- ────────────────────────────────────────────────
CREATE TABLE predictions (
    -- UUID lebih baik dari SERIAL untuk production API:
    -- tidak sequential (tidak bisa ditebak), aman di-expose ke client
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- foreign key ke tabel models
    -- ON DELETE RESTRICT: tidak bisa hapus model jika masih ada prediksi
    model_id        INTEGER NOT NULL
                    REFERENCES models(id) ON DELETE RESTRICT,

    -- input yang dikirim user ke API, disimpan sebagai JSON
    -- JSONB = binary JSON, lebih efisien untuk query
    input_data      JSONB NOT NULL,

    -- hasil prediksi dari model
    prediction      TEXT NOT NULL,

    -- confidence score: harus antara 0.0 dan 1.0
    -- CHECK constraint di-enforce oleh database, bukan aplikasi
    confidence      NUMERIC(5, 4) NOT NULL
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- latency API dalam milidetik, tidak boleh negatif
    latency_ms      INTEGER NOT NULL
                    CHECK (latency_ms >= 0),

    -- status prediksi: hanya nilai tertentu yang diizinkan
    status          TEXT NOT NULL DEFAULT 'success'
                    CHECK (status IN ('success', 'error', 'timeout')),

    -- timestamp
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ────────────────────────────────────────────────
-- INDEX: percepat query yang sering dipakai
-- (preview untuk W13D05 — dibahas lebih dalam nanti)
-- ────────────────────────────────────────────────

-- Query by model_id sangat umum: "tampilkan prediksi model X"
CREATE INDEX idx_predictions_model_id ON predictions(model_id);

-- Query by created_at untuk monitoring: "prediksi 1 jam terakhir"
CREATE INDEX idx_predictions_created_at ON predictions(created_at);