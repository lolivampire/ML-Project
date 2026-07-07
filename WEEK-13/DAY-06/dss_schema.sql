-- =============================================================
-- Schema: Decision Support System (Project 3)
-- PostgreSQL 14+
-- =============================================================

-- Enable UUID generation (built-in di PostgreSQL 13+)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -------------------------------------------------------------
-- Tabel: users
-- Menyimpan pengguna sistem. user_id di tabel lain mengacu ke sini.
-- -------------------------------------------------------------
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username    VARCHAR(50) NOT NULL,
    email       VARCHAR(255) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- Tabel: simulation_requests
-- Setiap kali user submit form simulasi = 1 baris di sini.
-- Ini adalah tabel pusat (parent) dari semua entitas lain.
-- -------------------------------------------------------------
CREATE TABLE simulation_requests (
    id                      UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID            NOT NULL REFERENCES users(id),
    budget                 NUMERIC(12,2)   NOT NULL,
    risk_tolerance          FLOAT           NOT NULL CHECK (risk_tolerance BETWEEN 0 AND 1),
    time_horizon_months     INTEGER         NOT NULL CHECK (time_horizon_months > 0),
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending'
                            CHECK           (status IN ('pending', 'processing', 'completed', 'failed')),
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- Tabel: simulation_scenarios
-- 3 baris per request — optimistic, realistic, pessimistic.
-- Child dari simulation_requests.
-- -------------------------------------------------------------
CREATE TABLE simulation_scenarios (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id      UUID        NOT NULL REFERENCES simulation_requests(id) ON DELETE CASCADE,
    scenario_type   VARCHAR(20) NOT NULL
                    CHECK (scenario_type IN ('optimistic', 'realistic', 'pessimistic')),
    projected_return    NUMERIC(10, 4),
    risk_score          FLOAT CHECK (risk_score >= 0),
    confidence_level    FLOAT CHECK (confidence_level BETWEEN 0 AND 1),
    parameters          JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (request_id, scenario_type)
);

-- -------------------------------------------------------------
-- Tabel: recommendations
-- Output berperingkat per request, opsional diikat ke skenario tertentu.
-- -------------------------------------------------------------
CREATE TABLE recommendations (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id      UUID        NOT NULL REFERENCES simulation_requests(id) ON DELETE CASCADE,  -- Setiap rekomendasi WAJIB terikat ke request
    scenario_id     UUID        REFERENCES simulation_scenarios(id) ON DELETE SET NULL,         -- Opsional: rekomendasi bisa spesifik ke skenario tertentu -- ON DELETE SET NULL: jika scenario dihapus, rekomendasi tetap ada
    priority_rank   INTEGER     NOT NULL CHECK (priority_rank > 0),                             -- 1 = prioritas tertinggi, 2 = kedua, dst.
    action_label    VARCHAR(255) NOT NULL,                                                      -- Label singkat tindakan: "Alokasikan 40% ke obligasi negara"
    confidence_score    FLOAT   NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),              -- Seberapa yakin model dengan rekomendasi ini
    reasoning       TEXT,                                                                       -- Penjelasan panjang kenapa rekomendasi ini diberikan -- TEXT: tidak dibatasi panjang
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================
-- TRIGGER: auto-update updated_at pada simulation_requests
-- =============================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    -- NEW = baris baru yang akan disimpan
    -- OLD = baris lama (sebelum UPDATE)
    -- Trigger ini dipanggil BEFORE UPDATE, jadi kita set NEW.updated_at
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_simulation_requests_updated_at
    -- Trigger ini otomatis mengisi updated_at setiap kali baris di-UPDATE
    -- Tanpa ini, kamu harus ingat SET updated_at = NOW() di setiap UPDATE query
    BEFORE UPDATE ON simulation_requests
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================
-- INDEX: berdasarkan pola query yang diantisipasi
-- =============================================================

-- Index pada FK paling sering di-JOIN
CREATE INDEX idx_simulation_requests_user_id        ON simulation_requests(user_id);

CREATE INDEX idx_simulation_scenarios_request_id    ON simulation_scenarios(request_id);

CREATE INDEX idx_recommendations_request_id         ON recommendations(request_id);

-- Index pada status — sering difilter di WHERE
CREATE INDEX idx_simulation_requests_status         ON simulation_requests(status)          WHERE status IN ('pending', 'processing');

-- Index pada scenario_type — sering difilter
CREATE INDEX idx_simulation_scenarios_type          ON simulation_scenarios(scenario_type);

-- Composite index: query sering minta semua scenarios untuk request tertentu
CREATE INDEX idx_simulation_scenarios_request_id_type ON simulation_scenarios(request_id, scenario_type);

-- Index pada priority_rank — untuk ORDER BY di recommendations
CREATE INDEX idx_recommendations_priority_rank      ON recommendations(request_id, priority_rank);

-- =============================================================
-- ANALYZE: update statistik planner
-- =============================================================

ANALYZE users;
ANALYZE simulation_requests;
ANALYZE simulation_scenarios;
ANALYZE recommendations; 