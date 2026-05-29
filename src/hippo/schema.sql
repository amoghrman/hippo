-- Hippo memory schema
-- Run via: psql $DATABASE_URL -f schema.sql
-- Or call Hippo.setup() which applies this automatically.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id      TEXT        NOT NULL,
    user_id       TEXT,
    content       TEXT        NOT NULL,
    embedding     vector(1536),
    importance    FLOAT       NOT NULL DEFAULT 0.5,
    metadata      JSONB       NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    superseded_by UUID        REFERENCES memories(id),
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS conflict_log (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id_old  UUID        NOT NULL REFERENCES memories(id),
    memory_id_new  UUID        NOT NULL REFERENCES memories(id),
    decision       TEXT        NOT NULL,   -- supersede | merge | coexist
    reason         TEXT,
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Composite index for filtered queries (agent scoping + active filter)
CREATE INDEX IF NOT EXISTS idx_memories_agent
    ON memories (agent_id, user_id, is_active, created_at);

-- HNSW index for fast approximate nearest-neighbour search
-- (HNSW works on empty tables, unlike IVFFlat)
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text search (BM25 via ts_rank)
CREATE INDEX IF NOT EXISTS idx_memories_fts
    ON memories USING gin(to_tsvector('english', content));
