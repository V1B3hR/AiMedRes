-- Initial schema for Agent Memory Store with pgvector

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector; -- pgvector
-- Optional: for gen_random_uuid() if you want DB-side UUIDs
-- CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Enums
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'memory_type') THEN
        CREATE TYPE memory_type AS ENUM ('reasoning', 'experience', 'knowledge');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'association_type') THEN
        CREATE TYPE association_type AS ENUM ('similar', 'causal', 'temporal', 'related');
    END IF;
END$$;

-- Sessions
CREATE TABLE IF NOT EXISTS agent_sessions (
    id UUID PRIMARY KEY,
    agent_name VARCHAR(255) NOT NULL,
    agent_version VARCHAR(50) NOT NULL DEFAULT '1.0',
    session_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent_name ON agent_sessions (agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions (status);

-- Memories
-- Note: embedding dimension must match your model (all-MiniLM-L6-v2 = 384).
-- Change if you use a different model and reindex accordingly.
CREATE TABLE IF NOT EXISTS agent_memory (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    memory_type memory_type NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    importance_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    CONSTRAINT chk_importance_between_0_and_1 CHECK (importance_score >= 0.0 AND importance_score <= 1.0)
);
CREATE INDEX IF NOT EXISTS idx_agent_memory_session ON agent_memory (session_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory (memory_type);
CREATE INDEX IF NOT EXISTS idx_agent_memory_importance ON agent_memory (importance_score);
-- ANN index for cosine distance (requires normalized embeddings in queries or normalization)
-- Adjust lists based on data size (e.g., 100, 200, 500)
CREATE INDEX IF NOT EXISTS idx_agent_memory_embedding_cosine ON agent_memory
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Associations
CREATE TABLE IF NOT EXISTS memory_associations (
    source_memory_id BIGINT NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    target_memory_id BIGINT NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    association_type association_type NOT NULL,
    strength DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_memory_id, target_memory_id, association_type),
    CONSTRAINT chk_association_strength_between_0_and_1 CHECK (strength >= 0.0 AND strength <= 1.0)
);
