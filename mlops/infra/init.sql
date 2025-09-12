-- Initialize DuetMind MLOps Database
-- Creates necessary extensions and basic setup

-- Create pgvector extension for vector similarity search (if available)
-- This will fail gracefully if pgvector is not installed
CREATE EXTENSION IF NOT EXISTS vector;

-- Create uuid extension for generating UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create basic schema (Alembic will handle the rest)
COMMENT ON DATABASE duetmind_mlops IS 'DuetMind Adaptive MLOps metadata store';