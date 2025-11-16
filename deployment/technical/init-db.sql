-- AiMedRes Database Initialization Script
-- Creates necessary databases, schemas, and base tables

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create extension for cryptographic functions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS aimedres;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS mlops;

-- Set search path
SET search_path TO aimedres, public;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit.access_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    details JSONB,
    session_id VARCHAR(255)
);

-- Create index on audit log for faster queries
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit.access_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit.access_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit.access_log(action);

-- Create model registry table
CREATE TABLE IF NOT EXISTS mlops.model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    model_path TEXT,
    metrics JSONB,
    metadata JSONB,
    UNIQUE(model_name, version)
);

-- Create predictions log table for monitoring
CREATE TABLE IF NOT EXISTS aimedres.prediction_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    input_data_hash VARCHAR(64), -- SHA256 hash of input
    prediction JSONB NOT NULL,
    confidence FLOAT,
    inference_time_ms INTEGER,
    user_id VARCHAR(255),
    session_id VARCHAR(255)
);

-- Create index on prediction log
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON aimedres.prediction_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_model ON aimedres.prediction_log(model_name, model_version);

-- Create user sessions table
CREATE TABLE IF NOT EXISTS aimedres.user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true
);

-- Create index on sessions
CREATE INDEX IF NOT EXISTS idx_sessions_user ON aimedres.user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON aimedres.user_sessions(is_active, last_activity);

-- Grant permissions
GRANT USAGE ON SCHEMA aimedres TO PUBLIC;
GRANT USAGE ON SCHEMA audit TO PUBLIC;
GRANT USAGE ON SCHEMA mlops TO PUBLIC;

-- Insert initial data
INSERT INTO mlops.model_registry (model_name, version, status, created_by, metadata)
VALUES 
    ('alzheimers_detection', '1.0.0', 'active', 'system', '{"description": "Alzheimer''s disease early detection model"}'),
    ('parkinsons_detection', '1.0.0', 'active', 'system', '{"description": "Parkinson''s disease detection model"}'),
    ('als_detection', '1.0.0', 'active', 'system', '{"description": "ALS detection model"}')
ON CONFLICT (model_name, version) DO NOTHING;

-- Log initialization
INSERT INTO audit.access_log (action, resource_type, success, details)
VALUES ('database_initialization', 'system', true, '{"message": "Database initialized successfully"}');
