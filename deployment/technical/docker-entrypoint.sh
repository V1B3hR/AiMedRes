#!/bin/bash
# AiMedRes Docker Entrypoint Script
# Handles initialization, health checks, and graceful startup

set -e

echo "=========================================="
echo "AiMedRes Healthcare Deployment Starting"
echo "=========================================="

# Function to log messages with timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a service is available
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name at $host:$port..."
    
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            log "ERROR: $service_name at $host:$port is not available after $max_attempts attempts"
            exit 1
        fi
        log "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log "$service_name is available!"
}

# Function to check database connectivity
check_database() {
    log "Checking database connection..."
    
    if [ -n "$DATABASE_URL" ]; then
        python3 -c "
import sys
from sqlalchemy import create_engine
try:
    engine = create_engine('$DATABASE_URL')
    conn = engine.connect()
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" || exit 1
    else
        log "WARNING: DATABASE_URL not set, skipping database check"
    fi
}

# Function to run database migrations
run_migrations() {
    log "Running database migrations..."
    
    if [ -f "/app/migrations/migrate.py" ]; then
        python3 /app/migrations/migrate.py || {
            log "WARNING: Migration script failed, continuing anyway..."
        }
    else
        log "No migration script found, skipping migrations"
    fi
}

# Function to initialize application data
initialize_app() {
    log "Initializing application..."
    
    # Create required directories if they don't exist
    mkdir -p /app/logs/application /app/logs/audit /app/logs/security
    mkdir -p /app/outputs/predictions /app/outputs/reports
    
    # Set proper permissions
    chmod 750 /app/logs /app/outputs
    
    # Initialize configuration
    if [ ! -f "/app/config.yml" ]; then
        log "WARNING: config.yml not found, using default configuration"
        if [ -f "/app/config.yml.example" ]; then
            cp /app/config.yml.example /app/config.yml
        fi
    fi
    
    log "Application initialization complete"
}

# Function to verify security settings
verify_security() {
    log "Verifying security configuration..."
    
    # Check for required security environment variables
    local required_vars=("SECRET_KEY" "ENCRYPTION_KEY")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log "ERROR: Required security environment variables are not set:"
        printf '%s\n' "${missing_vars[@]}"
        log "Please set these variables in your .env file or environment"
        exit 1
    fi
    
    # Verify PHI scrubbing is enabled in production
    if [ "$ENVIRONMENT" = "production" ] && [ "$ENABLE_PHI_SCRUBBING" != "true" ]; then
        log "WARNING: PHI scrubbing is not enabled in production environment!"
    fi
    
    # Verify audit logging is enabled
    if [ "$ENABLE_AUDIT_LOGGING" != "true" ]; then
        log "WARNING: Audit logging is not enabled!"
    fi
    
    log "Security verification complete"
}

# Function to check model files
check_models() {
    log "Checking for model files..."
    
    if [ -d "/app/models" ] && [ "$(ls -A /app/models)" ]; then
        log "Model files found in /app/models"
        ls -lh /app/models
    else
        log "WARNING: No model files found. Models may need to be downloaded or trained."
    fi
}

# Function to display startup information
display_info() {
    log "=========================================="
    log "Startup Information:"
    log "Environment: ${ENVIRONMENT:-not set}"
    log "Log Level: ${LOG_LEVEL:-INFO}"
    log "PHI Scrubbing: ${ENABLE_PHI_SCRUBBING:-false}"
    log "Audit Logging: ${ENABLE_AUDIT_LOGGING:-false}"
    log "Encryption at Rest: ${ENABLE_ENCRYPTION_AT_REST:-false}"
    log "=========================================="
}

# Main startup sequence
main() {
    log "Starting AiMedRes initialization sequence..."
    
    # Display configuration info
    display_info
    
    # Wait for dependent services if they're configured
    if [ -n "$DATABASE_URL" ]; then
        # Extract host and port from DATABASE_URL
        # Format: postgresql://user:pass@host:port/db
        DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        
        if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
            wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL"
            check_database
            run_migrations
        fi
    fi
    
    # Initialize application
    initialize_app
    
    # Verify security settings
    verify_security
    
    # Check for model files
    check_models
    
    log "Initialization complete, starting application..."
    log "=========================================="
    
    # Execute the main command
    exec "$@"
}

# Handle signals for graceful shutdown
trap 'log "Received SIGTERM, initiating graceful shutdown..."; kill -TERM $PID' TERM
trap 'log "Received SIGINT, initiating graceful shutdown..."; kill -INT $PID' INT

# Run main function with all arguments
main "$@" &
PID=$!
wait $PID
