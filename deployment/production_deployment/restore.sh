#!/bin/bash
#
# Comprehensive Restore Script for AiMedRes
# Restores from encrypted backups
#

set -e

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backup/aimedres}"
BACKUP_ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-/etc/aimedres/backup.key}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-aimedres_postgres_1}"

# Parse arguments
BACKUP_ID="$1"

if [ -z "$BACKUP_ID" ] || [ "$BACKUP_ID" == "--latest" ]; then
    # Find latest backup
    BACKUP_ID=$(ls -t "$BACKUP_ROOT" | grep "^backup_" | head -1)
fi

BACKUP_DIR="$BACKUP_ROOT/$BACKUP_ID"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "ERROR: $1" >&2
}

# Verify backup exists
verify_backup() {
    if [ ! -d "$BACKUP_DIR" ]; then
        error "Backup not found: $BACKUP_DIR"
        exit 1
    fi
    
    if [ ! -f "$BACKUP_ENCRYPTION_KEY" ]; then
        error "Encryption key not found: $BACKUP_ENCRYPTION_KEY"
        exit 1
    fi
    
    log "Backup directory: $BACKUP_DIR"
    log "Manifest:"
    if [ -f "$BACKUP_DIR/manifest.json" ]; then
        cat "$BACKUP_DIR/manifest.json"
    fi
}

# Verify checksums
verify_checksums() {
    log "Verifying backup integrity..."
    
    local failed=0
    for checksum_file in $(find "$BACKUP_DIR" -name "*.sha256"); do
        local encrypted_file="${checksum_file%.sha256}"
        
        if [ -f "$encrypted_file" ]; then
            cd "$(dirname "$encrypted_file")"
            if sha256sum -c "$(basename "$checksum_file")" > /dev/null 2>&1; then
                log "✓ Verified: $(basename "$encrypted_file")"
            else
                error "✗ Checksum mismatch: $(basename "$encrypted_file")"
                failed=$((failed + 1))
            fi
        fi
    done
    
    if [ $failed -gt 0 ]; then
        error "$failed file(s) failed checksum verification"
        exit 1
    fi
    
    log "All checksums verified successfully"
}

# Restore database
restore_database() {
    log "Restoring database..."
    
    local encrypted_db=$(find "$BACKUP_DIR/database" -name "*.sql.gz.enc" | head -1)
    
    if [ -z "$encrypted_db" ]; then
        error "Database backup not found"
        return 1
    fi
    
    local temp_dir=$(mktemp -d)
    local decrypted_db="${temp_dir}/database.sql.gz"
    
    # Decrypt database backup
    openssl enc -aes-256-cbc -d -in "$encrypted_db" \
        -out "$decrypted_db" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Stop services
    log "Stopping services..."
    docker-compose down
    
    # Drop and recreate database
    docker-compose up -d postgres
    sleep 5
    
    docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP DATABASE IF EXISTS aimedres;"
    docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "CREATE DATABASE aimedres OWNER aimedres;"
    
    # Restore database
    gunzip -c "$decrypted_db" | docker exec -i "$POSTGRES_CONTAINER" psql -U aimedres aimedres
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "Database restored successfully"
}

# Restore models
restore_models() {
    log "Restoring models..."
    
    local encrypted_models=$(find "$BACKUP_DIR/models" -name "*.tar.gz.enc" | head -1)
    
    if [ -z "$encrypted_models" ]; then
        log "WARNING: Models backup not found"
        return 0
    fi
    
    local temp_dir=$(mktemp -d)
    local decrypted_models="${temp_dir}/models.tar.gz"
    
    # Decrypt models backup
    openssl enc -aes-256-cbc -d -in "$encrypted_models" \
        -out "$decrypted_models" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Extract models
    tar -xzf "$decrypted_models" -C /var/aimedres/
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "Models restored successfully"
}

# Restore configuration
restore_config() {
    log "Restoring configuration..."
    
    local encrypted_config=$(find "$BACKUP_DIR/config" -name "*.tar.gz.enc" | head -1)
    
    if [ -z "$encrypted_config" ]; then
        log "WARNING: Configuration backup not found"
        return 0
    fi
    
    local temp_dir=$(mktemp -d)
    local decrypted_config="${temp_dir}/config.tar.gz"
    
    # Decrypt config backup
    openssl enc -aes-256-cbc -d -in "$encrypted_config" \
        -out "$decrypted_config" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Extract config
    tar -xzf "$decrypted_config" -C /opt/aimedres/
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "Configuration restored successfully"
}

# Restore results
restore_results() {
    log "Restoring results..."
    
    local encrypted_results=$(find "$BACKUP_DIR/results" -name "*.tar.gz.enc" | head -1)
    
    if [ -z "$encrypted_results" ]; then
        log "WARNING: Results backup not found"
        return 0
    fi
    
    local temp_dir=$(mktemp -d)
    local decrypted_results="${temp_dir}/results.tar.gz"
    
    # Decrypt results backup
    openssl enc -aes-256-cbc -d -in "$encrypted_results" \
        -out "$decrypted_results" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Extract results
    tar -xzf "$decrypted_results" -C /var/aimedres/
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "Results restored successfully"
}

# Restart services
restart_services() {
    log "Restarting services..."
    
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 10
    
    log "Services restarted"
}

# Verify restoration
verify_restoration() {
    log "Verifying restoration..."
    
    # Check if database is accessible
    if docker exec "$POSTGRES_CONTAINER" psql -U aimedres -d aimedres -c "SELECT 1;" > /dev/null 2>&1; then
        log "✓ Database is accessible"
    else
        error "✗ Database is not accessible"
        return 1
    fi
    
    # Check if services are running
    if docker ps | grep -q aimedres; then
        log "✓ Services are running"
    else
        error "✗ Services are not running"
        return 1
    fi
    
    log "Restoration verification completed"
}

# Main restore flow
main() {
    log "========================================="
    log "Starting Restore"
    log "========================================="
    
    verify_backup
    verify_checksums
    
    # Confirm with user
    echo ""
    echo "WARNING: This will overwrite the current system with the backup."
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        log "Restore cancelled by user"
        exit 0
    fi
    
    # Perform restore
    restore_database
    restore_models
    restore_config
    restore_results
    restart_services
    verify_restoration
    
    log "========================================="
    log "Restore completed successfully!"
    log "========================================="
}

main "$@"
