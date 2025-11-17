#!/bin/bash
#
# Comprehensive Backup Script for AiMedRes
# Performs full and incremental backups with encryption
#

set -e

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backup/aimedres}"
BACKUP_ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-/etc/aimedres/backup.key}"
S3_BUCKET="${S3_BUCKET:-s3://aimedres-backups}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-aimedres_postgres_1}"
MODEL_DIR="${MODEL_DIR:-/var/aimedres/models}"
CONFIG_DIR="${CONFIG_DIR:-/opt/aimedres/config}"
RESULTS_DIR="${RESULTS_DIR:-/var/aimedres/results}"
AUDIT_LOG_DIR="${AUDIT_LOG_DIR:-/var/log/aimedres/audit}"

# Parse arguments
BACKUP_TYPE="${1:-incremental}"
BACKUP_REASON="${2:-scheduled}"

# Create backup directory structure
mkdir -p "$BACKUP_ROOT"/{database,models,config,results,audit_logs}

# Timestamp for backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/backup_${TIMESTAMP}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Generate backup encryption key if it doesn't exist
generate_encryption_key() {
    if [ ! -f "$BACKUP_ENCRYPTION_KEY" ]; then
        log "Generating backup encryption key..."
        openssl rand -base64 32 > "$BACKUP_ENCRYPTION_KEY"
        chmod 600 "$BACKUP_ENCRYPTION_KEY"
        log "Encryption key generated at $BACKUP_ENCRYPTION_KEY"
    fi
}

# Backup PostgreSQL database
backup_database() {
    log "Backing up database..."
    
    local db_backup_file="${BACKUP_DIR}/database/aimedres_db_${TIMESTAMP}.sql.gz"
    mkdir -p "$(dirname "$db_backup_file")"
    
    # Dump database
    docker exec "$POSTGRES_CONTAINER" pg_dump -U aimedres aimedres | gzip > "$db_backup_file"
    
    # Encrypt backup
    openssl enc -aes-256-cbc -salt -in "$db_backup_file" \
        -out "${db_backup_file}.enc" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Remove unencrypted backup
    rm "$db_backup_file"
    
    # Generate checksum
    sha256sum "${db_backup_file}.enc" > "${db_backup_file}.enc.sha256"
    
    log "Database backup completed: ${db_backup_file}.enc"
}

# Backup models
backup_models() {
    log "Backing up models..."
    
    if [ ! -d "$MODEL_DIR" ]; then
        log "WARNING: Model directory not found: $MODEL_DIR"
        return 0
    fi
    
    local models_backup_file="${BACKUP_DIR}/models/models_${TIMESTAMP}.tar.gz"
    mkdir -p "$(dirname "$models_backup_file")"
    
    # Create tar archive
    tar -czf "$models_backup_file" -C "$(dirname "$MODEL_DIR")" "$(basename "$MODEL_DIR")"
    
    # Encrypt backup
    openssl enc -aes-256-cbc -salt -in "$models_backup_file" \
        -out "${models_backup_file}.enc" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Remove unencrypted backup
    rm "$models_backup_file"
    
    # Generate checksum
    sha256sum "${models_backup_file}.enc" > "${models_backup_file}.enc.sha256"
    
    log "Models backup completed: ${models_backup_file}.enc"
}

# Backup configuration
backup_config() {
    log "Backing up configuration..."
    
    if [ ! -d "$CONFIG_DIR" ]; then
        log "WARNING: Config directory not found: $CONFIG_DIR"
        return 0
    fi
    
    local config_backup_file="${BACKUP_DIR}/config/config_${TIMESTAMP}.tar.gz"
    mkdir -p "$(dirname "$config_backup_file")"
    
    # Create tar archive
    tar -czf "$config_backup_file" -C "$(dirname "$CONFIG_DIR")" "$(basename "$CONFIG_DIR")"
    
    # Encrypt backup
    openssl enc -aes-256-cbc -salt -in "$config_backup_file" \
        -out "${config_backup_file}.enc" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Remove unencrypted backup
    rm "$config_backup_file"
    
    # Generate checksum
    sha256sum "${config_backup_file}.enc" > "${config_backup_file}.enc.sha256"
    
    log "Configuration backup completed: ${config_backup_file}.enc"
}

# Backup results (incremental or full)
backup_results() {
    log "Backing up results (${BACKUP_TYPE})..."
    
    if [ ! -d "$RESULTS_DIR" ]; then
        log "WARNING: Results directory not found: $RESULTS_DIR"
        return 0
    fi
    
    local results_backup_file="${BACKUP_DIR}/results/results_${TIMESTAMP}.tar.gz"
    mkdir -p "$(dirname "$results_backup_file")"
    
    if [ "$BACKUP_TYPE" == "incremental" ]; then
        # Incremental backup: only files modified in last 24 hours
        find "$RESULTS_DIR" -type f -mtime -1 -print0 | \
            tar -czf "$results_backup_file" --null -T -
    else
        # Full backup
        tar -czf "$results_backup_file" -C "$(dirname "$RESULTS_DIR")" "$(basename "$RESULTS_DIR")"
    fi
    
    # Encrypt backup
    openssl enc -aes-256-cbc -salt -in "$results_backup_file" \
        -out "${results_backup_file}.enc" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Remove unencrypted backup
    rm "$results_backup_file"
    
    # Generate checksum
    sha256sum "${results_backup_file}.enc" > "${results_backup_file}.enc.sha256"
    
    log "Results backup completed: ${results_backup_file}.enc"
}

# Backup audit logs (always full for compliance)
backup_audit_logs() {
    log "Backing up audit logs..."
    
    if [ ! -d "$AUDIT_LOG_DIR" ]; then
        log "WARNING: Audit log directory not found: $AUDIT_LOG_DIR"
        return 0
    fi
    
    local audit_backup_file="${BACKUP_DIR}/audit_logs/audit_${TIMESTAMP}.tar.gz"
    mkdir -p "$(dirname "$audit_backup_file")"
    
    # Full backup of audit logs (compliance requirement)
    tar -czf "$audit_backup_file" -C "$(dirname "$AUDIT_LOG_DIR")" "$(basename "$AUDIT_LOG_DIR")"
    
    # Encrypt backup
    openssl enc -aes-256-cbc -salt -in "$audit_backup_file" \
        -out "${audit_backup_file}.enc" \
        -pass file:"$BACKUP_ENCRYPTION_KEY"
    
    # Remove unencrypted backup
    rm "$audit_backup_file"
    
    # Generate checksum
    sha256sum "${audit_backup_file}.enc" > "${audit_backup_file}.enc.sha256"
    
    log "Audit logs backup completed: ${audit_backup_file}.enc"
}

# Create backup manifest
create_manifest() {
    log "Creating backup manifest..."
    
    local manifest_file="${BACKUP_DIR}/manifest.json"
    
    cat > "$manifest_file" << EOF
{
    "backup_id": "backup_${TIMESTAMP}",
    "timestamp": "$(date -Iseconds)",
    "type": "${BACKUP_TYPE}",
    "reason": "${BACKUP_REASON}",
    "components": {
        "database": "$(ls ${BACKUP_DIR}/database/*.enc 2>/dev/null | wc -l) files",
        "models": "$(ls ${BACKUP_DIR}/models/*.enc 2>/dev/null | wc -l) files",
        "config": "$(ls ${BACKUP_DIR}/config/*.enc 2>/dev/null | wc -l) files",
        "results": "$(ls ${BACKUP_DIR}/results/*.enc 2>/dev/null | wc -l) files",
        "audit_logs": "$(ls ${BACKUP_DIR}/audit_logs/*.enc 2>/dev/null | wc -l) files"
    },
    "encryption": "AES-256-CBC",
    "total_size": "$(du -sh ${BACKUP_DIR} | cut -f1)"
}
EOF
    
    log "Manifest created: $manifest_file"
}

# Sync to S3 (optional)
sync_to_s3() {
    if command -v aws &> /dev/null && [ -n "$S3_BUCKET" ]; then
        log "Syncing backup to S3..."
        
        aws s3 sync "$BACKUP_DIR" "${S3_BUCKET}/backup_${TIMESTAMP}/" \
            --storage-class STANDARD_IA \
            --server-side-encryption AES256
        
        if [ $? -eq 0 ]; then
            log "Backup synced to S3 successfully"
        else
            log "WARNING: Failed to sync to S3"
        fi
    else
        log "Skipping S3 sync (aws CLI not available or S3_BUCKET not configured)"
    fi
}

# Cleanup old backups (retain 30 days locally)
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    find "$BACKUP_ROOT" -maxdepth 1 -type d -name "backup_*" -mtime +30 -exec rm -rf {} \;
    
    log "Old backups cleaned up (kept last 30 days)"
}

# Main backup flow
main() {
    log "========================================="
    log "Starting Backup (Type: ${BACKUP_TYPE})"
    log "Reason: ${BACKUP_REASON}"
    log "========================================="
    
    # Generate encryption key if needed
    generate_encryption_key
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Perform backups
    backup_database
    backup_models
    backup_config
    backup_results
    backup_audit_logs
    
    # Create manifest
    create_manifest
    
    # Sync to S3
    sync_to_s3
    
    # Cleanup old backups
    cleanup_old_backups
    
    log "========================================="
    log "Backup completed successfully!"
    log "Backup location: $BACKUP_DIR"
    log "========================================="
}

main "$@"
