#!/bin/bash
#
# Quick Model Rollback Script
# Rolls back a specific model to a previous version
#

set -e

MODEL_NAME="$1"
MODEL_VERSION="$2"

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_VERSION" ]; then
    echo "Usage: $0 <model_name> <model_version>"
    echo "Example: $0 alzheimer 1.1.0"
    exit 1
fi

# Configuration
MODEL_REGISTRY="${MODEL_REGISTRY:-/var/aimedres/models}"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Verify model version exists
verify_model() {
    local model_path="${MODEL_REGISTRY}/${MODEL_NAME}/${MODEL_VERSION}"
    
    if [ ! -d "$model_path" ]; then
        echo "ERROR: Model version not found: $model_path"
        exit 1
    fi
    
    log "Found model: ${MODEL_NAME} version ${MODEL_VERSION}"
}

# Update model symlink
update_symlink() {
    local source="${MODEL_REGISTRY}/${MODEL_NAME}/${MODEL_VERSION}"
    local target="${MODEL_REGISTRY}/${MODEL_NAME}/current"
    
    log "Updating model symlink..."
    
    # Backup current symlink
    if [ -L "$target" ]; then
        local current_version=$(readlink "$target")
        log "Current version: $current_version"
        log "Rolling back to: ${MODEL_VERSION}"
    fi
    
    # Update symlink
    ln -sfn "$source" "$target"
    
    log "Symlink updated successfully"
}

# Reload model in application
reload_model() {
    log "Reloading model in application..."
    
    # Send SIGHUP to application to reload models
    if docker ps | grep -q aimedres; then
        docker exec $(docker ps -q -f name=aimedres) pkill -HUP python || true
        log "Model reload signal sent"
    else
        log "WARNING: Application container not running"
    fi
}

# Main rollback flow
main() {
    log "========================================="
    log "Starting Model Rollback"
    log "Model: ${MODEL_NAME}"
    log "Version: ${MODEL_VERSION}"
    log "========================================="
    
    verify_model
    update_symlink
    reload_model
    
    log "========================================="
    log "Model rollback completed!"
    log "========================================="
}

main "$@"
