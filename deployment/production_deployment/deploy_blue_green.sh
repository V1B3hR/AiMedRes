#!/bin/bash
#
# Blue/Green Deployment Script for AiMedRes
# For medium-frequency updates with zero-downtime deployment
#
# This script:
# 1. Deploys to green environment while blue is serving traffic
# 2. Validates green environment
# 3. Switches traffic from blue to green
# 4. Keeps blue as backup for quick rollback
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLUE_ENV="${BLUE_ENV:-blue}"
GREEN_ENV="${GREEN_ENV:-green}"
CURRENT_ENV_FILE="${CURRENT_ENV_FILE:-/opt/aimedres/.current_env}"
LOG_FILE="${LOG_FILE:-/var/log/aimedres/deployment.log}"

# Colors
RED='\033[0;31m'
GREEN_COLOR='\033[0;32m'
YELLOW='\033[1;33m'
BLUE_COLOR='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN_COLOR}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE_COLOR}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Get current active environment
get_current_env() {
    if [ -f "$CURRENT_ENV_FILE" ]; then
        cat "$CURRENT_ENV_FILE"
    else
        echo "$BLUE_ENV"
    fi
}

# Get target environment (opposite of current)
get_target_env() {
    local current=$(get_current_env)
    if [ "$current" == "$BLUE_ENV" ]; then
        echo "$GREEN_ENV"
    else
        echo "$BLUE_ENV"
    fi
}

# Deploy to target environment
deploy_to_target() {
    local target_env=$1
    
    log "Deploying to $target_env environment..."
    
    # Set environment-specific variables
    export DEPLOYMENT_ENV="$target_env"
    export COMPOSE_PROJECT_NAME="aimedres_${target_env}"
    
    # Deploy using docker-compose
    cd "${SCRIPT_DIR}/../technical"
    docker-compose -p "aimedres_${target_env}" up -d --build
    
    if [ $? -eq 0 ]; then
        success "Deployed to $target_env environment"
        return 0
    else
        error "Failed to deploy to $target_env environment"
        return 1
    fi
}

# Validate target environment
validate_target() {
    local target_env=$1
    local port=8002
    
    if [ "$target_env" == "$GREEN_ENV" ]; then
        port=8003
    fi
    
    log "Validating $target_env environment on port $port..."
    
    # Wait for service to be ready
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s -o /dev/null "http://localhost:${port}/health"; then
            success "$target_env environment is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    error "$target_env environment health check failed"
    return 1
}

# Switch traffic to target environment
switch_traffic() {
    local target_env=$1
    
    log "Switching traffic to $target_env environment..."
    
    if [ -f "${SCRIPT_DIR}/switch_traffic.sh" ]; then
        bash "${SCRIPT_DIR}/switch_traffic.sh" "$target_env"
        if [ $? -eq 0 ]; then
            # Update current environment file
            echo "$target_env" > "$CURRENT_ENV_FILE"
            success "Traffic switched to $target_env"
            return 0
        else
            error "Failed to switch traffic"
            return 1
        fi
    else
        warning "switch_traffic.sh not found. Manual traffic switching required."
        # Update current environment file anyway
        echo "$target_env" > "$CURRENT_ENV_FILE"
        return 0
    fi
}

# Cleanup old environment
cleanup_old() {
    local old_env=$1
    
    log "Keeping $old_env environment for quick rollback..."
    info "To remove $old_env: docker-compose -p aimedres_${old_env} down"
}

# Main deployment flow
main() {
    log "========================================="
    log "Starting Blue/Green Deployment"
    log "========================================="
    
    local current_env=$(get_current_env)
    local target_env=$(get_target_env)
    
    info "Current environment: $current_env"
    info "Target environment: $target_env"
    
    # Deploy to target environment
    if ! deploy_to_target "$target_env"; then
        error "Deployment failed"
        exit 1
    fi
    
    # Validate target environment
    if ! validate_target "$target_env"; then
        error "Validation failed"
        error "Cleaning up failed deployment..."
        docker-compose -p "aimedres_${target_env}" down
        exit 1
    fi
    
    # Switch traffic
    if ! switch_traffic "$target_env"; then
        error "Traffic switching failed"
        exit 1
    fi
    
    # Keep old environment for rollback
    cleanup_old "$current_env"
    
    success "========================================="
    success "Blue/Green Deployment Completed!"
    success "Active environment: $target_env"
    success "Backup environment: $current_env (can be removed)"
    success "========================================="
    
    exit 0
}

# Run main function
main "$@"
