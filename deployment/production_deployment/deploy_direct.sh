#!/bin/bash
#
# Direct Deployment Script for AiMedRes
# For infrequent updates (monthly+) with simple, straightforward deployment
#
# This script performs:
# 1. Pre-deployment backup
# 2. Health checks
# 3. Service deployment
# 4. Post-deployment verification
#

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_ROOT="${DEPLOYMENT_ROOT:-/opt/aimedres}"
BACKUP_DIR="${BACKUP_DIR:-/var/backup/aimedres}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-${SCRIPT_DIR}/../technical/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-${DEPLOYMENT_ROOT}/.env.production}"
LOG_FILE="${LOG_FILE:-/var/log/aimedres/deployment.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Create backup before deployment
create_backup() {
    log "Creating pre-deployment backup..."
    
    if [ -f "${SCRIPT_DIR}/backup.sh" ]; then
        bash "${SCRIPT_DIR}/backup.sh" --type full --reason "pre-deployment"
        if [ $? -eq 0 ]; then
            success "Backup created successfully"
        else
            error "Backup failed. Aborting deployment."
            exit 1
        fi
    else
        warning "Backup script not found. Skipping backup."
    fi
}

# Perform health check
health_check() {
    local service_url="$1"
    local max_attempts=30
    local attempt=0
    
    log "Performing health check on $service_url..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s -o /dev/null "$service_url"; then
            success "Health check passed"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    cd "$(dirname "$DOCKER_COMPOSE_FILE")"
    docker-compose pull
    
    if [ $? -eq 0 ]; then
        success "Images pulled successfully"
    else
        error "Failed to pull images"
        exit 1
    fi
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    cd "$(dirname "$DOCKER_COMPOSE_FILE")"
    
    # Stop existing services
    log "Stopping existing services..."
    docker-compose down
    
    # Start services with new images
    log "Starting services..."
    docker-compose --env-file "$ENV_FILE" up -d
    
    if [ $? -eq 0 ]; then
        success "Services started successfully"
    else
        error "Failed to start services"
        exit 1
    fi
}

# Post-deployment verification
verify_deployment() {
    log "Performing post-deployment verification..."
    
    # Wait for services to be ready
    sleep 10
    
    # Check if all containers are running
    log "Checking container status..."
    local containers=$(docker-compose ps -q)
    local running=0
    local total=0
    
    for container in $containers; do
        total=$((total + 1))
        if docker inspect -f '{{.State.Running}}' "$container" | grep -q "true"; then
            running=$((running + 1))
        fi
    done
    
    log "Containers running: $running/$total"
    
    if [ $running -eq $total ]; then
        success "All containers are running"
    else
        error "Some containers are not running"
        return 1
    fi
    
    # Health check endpoints
    if [ -f "${SCRIPT_DIR}/verify_deployment.sh" ]; then
        bash "${SCRIPT_DIR}/verify_deployment.sh"
        if [ $? -eq 0 ]; then
            success "Deployment verification passed"
        else
            error "Deployment verification failed"
            return 1
        fi
    else
        warning "Verification script not found. Skipping detailed verification."
    fi
    
    return 0
}

# Rollback on failure
rollback() {
    error "Deployment failed. Rolling back..."
    
    cd "$(dirname "$DOCKER_COMPOSE_FILE")"
    docker-compose down
    
    # Restore from backup if available
    if [ -f "${SCRIPT_DIR}/restore.sh" ]; then
        warning "Attempting to restore from backup..."
        bash "${SCRIPT_DIR}/restore.sh" --latest
    fi
    
    error "Rollback completed. Please investigate the issue."
    exit 1
}

# Main deployment flow
main() {
    log "========================================="
    log "Starting Direct Deployment"
    log "========================================="
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup
    create_backup
    
    # Pull images
    pull_images
    
    # Deploy services
    deploy_services
    
    # Verify deployment
    if verify_deployment; then
        success "========================================="
        success "Deployment completed successfully!"
        success "========================================="
        log "Deployment timestamp: $(date)"
        exit 0
    else
        rollback
    fi
}

# Run main function
main "$@"
