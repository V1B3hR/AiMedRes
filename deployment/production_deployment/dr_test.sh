#!/bin/bash
#
# Disaster Recovery Test Script
# Tests backup and restore procedures without affecting production
#

set -e

# Configuration
TEST_ENV="${TEST_ENV:-dr-test}"
BACKUP_ID="${1:-latest}"
LOG_FILE="/var/log/aimedres/dr_test_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "ERROR: $1" | tee -a "$LOG_FILE" >&2
}

success() {
    echo "✓ $1" | tee -a "$LOG_FILE"
}

log "========================================="
log "Disaster Recovery Test"
log "Backup: $BACKUP_ID"
log "Test Environment: $TEST_ENV"
log "========================================="

# Step 1: Create test environment
log "\n1. Creating test environment..."
export COMPOSE_PROJECT_NAME="aimedres_${TEST_ENV}"
export DEPLOYMENT_ENV="$TEST_ENV"

cd "$(dirname "$0")/../technical"

if docker-compose ps | grep -q "$TEST_ENV"; then
    log "Cleaning up existing test environment..."
    docker-compose down
fi

success "Test environment prepared"

# Step 2: Start test services
log "\n2. Starting test services..."
docker-compose up -d postgres redis

# Wait for services to be ready
sleep 10

if docker-compose ps | grep -q postgres; then
    success "Test services started"
else
    error "Failed to start test services"
    exit 1
fi

# Step 3: Restore from backup
log "\n3. Restoring from backup..."

if [ "$BACKUP_ID" == "latest" ]; then
    BACKUP_ID=$(ls -t /var/backup/aimedres | grep "^backup_" | head -1)
fi

log "Using backup: $BACKUP_ID"

# Run restore script in test mode
POSTGRES_CONTAINER="${COMPOSE_PROJECT_NAME}_postgres_1" \
    bash "$(dirname "$0")/restore.sh" "$BACKUP_ID" <<< "yes"

if [ $? -eq 0 ]; then
    success "Backup restored successfully"
else
    error "Restore failed"
    docker-compose down
    exit 1
fi

# Step 4: Verify restoration
log "\n4. Verifying restored data..."

# Check database
log "Checking database..."
if docker exec "${COMPOSE_PROJECT_NAME}_postgres_1" psql -U aimedres -d aimedres -c "SELECT COUNT(*) FROM information_schema.tables;" > /dev/null 2>&1; then
    TABLE_COUNT=$(docker exec "${COMPOSE_PROJECT_NAME}_postgres_1" psql -U aimedres -d aimedres -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    success "Database accessible with $TABLE_COUNT tables"
else
    error "Database verification failed"
    docker-compose down
    exit 1
fi

# Check models
log "Checking models..."
if [ -d "/var/aimedres/models" ] && [ "$(ls -A /var/aimedres/models)" ]; then
    MODEL_COUNT=$(find /var/aimedres/models -type f -name "*.pkl" -o -name "*.pth" | wc -l)
    success "Models restored: $MODEL_COUNT model files found"
else
    error "Model verification failed"
fi

# Check configuration
log "Checking configuration..."
if [ -d "/opt/aimedres/config" ] && [ "$(ls -A /opt/aimedres/config)" ]; then
    success "Configuration files present"
else
    error "Configuration verification failed"
fi

# Step 5: Start application
log "\n5. Starting application in test environment..."
docker-compose up -d

sleep 15

# Check if application is responding
if curl -f -s http://localhost:8002/health > /dev/null 2>&1; then
    success "Application responding to health checks"
else
    error "Application not responding"
fi

# Step 6: Performance check
log "\n6. Running performance checks..."

RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8002/health)
log "Response time: ${RESPONSE_TIME}s"

if (( $(echo "$RESPONSE_TIME < 2.0" | bc -l) )); then
    success "Response time acceptable"
else
    error "Response time too high"
fi

# Step 7: Calculate RTO
log "\n7. Calculating Recovery Time Objective (RTO)..."

TEST_START=$(head -1 "$LOG_FILE" | grep -oP '\[\K[^\]]+')
TEST_END=$(date +'%Y-%m-%d %H:%M:%S')

START_SECONDS=$(date -d "$TEST_START" +%s)
END_SECONDS=$(date -d "$TEST_END" +%s)
RTO_SECONDS=$((END_SECONDS - START_SECONDS))
RTO_MINUTES=$((RTO_SECONDS / 60))

log "Recovery Time: ${RTO_MINUTES} minutes (${RTO_SECONDS} seconds)"

if [ $RTO_MINUTES -lt 240 ]; then  # 4 hours = 240 minutes
    success "RTO within target (< 4 hours)"
else
    error "RTO exceeds target (> 4 hours)"
fi

# Step 8: Cleanup
log "\n8. Cleaning up test environment..."
docker-compose down

success "Test environment cleaned up"

# Generate summary
log "\n========================================="
log "Disaster Recovery Test Summary"
log "========================================="
log "Backup ID: $BACKUP_ID"
log "RTO Achieved: ${RTO_MINUTES} minutes"
log "RTO Target: 240 minutes (4 hours)"
log "Status: $([ $RTO_MINUTES -lt 240 ] && echo "PASS" || echo "FAIL")"
log "Log file: $LOG_FILE"
log "========================================="

if [ $RTO_MINUTES -lt 240 ]; then
    success "Disaster Recovery Test PASSED"
    exit 0
else
    error "Disaster Recovery Test FAILED"
    exit 1
fi
