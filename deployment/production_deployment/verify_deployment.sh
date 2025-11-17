#!/bin/bash
#
# Post-Deployment Verification Script
# Validates that all services are running correctly after deployment
#

set -e

# Configuration
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-8002}"
API_URL="http://${API_HOST}:${API_PORT}"
LOG_FILE="${LOG_FILE:-/var/log/aimedres/verification.log}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

pass() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
    PASSED=$((PASSED + 1))
}

fail() {
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
}

# Check health endpoint
check_health() {
    log "Checking health endpoint..."
    
    if curl -f -s "${API_URL}/health" > /dev/null 2>&1; then
        pass "Health endpoint responding"
        return 0
    else
        fail "Health endpoint not responding"
        return 1
    fi
}

# Check API root endpoint
check_api_root() {
    log "Checking API root endpoint..."
    
    if curl -f -s "${API_URL}/" > /dev/null 2>&1; then
        pass "API root endpoint responding"
        return 0
    else
        fail "API root endpoint not responding"
        return 1
    fi
}

# Check model loading
check_models() {
    log "Checking if models are loaded..."
    
    # Try to access models endpoint (may require authentication)
    local response=$(curl -s "${API_URL}/api/v1/models" 2>&1)
    
    if echo "$response" | grep -q "models\|alzheimer\|parkinsons"; then
        pass "Models endpoint accessible"
        return 0
    else
        fail "Models endpoint not accessible or models not loaded"
        return 1
    fi
}

# Check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    # Check if postgres container is running
    if docker ps | grep -q postgres; then
        pass "Database container running"
        
        # Try to connect to database
        if docker exec $(docker ps -q -f name=postgres) pg_isready -U aimedres > /dev/null 2>&1; then
            pass "Database accepting connections"
            return 0
        else
            fail "Database not accepting connections"
            return 1
        fi
    else
        fail "Database container not running"
        return 1
    fi
}

# Check Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."
    
    if docker ps | grep -q redis; then
        pass "Redis container running"
        
        # Try to ping Redis
        if docker exec $(docker ps -q -f name=redis) redis-cli ping | grep -q PONG; then
            pass "Redis responding to PING"
            return 0
        else
            fail "Redis not responding"
            return 1
        fi
    else
        fail "Redis container not running"
        return 1
    fi
}

# Check logs for errors
check_logs() {
    log "Checking application logs for errors..."
    
    local error_count=$(docker logs $(docker ps -q -f name=aimedres) 2>&1 | grep -i "error\|exception\|critical" | wc -l)
    
    if [ $error_count -eq 0 ]; then
        pass "No errors found in recent logs"
        return 0
    elif [ $error_count -lt 5 ]; then
        fail "Found $error_count errors in logs (review recommended)"
        return 1
    else
        fail "Found $error_count errors in logs (CRITICAL)"
        return 1
    fi
}

# Check resource utilization
check_resources() {
    log "Checking resource utilization..."
    
    # Check CPU usage
    local cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" $(docker ps -q -f name=aimedres) | sed 's/%//')
    
    if [ -n "$cpu_usage" ]; then
        if (( $(echo "$cpu_usage < 80" | bc -l) )); then
            pass "CPU usage: ${cpu_usage}% (acceptable)"
        else
            fail "CPU usage: ${cpu_usage}% (HIGH)"
        fi
    fi
    
    # Check memory usage
    local mem_usage=$(docker stats --no-stream --format "{{.MemPerc}}" $(docker ps -q -f name=aimedres) | sed 's/%//')
    
    if [ -n "$mem_usage" ]; then
        if (( $(echo "$mem_usage < 85" | bc -l) )); then
            pass "Memory usage: ${mem_usage}% (acceptable)"
        else
            fail "Memory usage: ${mem_usage}% (HIGH)"
        fi
    fi
}

# Main verification flow
main() {
    log "========================================="
    log "Starting Post-Deployment Verification"
    log "========================================="
    
    # Run all checks
    check_health
    check_api_root
    check_models
    check_database
    check_redis
    check_logs
    check_resources
    
    # Summary
    log "========================================="
    log "Verification Results:"
    log "Passed: $PASSED"
    log "Failed: $FAILED"
    log "========================================="
    
    if [ $FAILED -eq 0 ]; then
        log "✓ All verification checks passed!"
        exit 0
    else
        log "✗ Some verification checks failed. Please review."
        exit 1
    fi
}

main "$@"
