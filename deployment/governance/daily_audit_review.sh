#!/bin/bash
#
# Daily Audit Log Review Script
# Automated checks for security and compliance issues
#

set -e

# Configuration
AUDIT_LOG_DIR="${AUDIT_LOG_DIR:-/var/log/aimedres/audit}"
OUTPUT_DIR="${OUTPUT_DIR:-/var/aimedres/audit_reports}"
ALERT_EMAIL="${ALERT_EMAIL:-security@hospital.org}"

mkdir -p "$OUTPUT_DIR"

DATE=$(date +%Y-%m-%d)
REPORT_FILE="$OUTPUT_DIR/daily_audit_${DATE}.txt"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$REPORT_FILE"
}

alert() {
    echo "🚨 ALERT: $1" | tee -a "$REPORT_FILE"
}

check() {
    echo "✓ $1" | tee -a "$REPORT_FILE"
}

log "========================================="
log "Daily Audit Log Review"
log "Date: $DATE"
log "========================================="

# Check for failed login attempts
log "\n1. Checking for failed login attempts..."
FAILED_LOGINS=$(grep -c "login.*failed" "$AUDIT_LOG_DIR"/*.log 2>/dev/null || echo 0)
if [ "$FAILED_LOGINS" -gt 5 ]; then
    alert "High number of failed logins: $FAILED_LOGINS (threshold: 5)"
else
    check "Failed logins: $FAILED_LOGINS (acceptable)"
fi

# Check for unusual data access patterns
log "\n2. Checking for unusual data access patterns..."
BULK_EXPORTS=$(grep -c "bulk.*export" "$AUDIT_LOG_DIR"/*.log 2>/dev/null || echo 0)
if [ "$BULK_EXPORTS" -gt 0 ]; then
    alert "Bulk data exports detected: $BULK_EXPORTS"
else
    check "No bulk exports detected"
fi

# Check for after-hours access
log "\n3. Checking for after-hours access..."
AFTER_HOURS=$(grep -E "(2[2-3]|0[0-5]):[0-5][0-9]" "$AUDIT_LOG_DIR"/*.log | wc -l)
if [ "$AFTER_HOURS" -gt 10 ]; then
    alert "High after-hours access: $AFTER_HOURS events"
else
    check "After-hours access: $AFTER_HOURS events (acceptable)"
fi

# Check for system errors
log "\n4. Checking for system errors..."
ERRORS=$(grep -ic "error\|exception\|critical" "$AUDIT_LOG_DIR"/*.log 2>/dev/null || echo 0)
if [ "$ERRORS" -gt 20 ]; then
    alert "High error count: $ERRORS (threshold: 20)"
else
    check "System errors: $ERRORS (acceptable)"
fi

# Check for security events
log "\n5. Checking for security events..."
SECURITY_EVENTS=$(grep -ic "security.*event\|breach\|unauthorized" "$AUDIT_LOG_DIR"/*.log 2>/dev/null || echo 0)
if [ "$SECURITY_EVENTS" -gt 0 ]; then
    alert "Security events detected: $SECURITY_EVENTS"
else
    check "No security events detected"
fi

log "\n========================================="
log "Daily Audit Review Complete"
log "Report saved to: $REPORT_FILE"
log "========================================="

# Send email if alerts were generated
if grep -q "🚨 ALERT" "$REPORT_FILE"; then
    log "Alerts detected - sending notification email"
    if command -v mail &> /dev/null; then
        mail -s "AiMedRes Daily Audit Alert - $DATE" "$ALERT_EMAIL" < "$REPORT_FILE"
    else
        log "WARNING: mail command not available, email not sent"
    fi
fi
