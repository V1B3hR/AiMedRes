# Audit and Compliance Logging Guide

## Overview

This guide provides comprehensive procedures for audit log management, compliance reporting, and regulatory audit preparation for the AiMedRes healthcare AI platform.

## Table of Contents

1. [Audit Logging Architecture](#audit-logging-architecture)
2. [Log Review Procedures](#log-review-procedures)
3. [Compliance Reporting](#compliance-reporting)
4. [Audit Preparation](#audit-preparation)
5. [Automated Tools](#automated-tools)

---

## Audit Logging Architecture

### Audit Log Categories

AiMedRes implements comprehensive audit logging across four main categories:

#### 1. Access Logs
**Captures:** User authentication, authorization, and data access events

**Events Logged:**
- User login/logout (successful and failed attempts)
- Session creation and termination
- API authentication attempts
- Role assignments and modifications
- Permission grants and revocations
- Password changes and resets
- MFA enrollment and validation

**Implementation:**
```python
from src.aimedres.security.auth import SecureAuthManager

# Access audit logging is automatic via SecureAuthManager
auth_manager = SecureAuthManager({
    'jwt_secret': os.getenv('JWT_SECRET_KEY'),
    'audit_logging': True,
    'audit_destination': '/var/log/aimedres/access_audit.log'
})
```

#### 2. Data Access Logs
**Captures:** All PHI/PII access and modifications

**Events Logged:**
- Patient record access (read, write, update, delete)
- Data export operations
- Report generation
- Bulk data operations
- PHI scrubbing operations
- Data anonymization/de-identification
- EMR/EHR data synchronization

**Required Fields:**
- Timestamp (ISO 8601 format with timezone)
- User ID and username
- User role
- IP address and location
- Action type (read/write/update/delete)
- Resource type and ID (e.g., patient:12345)
- Data fields accessed
- Operation outcome (success/failure)
- Justification/reason (if required by policy)

**Implementation:**
```python
from security.hipaa_audit import HIPAAAuditLogger

# Initialize HIPAA-compliant audit logger
audit_logger = HIPAAAuditLogger(
    log_file='/var/log/aimedres/data_access_audit.log',
    retention_days=2555,  # 7 years for HIPAA compliance
    encryption_enabled=True
)

# Log data access
audit_logger.log_data_access(
    user_id='dr_smith',
    action='read',
    resource_type='patient',
    resource_id='12345',
    fields_accessed=['demographics', 'medical_history'],
    outcome='success'
)
```

#### 3. Model Operations Logs
**Captures:** AI model training, inference, and maintenance activities

**Events Logged:**
- Model training initiated/completed
- Model inference requests
- Model validation and benchmarking
- Model deployment and version updates
- Model rollback operations
- Hyperparameter tuning
- Dataset usage for training/validation
- Model performance metrics

**Implementation:**
```python
from mlops.monitoring.production_monitor import ProductionMonitor

monitor = ProductionMonitor(
    model_name='alzheimer_v1',
    audit_logging=True
)

# Audit log automatically created for each inference
result = monitor.predict(patient_data)
```

#### 4. System Operations Logs
**Captures:** System administration, configuration changes, and security events

**Events Logged:**
- Configuration changes
- System backups and restores
- Security incidents
- Network access attempts
- Firewall rule changes
- Certificate renewals
- Vulnerability scan results
- Software updates and patches

**Implementation:**
```python
import logging

# System operations logger
sys_logger = logging.getLogger('aimedres.system')
sys_logger.setLevel(logging.INFO)

# Handler for system audit log
handler = logging.handlers.RotatingFileHandler(
    '/var/log/aimedres/system_audit.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=50
)

formatter = logging.Formatter(
    '%(asctime)s|%(levelname)s|%(user)s|%(action)s|%(details)s'
)
handler.setFormatter(formatter)
sys_logger.addHandler(handler)
```

### Log Storage and Retention

**Storage Requirements:**

1. **Primary Storage:**
   - Location: `/var/log/aimedres/`
   - Format: JSON Lines (JSONL) for structured logging
   - Encryption: AES-256 at rest
   - Permissions: 600 (read/write by aimedres service user only)

2. **Long-term Archival:**
   - Location: Write-once storage (WORM) or S3 Glacier
   - Retention: 7 years (HIPAA requirement for audit logs)
   - Encryption: AES-256 with separate encryption keys
   - Integrity: SHA-256 checksums for tamper detection

3. **SIEM Integration:**
   - Protocol: Syslog over TLS
   - Format: CEF (Common Event Format) or JSON
   - Real-time streaming for security events
   - Long-term retention in institutional SIEM

**Retention Policies:**

| Log Type | Primary Retention | Archive Retention | Compliance Requirement |
|----------|------------------|-------------------|------------------------|
| Access Logs | 90 days | 7 years | HIPAA §164.312(b) |
| Data Access Logs | 90 days | 7 years | HIPAA §164.308(a)(1)(ii)(D) |
| Model Operations | 30 days | 3 years | FDA guidance (if applicable) |
| System Operations | 90 days | 3 years | NIST CSF |
| Security Events | 365 days | 7 years | HIPAA Security Rule |

### Log Integrity and Tamper Protection

**Measures Implemented:**

1. **Blockchain Integration:**
   ```python
   from security.blockchain_records import BlockchainAuditLog
   
   # Initialize blockchain audit trail
   blockchain_audit = BlockchainAuditLog(
       network='private',
       consensus='proof-of-authority'
   )
   
   # Record critical audit events in blockchain
   blockchain_audit.record_event({
       'timestamp': datetime.utcnow().isoformat(),
       'event_type': 'data_access',
       'user': 'dr_smith',
       'resource': 'patient:12345',
       'action': 'read'
   })
   ```

2. **Digital Signatures:**
   - Each log entry signed with HMAC-SHA256
   - Private key stored in HSM or KMS
   - Signature verification on log retrieval

3. **Append-only Storage:**
   - Write-once, read-many (WORM) filesystem
   - No delete or modify permissions
   - Immutable audit trail

---

## Log Review Procedures

### Daily Review (Automated)

**Frequency:** Every 24 hours at 2:00 AM

**Automated Checks:**
```bash
#!/bin/bash
# Script: /opt/aimedres/scripts/daily_audit_review.sh

# 1. Failed login attempts (> 5 from same user)
python3 /opt/aimedres/scripts/check_failed_logins.py \
    --threshold 5 \
    --alert-email security@hospital.org

# 2. Unusual data access patterns
python3 /opt/aimedres/scripts/detect_anomalous_access.py \
    --lookback-hours 24 \
    --alert-email compliance@hospital.org

# 3. After-hours access (outside 7 AM - 7 PM)
python3 /opt/aimedres/scripts/check_after_hours_access.py \
    --alert-email security@hospital.org

# 4. Bulk data exports (> 100 patients)
python3 /opt/aimedres/scripts/check_bulk_exports.py \
    --threshold 100 \
    --alert-email privacy-officer@hospital.org

# 5. System errors and exceptions
grep -i "error\|exception\|critical" /var/log/aimedres/app.log \
    | tail -n 100 \
    | mail -s "AiMedRes Daily Errors" devops@hospital.org
```

**Alert Thresholds:**
- Failed logins: > 5 attempts from same user in 1 hour
- After-hours access: Any access between 10 PM - 6 AM
- Bulk exports: > 100 patient records in single export
- Role escalation: Any admin role assignment
- PHI access: > 50 patient records accessed by single user per day

### Weekly Review (Manual)

**Frequency:** Every Monday morning

**Review Checklist:**

1. **Access Patterns Review (30 minutes)**
   - [ ] Review top 10 users by data access volume
   - [ ] Identify any unusual access patterns
   - [ ] Verify access aligns with job responsibilities
   - [ ] Check for shared account usage
   - [ ] Review API key usage and rotation status

2. **Security Events Review (20 minutes)**
   - [ ] Review all failed authentication attempts
   - [ ] Check for brute force attack patterns
   - [ ] Verify no unauthorized access attempts succeeded
   - [ ] Review firewall logs for blocked connections
   - [ ] Check intrusion detection system alerts

3. **Model Operations Review (15 minutes)**
   - [ ] Review model inference volumes and patterns
   - [ ] Check for unusual prediction requests
   - [ ] Verify model performance metrics within thresholds
   - [ ] Review any model deployment or rollback events

4. **Compliance Indicators (15 minutes)**
   - [ ] Verify all data access has proper authorization
   - [ ] Check that PHI scrubber is functioning correctly
   - [ ] Review any policy violations or exceptions
   - [ ] Verify audit log backup completion

**Documentation:**
- Findings recorded in weekly audit log review report
- Template: `deployment/governance/templates/weekly_audit_review.md`
- Storage: `/var/log/aimedres/reviews/weekly/YYYY-MM-DD_review.pdf`

### Monthly Review (Comprehensive)

**Frequency:** First business day of each month

**Review Scope:**

1. **Access Control Audit (1 hour)**
   - Review all user accounts and active status
   - Verify role assignments match current responsibilities
   - Identify and disable inactive accounts (> 90 days)
   - Review privileged access (admin, auditor roles)
   - Audit API keys and service accounts

2. **Data Access Audit (1 hour)**
   - Analyze data access patterns and trends
   - Identify high-volume users and verify legitimacy
   - Review any data exports or bulk operations
   - Check PHI scrubbing effectiveness
   - Verify no data leakage or unauthorized disclosure

3. **Security Posture (1 hour)**
   - Review security incident log
   - Analyze vulnerability scan results
   - Check patch management status
   - Review certificate expiration dates
   - Assess threat intelligence and indicators

4. **Model Governance (30 minutes)**
   - Review model performance metrics
   - Check for model drift or degradation
   - Verify model version control and documentation
   - Review training data provenance
   - Check for bias or fairness issues

5. **Compliance Status (30 minutes)**
   - Verify HIPAA compliance measures
   - Review audit log completeness and integrity
   - Check backup and disaster recovery status
   - Assess regulatory requirement adherence
   - Document any compliance gaps

**Deliverables:**
- Monthly audit report (template: `deployment/governance/templates/monthly_audit_report.md`)
- Executive summary for leadership
- Action items and remediation plan
- Trend analysis and recommendations

### Quarterly Review (Executive)

**Frequency:** End of each calendar quarter

**Executive Review Meeting (2 hours)**

**Attendees:**
- Chief Information Security Officer (CISO)
- Chief Privacy Officer (CPO)
- Chief Medical Officer (CMO)
- IT Director
- Compliance Officer
- AiMedRes Project Lead

**Agenda:**
1. Quarterly metrics and KPIs (30 min)
2. Security and compliance highlights (20 min)
3. Incident review and lessons learned (20 min)
4. Model governance and AI safety (20 min)
5. Regulatory updates and compliance gaps (15 min)
6. Strategic planning and roadmap (15 min)

**Quarterly Metrics:**
- Total user accounts (active/inactive)
- Authentication success/failure rates
- Data access volumes and patterns
- Model inference requests
- Security incidents (by severity)
- Compliance audit findings
- Vulnerability remediation status
- System uptime and availability

---

## Compliance Reporting

### HIPAA Compliance Reports

#### 1. Access Report (§164.308(a)(1)(ii)(D))

**Frequency:** Monthly and on-demand

**Report Contents:**
- All PHI access events
- User identification and role
- Timestamp and duration of access
- Data accessed (patient ID, record type)
- Purpose of access (if documented)
- Authorization basis

**Generation:**
```bash
# Generate HIPAA access report
python3 /opt/aimedres/scripts/generate_hipaa_access_report.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --output /var/reports/hipaa_access_2024-01.pdf
```

#### 2. Security Incident Report (§164.308(a)(6))

**Frequency:** As incidents occur + quarterly summary

**Report Contents:**
- Incident date/time and detection method
- Incident classification and severity
- Affected systems and data
- Root cause analysis
- Containment and remediation actions
- Preventive measures implemented
- Lessons learned

**Template:** `deployment/governance/templates/security_incident_report.md`

#### 3. Risk Assessment Report (§164.308(a)(1)(ii)(A))

**Frequency:** Annually

**Report Contents:**
- Asset inventory and criticality
- Threat and vulnerability assessment
- Current safeguards and controls
- Risk likelihood and impact analysis
- Risk treatment plan
- Residual risk acceptance

**Template:** `deployment/governance/templates/risk_assessment_report.md`

### FDA Compliance (if applicable)

#### Adverse Event Reporting

**Requirement:** Report adverse events within specified timeframes

**Events to Report:**
- Incorrect diagnosis or treatment recommendation
- Patient harm or near-miss incidents
- Software malfunction affecting patient care
- Unexpected model behavior
- Data integrity issues affecting clinical decisions

**Reporting Timeline:**
- Death or serious injury: 30 calendar days
- Malfunction: 30 calendar days
- 5-day report: Life-threatening events

**Process:**
```python
from src.aimedres.compliance.fda import FDAAdverseEventReporter

reporter = FDAAdverseEventReporter()

reporter.report_adverse_event({
    'event_type': 'incorrect_recommendation',
    'severity': 'serious',
    'patient_outcome': 'no_harm',
    'model_version': 'alzheimer_v1',
    'date_occurred': '2024-01-15',
    'description': 'Model recommended incorrect treatment pathway',
    'corrective_action': 'Model temporarily disabled, issue resolved in v1.1'
})
```

### State-Specific Reporting

**Requirements vary by jurisdiction:**

- **California (CCPA/CPRA):** Data breach notification within 72 hours
- **Massachusetts:** Written security program and annual review
- **New York (SHIELD Act):** Data security program requirements
- **Texas:** Medical records security and breach notification

**Implementation:**
```python
from src.aimedres.compliance.regulatory import StateComplianceReporter

reporter = StateComplianceReporter(state='california')

# Automatically generates required reports
reporter.generate_annual_security_report()
reporter.check_breach_notification_requirements()
```

---

## Audit Preparation

### Internal Audit Preparation

**Timeline:** 2 weeks before audit

**Preparation Checklist:**

#### Week 1: Documentation Gathering
- [ ] Compile all audit logs for review period
- [ ] Generate access reports by user and by resource
- [ ] Document all security incidents and resolutions
- [ ] Prepare system configuration documentation
- [ ] Collect evidence of security controls
- [ ] Review and update policies and procedures
- [ ] Prepare list of system changes and updates

#### Week 2: Review and Validation
- [ ] Validate audit log completeness and integrity
- [ ] Test audit log retrieval and search capabilities
- [ ] Review compliance with documented policies
- [ ] Identify and document any gaps or exceptions
- [ ] Prepare remediation plans for identified issues
- [ ] Conduct mock audit with internal team
- [ ] Brief stakeholders on audit scope and process

**Preparation Script:**
```bash
#!/bin/bash
# Script: prepare_for_audit.sh

AUDIT_START_DATE="2024-01-01"
AUDIT_END_DATE="2024-12-31"
OUTPUT_DIR="/var/reports/audit_prep_$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"

# 1. Generate comprehensive access report
python3 generate_hipaa_access_report.py \
    --start-date "$AUDIT_START_DATE" \
    --end-date "$AUDIT_END_DATE" \
    --output "$OUTPUT_DIR/access_report.pdf"

# 2. Generate security incident summary
python3 generate_security_incident_summary.py \
    --start-date "$AUDIT_START_DATE" \
    --end-date "$AUDIT_END_DATE" \
    --output "$OUTPUT_DIR/incident_summary.pdf"

# 3. Generate user access review
python3 generate_user_access_review.py \
    --output "$OUTPUT_DIR/user_access_review.xlsx"

# 4. Export audit logs
python3 export_audit_logs.py \
    --start-date "$AUDIT_START_DATE" \
    --end-date "$AUDIT_END_DATE" \
    --output "$OUTPUT_DIR/audit_logs/"

# 5. Verify log integrity
python3 verify_audit_log_integrity.py \
    --log-dir /var/log/aimedres/ \
    --output "$OUTPUT_DIR/integrity_verification.txt"

# 6. Generate compliance checklist
python3 generate_compliance_checklist.py \
    --framework hipaa \
    --output "$OUTPUT_DIR/compliance_checklist.pdf"

# 7. Create audit evidence package
cd "$OUTPUT_DIR"
tar -czf audit_evidence_$(date +%Y%m%d).tar.gz *

echo "Audit preparation complete. Evidence package: $OUTPUT_DIR/audit_evidence_$(date +%Y%m%d).tar.gz"
```

### External Audit (HIPAA, SOC 2, etc.)

**Timeline:** 4-6 weeks before audit

**Additional Preparation:**

1. **Engage Audit Firm:**
   - Confirm audit scope and standards
   - Provide system documentation and architecture
   - Schedule audit dates and logistics
   - Identify audit liaisons and contacts

2. **Documentation Review:**
   - Security policies and procedures
   - Risk assessment documentation
   - Business associate agreements (BAAs)
   - Training records and certifications
   - Incident response plans
   - Disaster recovery documentation

3. **Technical Preparation:**
   - Prepare system access for auditors (read-only)
   - Set up secure file sharing for evidence
   - Configure audit logging for auditor activities
   - Prepare system demonstrations
   - Document system architecture and data flows

4. **Stakeholder Briefing:**
   - Brief all interviewees on audit process
   - Review roles and responsibilities
   - Align on key messages and talking points
   - Prepare for common audit questions

**Common Audit Questions:**

1. **Access Control:**
   - How do you manage user access and permissions?
   - How often do you review user access rights?
   - How do you handle terminations and role changes?

2. **Data Protection:**
   - How do you protect PHI at rest and in transit?
   - What encryption methods do you use?
   - How do you manage encryption keys?

3. **Audit Logging:**
   - What events do you log and why?
   - How long do you retain audit logs?
   - How do you protect log integrity?
   - Can you demonstrate log search and retrieval?

4. **Incident Response:**
   - Describe your incident response process
   - Provide examples of incidents and resolutions
   - How do you prevent similar incidents?

5. **Vulnerability Management:**
   - How do you identify and remediate vulnerabilities?
   - What is your patch management process?
   - How do you perform security testing?

**Post-Audit:**
- Review audit findings and recommendations
- Develop remediation plan with timelines
- Track remediation progress
- Schedule follow-up audit if required

---

## Automated Tools

### Audit Log Analysis Scripts

All scripts located in: `/opt/aimedres/scripts/audit/`

#### 1. Failed Login Checker
**Script:** `check_failed_logins.py`

```python
#!/usr/bin/env python3
"""
Check for failed login attempts and alert on suspicious activity.
"""

import json
import sys
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_failed_logins(log_file, threshold=5, lookback_hours=24):
    """Analyze failed logins and identify suspicious patterns."""
    
    failed_attempts = defaultdict(list)
    cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('event_type') == 'authentication_failed':
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    if timestamp > cutoff_time:
                        username = entry.get('username', 'unknown')
                        failed_attempts[username].append(entry)
            except json.JSONDecodeError:
                continue
    
    # Identify users exceeding threshold
    alerts = []
    for username, attempts in failed_attempts.items():
        if len(attempts) >= threshold:
            alerts.append({
                'username': username,
                'failed_attempts': len(attempts),
                'first_attempt': attempts[0]['timestamp'],
                'last_attempt': attempts[-1]['timestamp'],
                'ip_addresses': list(set(a.get('ip_address') for a in attempts))
            })
    
    return alerts

if __name__ == '__main__':
    alerts = analyze_failed_logins(
        '/var/log/aimedres/access_audit.log',
        threshold=5,
        lookback_hours=24
    )
    
    if alerts:
        print(f"ALERT: {len(alerts)} users with excessive failed login attempts:")
        for alert in alerts:
            print(f"  - {alert['username']}: {alert['failed_attempts']} attempts")
            print(f"    IP addresses: {', '.join(alert['ip_addresses'])}")
        sys.exit(1)
    else:
        print("No suspicious login activity detected.")
        sys.exit(0)
```

#### 2. Anomalous Access Detector
**Script:** `detect_anomalous_access.py`

```python
#!/usr/bin/env python3
"""
Detect anomalous data access patterns using statistical analysis.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

def detect_anomalies(log_file, lookback_hours=24, std_threshold=3):
    """Detect anomalous data access using z-score analysis."""
    
    user_access_counts = defaultdict(int)
    cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
    
    # Count access events per user
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('event_type') == 'data_access':
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    if timestamp > cutoff_time:
                        user_access_counts[entry['user_id']] += 1
            except json.JSONDecodeError:
                continue
    
    # Calculate mean and standard deviation
    access_counts = list(user_access_counts.values())
    if len(access_counts) < 3:
        return []
    
    mean = np.mean(access_counts)
    std = np.std(access_counts)
    
    # Identify outliers (z-score > threshold)
    anomalies = []
    for user, count in user_access_counts.items():
        z_score = (count - mean) / std if std > 0 else 0
        if z_score > std_threshold:
            anomalies.append({
                'user_id': user,
                'access_count': count,
                'z_score': round(z_score, 2),
                'threshold': std_threshold
            })
    
    return anomalies

if __name__ == '__main__':
    anomalies = detect_anomalies('/var/log/aimedres/data_access_audit.log')
    
    if anomalies:
        print(f"ALERT: {len(anomalies)} users with anomalous access patterns:")
        for anomaly in anomalies:
            print(f"  - {anomaly['user_id']}: {anomaly['access_count']} accesses "
                  f"(z-score: {anomaly['z_score']})")
    else:
        print("No anomalous access patterns detected.")
```

#### 3. Bulk Export Checker
**Script:** `check_bulk_exports.py`

```python
#!/usr/bin/env python3
"""
Monitor and alert on bulk data export operations.
"""

import json
from datetime import datetime, timedelta

def check_bulk_exports(log_file, threshold=100, lookback_hours=24):
    """Check for large data export operations."""
    
    large_exports = []
    cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('action') == 'export':
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    record_count = entry.get('record_count', 0)
                    
                    if timestamp > cutoff_time and record_count >= threshold:
                        large_exports.append({
                            'timestamp': entry['timestamp'],
                            'user_id': entry['user_id'],
                            'record_count': record_count,
                            'export_type': entry.get('export_type'),
                            'justification': entry.get('justification', 'None provided')
                        })
            except json.JSONDecodeError:
                continue
    
    return large_exports

if __name__ == '__main__':
    exports = check_bulk_exports('/var/log/aimedres/data_access_audit.log')
    
    if exports:
        print(f"ALERT: {len(exports)} large data export operations detected:")
        for export in exports:
            print(f"  - {export['timestamp']}: {export['user_id']} exported "
                  f"{export['record_count']} records")
            print(f"    Justification: {export['justification']}")
    else:
        print("No large export operations detected.")
```

### Compliance Report Generators

#### Generate HIPAA Access Report
**Script:** `generate_hipaa_access_report.py`

```python
#!/usr/bin/env python3
"""
Generate HIPAA-compliant access report.
"""

import json
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_hipaa_access_report(log_file, start_date, end_date, output_file):
    """Generate comprehensive HIPAA access report."""
    
    # Parse dates
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    # Extract access events
    access_events = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('event_type') == 'data_access':
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    if start <= timestamp <= end:
                        access_events.append([
                            entry['timestamp'],
                            entry.get('user_id', 'Unknown'),
                            entry.get('user_role', 'Unknown'),
                            entry.get('resource_type', 'Unknown'),
                            entry.get('resource_id', 'Unknown'),
                            entry.get('action', 'Unknown'),
                            entry.get('outcome', 'Unknown')
                        ])
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Create PDF report
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"<b>HIPAA Access Report</b><br/>"
                     f"Period: {start_date} to {end_date}", styles['Title'])
    elements.append(title)
    elements.append(Paragraph("<br/><br/>", styles['Normal']))
    
    # Summary statistics
    summary = f"Total Access Events: {len(access_events)}"
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Paragraph("<br/>", styles['Normal']))
    
    # Access table
    if access_events:
        table_data = [['Timestamp', 'User', 'Role', 'Resource Type', 
                      'Resource ID', 'Action', 'Outcome']] + access_events[:100]
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#cccccc'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, '#000000')
        ]))
        elements.append(table)
    
    # Build PDF
    doc.build(elements)
    print(f"HIPAA access report generated: {output_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print("Usage: generate_hipaa_access_report.py <log_file> <start_date> <end_date> <output_file>")
        sys.exit(1)
    
    generate_hipaa_access_report(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
```

### Automated Compliance Monitoring

**Cron Configuration:**

```bash
# /etc/cron.d/aimedres-audit

# Daily audit checks (2 AM)
0 2 * * * aimedres /opt/aimedres/scripts/audit/daily_audit_review.sh

# Weekly audit review reminder (Monday 8 AM)
0 8 * * 1 aimedres /opt/aimedres/scripts/audit/send_weekly_review_reminder.sh

# Monthly compliance report (1st of month, 6 AM)
0 6 1 * * aimedres /opt/aimedres/scripts/audit/generate_monthly_compliance_report.sh

# Continuous log integrity check (every 4 hours)
0 */4 * * * aimedres /opt/aimedres/scripts/audit/verify_log_integrity.sh
```

---

## Summary

This audit and compliance logging guide provides:

1. **Comprehensive audit logging** across access, data, model operations, and system events
2. **Structured review procedures** (daily, weekly, monthly, quarterly)
3. **Compliance reporting** for HIPAA, FDA, and state-specific requirements
4. **Audit preparation checklists** for internal and external audits
5. **Automated tools** for log analysis and anomaly detection

**Key Compliance Standards Met:**
- ✅ HIPAA §164.308(a)(1)(ii)(D) - Audit controls
- ✅ HIPAA §164.312(b) - Audit controls technical specification
- ✅ NIST CSF - Detect and Respond functions
- ✅ SOC 2 Type II - Logging and monitoring controls
- ✅ FDA 21 CFR Part 11 - Audit trail requirements (if applicable)

**Next Steps:**
1. Configure audit logging for your environment
2. Set up automated daily/weekly/monthly reviews
3. Schedule first comprehensive audit
4. Train staff on audit procedures and tools
5. Integrate with institutional SIEM/compliance systems
