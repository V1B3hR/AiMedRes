# Incident Management Guide

## Overview

This guide establishes Standard Operating Procedures (SOPs) for managing security events, data breaches, adverse clinical outcomes, and system incidents for the AiMedRes healthcare AI platform.

## Table of Contents

1. [Incident Classification](#incident-classification)
2. [Incident Response Team](#incident-response-team)
3. [Security Incident Management](#security-incident-management)
4. [Data Breach Response](#data-breach-response)
5. [Adverse Outcome Management](#adverse-outcome-management)
6. [System Incident Management](#system-incident-management)
7. [Communication Plans](#communication-plans)
8. [Post-Incident Review](#post-incident-review)

---

## Incident Classification

### Incident Severity Levels

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| **P1 - Critical** | Immediate patient safety risk or complete system failure | 15 minutes | Patient harm, complete system outage, active security breach |
| **P2 - High** | Significant functionality impact or security concern | 1 hour | Major feature failure, suspected breach, data integrity issue |
| **P3 - Medium** | Limited functionality impact with workaround | 4 hours | Minor feature bug, performance degradation, isolated errors |
| **P4 - Low** | Minimal impact, cosmetic issues | 24 hours | UI glitches, documentation errors, minor enhancements |

### Incident Types

#### 1. Security Incidents
- Unauthorized access attempts
- Malware detection
- DDoS attacks
- Insider threats
- Vulnerability exploitation
- Social engineering attacks

#### 2. Data Breaches
- PHI/PII unauthorized disclosure
- Data exfiltration
- Accidental data exposure
- Lost/stolen devices with data
- Improper data disposal

#### 3. Adverse Clinical Outcomes
- Incorrect diagnosis suggestion
- Treatment recommendation errors
- Model prediction errors with patient impact
- Near-miss incidents
- Patient harm events

#### 4. System Incidents
- Application crashes
- Service outages
- Performance degradation
- Database failures
- Network disruptions
- Infrastructure failures

---

## Incident Response Team

### Core Team Structure

**Incident Commander (IC):**
- **Primary:** IT Director
- **Backup:** Senior DevOps Engineer
- **Responsibilities:** Overall incident coordination, decision-making, escalation

**Technical Lead:**
- **Primary:** ML/AI Team Lead
- **Backup:** Senior ML Engineer
- **Responsibilities:** Technical investigation, root cause analysis, remediation

**Security Lead:**
- **Primary:** Chief Information Security Officer (CISO)
- **Backup:** Security Engineer
- **Responsibilities:** Security assessment, threat containment, forensics

**Clinical Lead:**
- **Primary:** Chief Medical Informatics Officer
- **Backup:** Clinical Champion
- **Responsibilities:** Clinical impact assessment, patient safety evaluation

**Communications Lead:**
- **Primary:** Communications Director
- **Backup:** PR Manager
- **Responsibilities:** Stakeholder communications, media relations, notifications

**Compliance Lead:**
- **Primary:** Chief Privacy Officer
- **Backup:** Compliance Officer
- **Responsibilities:** Regulatory compliance, breach notification, legal liaison

### Contact Information

```
EMERGENCY HOTLINE: 1-800-AIMEDRES (1-800-246-3373)

Incident Commander:
- Primary: +1-XXX-XXX-XXXX (available 24/7)
- Email: incident-commander@hospital.org
- Pager: XXXX

Technical Lead:
- Primary: +1-XXX-XXX-XXXX
- Email: tech-lead@hospital.org
- Slack: @tech-lead

Security Lead:
- Primary: +1-XXX-XXX-XXXX
- Email: ciso@hospital.org
- Pager: XXXX

Clinical Lead:
- Primary: +1-XXX-XXX-XXXX
- Email: cmo@hospital.org

Communications Lead:
- Primary: +1-XXX-XXX-XXXX
- Email: communications@hospital.org

Compliance Lead:
- Primary: +1-XXX-XXX-XXXX
- Email: privacy-officer@hospital.org
```

### Escalation Path

```
User/Monitor → Help Desk (Tier 1) → On-Call Engineer (Tier 2) → Incident Commander → Executive Team
                                                                        ↓
                                    Specialized Team Members (Security, Clinical, Compliance) as needed
```

---

## Security Incident Management

### Detection and Alerting

**Automated Detection:**

1. **Intrusion Detection System (IDS):**
   ```bash
   # Snort/Suricata rules for AiMedRes
   alert tcp any any -> $AIMEDRES_SERVERS $HTTP_PORTS (msg:"Potential SQL Injection"; 
       flow:to_server,established; content:"union"; nocase; sid:100001;)
   
   alert tcp any any -> $AIMEDRES_SERVERS any (msg:"Suspicious Failed Login Pattern"; 
       threshold:type threshold, track by_src, count 5, seconds 60; sid:100002;)
   ```

2. **SIEM Correlation Rules:**
   ```python
   # Example SIEM rule in Python (for Splunk/ELK)
   def detect_brute_force():
       """Detect potential brute force attacks."""
       query = """
       SELECT user_id, COUNT(*) as failed_attempts
       FROM access_logs
       WHERE event_type = 'authentication_failed'
         AND timestamp > NOW() - INTERVAL 1 HOUR
       GROUP BY user_id
       HAVING COUNT(*) > 5
       """
       
       results = execute_query(query)
       if results:
           trigger_alert(
               severity='high',
               title='Potential Brute Force Attack Detected',
               details=results
           )
   ```

3. **Behavioral Anomaly Detection:**
   ```python
   from security.monitoring import AnomalyDetector
   
   detector = AnomalyDetector()
   
   # Detect unusual data access patterns
   if detector.detect_anomalous_access(user_id='dr_smith'):
       create_security_incident(
           type='anomalous_access',
           user='dr_smith',
           severity='medium'
       )
   ```

**Manual Reporting:**

- Security hotline: 1-800-SECURITY
- Email: security@hospital.org
- Web form: https://security.hospital.org/report
- In-person: Security Operations Center (SOC)

### Response Procedure

**Phase 1: Initial Response (0-15 minutes)**

```bash
#!/bin/bash
# Script: initial_security_response.sh

INCIDENT_ID=$1
INCIDENT_TYPE=$2

echo "=== SECURITY INCIDENT RESPONSE ==="
echo "Incident ID: $INCIDENT_ID"
echo "Type: $INCIDENT_TYPE"
echo "Time: $(date)"

# 1. Alert incident response team
python3 /opt/aimedres/scripts/alert_incident_team.py \
    --incident-id "$INCIDENT_ID" \
    --severity critical \
    --type security

# 2. Create incident war room
python3 /opt/aimedres/scripts/create_incident_channel.py \
    --incident-id "$INCIDENT_ID" \
    --team security

# 3. Capture initial evidence
mkdir -p /var/incidents/$INCIDENT_ID/evidence
cp -r /var/log/aimedres/* /var/incidents/$INCIDENT_ID/evidence/
netstat -an > /var/incidents/$INCIDENT_ID/evidence/netstat.txt
ps aux > /var/incidents/$INCIDENT_ID/evidence/processes.txt

# 4. Activate containment if needed
case "$INCIDENT_TYPE" in
    "active_breach")
        echo "ACTIVE BREACH - Initiating containment"
        /opt/aimedres/scripts/isolate_system.sh
        ;;
    "malware")
        echo "MALWARE DETECTED - Isolating affected systems"
        /opt/aimedres/scripts/quarantine_host.sh
        ;;
esac

echo "Initial response complete. Incident Commander notified."
```

**Phase 2: Containment (15 minutes - 2 hours)**

**Containment Actions by Incident Type:**

1. **Unauthorized Access:**
   - Immediately disable compromised accounts
   - Force password reset for all users
   - Review and lock down access controls
   - Enable enhanced logging

2. **Malware Detection:**
   - Isolate infected systems from network
   - Block malware signatures at firewall
   - Scan all systems for infections
   - Restore from clean backups if needed

3. **DDoS Attack:**
   - Activate DDoS mitigation service
   - Rate limit affected endpoints
   - Block attacking IP addresses
   - Scale infrastructure if needed

**Containment Script Example:**

```python
#!/usr/bin/env python3
"""
Security incident containment automation.
"""

from src.aimedres.security.auth import SecureAuthManager
from security.monitoring import SecurityMonitor

def contain_unauthorized_access(incident_details):
    """Contain unauthorized access incident."""
    
    auth_manager = SecureAuthManager()
    security_monitor = SecurityMonitor()
    
    # 1. Disable compromised accounts
    compromised_users = incident_details.get('compromised_users', [])
    for user in compromised_users:
        auth_manager.disable_user(user, reason='Security incident - unauthorized access')
        print(f"Disabled user: {user}")
    
    # 2. Revoke all active sessions
    if incident_details.get('revoke_all_sessions'):
        auth_manager.revoke_all_sessions()
        print("All active sessions revoked")
    
    # 3. Enable enhanced logging
    security_monitor.enable_enhanced_logging(duration_hours=24)
    print("Enhanced logging enabled for 24 hours")
    
    # 4. Notify affected users
    for user in compromised_users:
        send_security_notification(
            user=user,
            incident_type='account_compromise',
            action_required='password_reset'
        )
    
    print("Containment actions complete")

def contain_data_exfiltration(incident_details):
    """Contain data exfiltration incident."""
    
    # 1. Block suspicious IPs
    suspicious_ips = incident_details.get('suspicious_ips', [])
    for ip in suspicious_ips:
        block_ip_address(ip)
        print(f"Blocked IP: {ip}")
    
    # 2. Disable external data transfers
    disable_external_transfers(duration_hours=2)
    print("External data transfers disabled temporarily")
    
    # 3. Snapshot current state for forensics
    create_forensic_snapshot()
    print("Forensic snapshot created")
    
    # 4. Activate DLP rules
    activate_strict_dlp_rules()
    print("Strict DLP rules activated")

if __name__ == '__main__':
    import json
    import sys
    
    incident_file = sys.argv[1]
    with open(incident_file, 'r') as f:
        incident = json.load(f)
    
    if incident['type'] == 'unauthorized_access':
        contain_unauthorized_access(incident)
    elif incident['type'] == 'data_exfiltration':
        contain_data_exfiltration(incident)
    else:
        print(f"No containment procedure defined for: {incident['type']}")
```

**Phase 3: Investigation (2-24 hours)**

**Investigation Checklist:**

- [ ] Identify attack vector and entry point
- [ ] Determine scope of compromise
- [ ] Collect and preserve evidence
- [ ] Analyze logs and system artifacts
- [ ] Interview relevant personnel
- [ ] Document timeline of events
- [ ] Identify affected systems and data
- [ ] Assess damage and impact

**Forensic Evidence Collection:**

```bash
#!/bin/bash
# Script: collect_forensic_evidence.sh

INCIDENT_ID=$1
EVIDENCE_DIR="/var/incidents/$INCIDENT_ID/forensics"
mkdir -p "$EVIDENCE_DIR"

# 1. System information
uname -a > "$EVIDENCE_DIR/system_info.txt"
hostname > "$EVIDENCE_DIR/hostname.txt"
date > "$EVIDENCE_DIR/collection_time.txt"

# 2. Network connections
netstat -an > "$EVIDENCE_DIR/network_connections.txt"
ss -tulpn > "$EVIDENCE_DIR/listening_ports.txt"
arp -a > "$EVIDENCE_DIR/arp_table.txt"

# 3. Running processes
ps auxf > "$EVIDENCE_DIR/processes.txt"
lsof > "$EVIDENCE_DIR/open_files.txt"

# 4. User activity
w > "$EVIDENCE_DIR/logged_in_users.txt"
last -50 > "$EVIDENCE_DIR/login_history.txt"

# 5. System logs
cp -r /var/log/aimedres "$EVIDENCE_DIR/application_logs/"
cp -r /var/log/auth.log "$EVIDENCE_DIR/auth.log"
cp -r /var/log/syslog "$EVIDENCE_DIR/syslog"

# 6. Database queries (if applicable)
pg_dump aimedres_db > "$EVIDENCE_DIR/database_snapshot.sql"

# 7. Memory dump (if needed for malware analysis)
# dd if=/dev/mem of="$EVIDENCE_DIR/memory_dump.raw" bs=1M

# 8. Create evidence hash
find "$EVIDENCE_DIR" -type f -exec sha256sum {} \; > "$EVIDENCE_DIR/evidence_hashes.txt"

# 9. Package evidence
tar -czf "/var/incidents/$INCIDENT_ID/${INCIDENT_ID}_forensics_$(date +%Y%m%d_%H%M%S).tar.gz" "$EVIDENCE_DIR"

echo "Forensic evidence collected: /var/incidents/$INCIDENT_ID/"
```

**Phase 4: Eradication (4-48 hours)**

**Eradication Actions:**

1. **Remove Malware:**
   - Delete malicious files
   - Clean infected systems
   - Verify removal with security scans

2. **Close Vulnerabilities:**
   - Patch exploited vulnerabilities
   - Update configurations
   - Harden security controls

3. **Remove Attacker Access:**
   - Delete unauthorized accounts
   - Revoke compromised credentials
   - Close backdoors and persistent access

**Phase 5: Recovery (1-7 days)**

**Recovery Steps:**

1. Restore from clean backups if needed
2. Rebuild compromised systems
3. Re-enable services gradually
4. Monitor for re-infection or re-compromise
5. Conduct security testing
6. Return to normal operations

**Recovery Validation:**

```python
def validate_recovery(incident_id):
    """Validate system recovery after security incident."""
    
    validation_checks = {
        'malware_scan': run_malware_scan(),
        'vulnerability_scan': run_vulnerability_scan(),
        'access_controls': verify_access_controls(),
        'log_integrity': verify_log_integrity(),
        'backup_integrity': verify_backup_integrity(),
        'security_monitoring': verify_security_monitoring()
    }
    
    all_passed = all(validation_checks.values())
    
    if all_passed:
        print(f"Recovery validated for incident {incident_id}")
        update_incident_status(incident_id, 'recovered')
        return True
    else:
        failed_checks = [k for k, v in validation_checks.items() if not v]
        print(f"Recovery validation failed: {failed_checks}")
        return False
```

---

## Data Breach Response

### HIPAA Breach Notification Requirements

**Breach Definition (HIPAA):**
Unauthorized acquisition, access, use, or disclosure of PHI that compromises security or privacy.

**Notification Timelines:**

| Affected Individuals | Notification Deadline | Method |
|---------------------|----------------------|--------|
| < 500 in one jurisdiction | 60 days | Written notice by mail |
| ≥ 500 in one jurisdiction | 60 days | Written notice + Media notification + HHS notification |
| All breaches | 60 days | Submit to HHS (annually if < 500, immediately if ≥ 500) |

### Breach Assessment

**Phase 1: Breach Determination (24-48 hours)**

```python
#!/usr/bin/env python3
"""
HIPAA breach risk assessment tool.
"""

class BreachRiskAssessment:
    """Assess if incident constitutes HIPAA breach."""
    
    def __init__(self, incident_details):
        self.incident = incident_details
        self.risk_factors = {
            'phi_type': 0,
            'phi_sensitivity': 0,
            'who_accessed': 0,
            'safeguards': 0
        }
    
    def assess_phi_type(self):
        """Assess type and amount of PHI involved."""
        phi_types = self.incident.get('phi_types', [])
        
        # High-risk PHI types
        high_risk = ['ssn', 'financial', 'medical_records', 'genetic_data']
        if any(t in high_risk for t in phi_types):
            self.risk_factors['phi_type'] = 3  # High risk
        elif phi_types:
            self.risk_factors['phi_type'] = 2  # Medium risk
        else:
            self.risk_factors['phi_type'] = 1  # Low risk
    
    def assess_sensitivity(self):
        """Assess sensitivity of exposed information."""
        conditions = self.incident.get('conditions', [])
        
        # Sensitive conditions (e.g., mental health, HIV, substance abuse)
        sensitive = ['mental_health', 'hiv', 'substance_abuse', 'sexual_health']
        if any(c in sensitive for c in conditions):
            self.risk_factors['phi_sensitivity'] = 3
        elif conditions:
            self.risk_factors['phi_sensitivity'] = 2
        else:
            self.risk_factors['phi_sensitivity'] = 1
    
    def assess_unauthorized_access(self):
        """Assess who had unauthorized access."""
        accessor_type = self.incident.get('accessor_type')
        
        if accessor_type == 'external':
            self.risk_factors['who_accessed'] = 3  # External party
        elif accessor_type == 'internal_unauthorized':
            self.risk_factors['who_accessed'] = 2  # Internal but unauthorized
        elif accessor_type == 'internal_authorized':
            self.risk_factors['who_accessed'] = 1  # Internal authorized (lower risk)
    
    def assess_safeguards(self):
        """Assess if safeguards reduce breach risk."""
        encrypted = self.incident.get('encrypted', False)
        limited_view = self.incident.get('limited_view', False)
        no_download = self.incident.get('no_download', False)
        
        if encrypted:
            self.risk_factors['safeguards'] = 1  # Encryption significantly reduces risk
        elif limited_view and no_download:
            self.risk_factors['safeguards'] = 2  # Some safeguards
        else:
            self.risk_factors['safeguards'] = 3  # Minimal safeguards
    
    def determine_breach(self):
        """Determine if incident is a reportable breach."""
        self.assess_phi_type()
        self.assess_sensitivity()
        self.assess_unauthorized_access()
        self.assess_safeguards()
        
        total_risk = sum(self.risk_factors.values())
        
        # Risk scoring:
        # 4-6: Low probability of breach (no notification)
        # 7-9: Medium probability (consult legal)
        # 10-12: High probability (reportable breach)
        
        if total_risk >= 10:
            return {
                'is_breach': True,
                'confidence': 'high',
                'recommendation': 'Reportable breach - initiate notification procedures'
            }
        elif total_risk >= 7:
            return {
                'is_breach': 'uncertain',
                'confidence': 'medium',
                'recommendation': 'Consult legal counsel and privacy officer'
            }
        else:
            return {
                'is_breach': False,
                'confidence': 'high',
                'recommendation': 'Low probability of breach - document assessment'
            }

# Example usage
if __name__ == '__main__':
    incident = {
        'phi_types': ['medical_records', 'demographics'],
        'conditions': ['alzheimer', 'diabetes'],
        'accessor_type': 'external',
        'encrypted': False,
        'affected_count': 150
    }
    
    assessment = BreachRiskAssessment(incident)
    result = assessment.determine_breach()
    
    print(f"Is Breach: {result['is_breach']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Recommendation: {result['recommendation']}")
```

### Breach Notification Process

**Phase 2: Individual Notification (within 60 days)**

**Notification Template:**

```
[Hospital/Organization Letterhead]

[Date]

[Patient Name]
[Address]

Re: Notice of Data Breach

Dear [Patient Name],

We are writing to inform you of a data security incident that may have affected your protected health information.

What Happened:
On [date], we discovered [brief description of incident]. We immediately began an investigation and took steps to address the situation.

What Information Was Involved:
The information that may have been accessed includes: [list specific data elements: name, date of birth, medical record number, diagnosis, etc.].

What We Are Doing:
We have [describe containment and remediation actions]. We are also [describe preventive measures].

What You Can Do:
[Provide recommendations: monitor medical records, place fraud alerts, etc.]

For More Information:
If you have questions, please contact us at [phone number] or [email address]. Additional information is available at [website].

We sincerely apologize for this incident and any concern it may cause.

Sincerely,

[Name]
[Title]
[Organization]
```

**Phase 3: HHS Notification**

**If ≥ 500 individuals affected:**

1. Submit breach notification to HHS within 60 days
2. Use HHS breach reporting portal: https://ocrportal.hhs.gov/ocr/breach
3. Provide required information:
   - Number of individuals affected
   - Types of information involved
   - Brief description of incident
   - Actions taken to mitigate harm

**If < 500 individuals affected:**

1. Maintain log of breaches
2. Submit annual report to HHS by March 1 of following year

**Phase 4: Media Notification (if ≥ 500 in same jurisdiction)**

**Media Notice Requirements:**

- Provide notice to prominent media outlets
- Same content as individual notification
- Within 60 days of breach discovery

### State-Specific Breach Notification

**California (Example):**

- **Timeline:** "Without unreasonable delay"
- **Method:** Written or electronic notification
- **Attorney General:** If > 500 California residents affected

**Massachusetts (Example):**

- **Timeline:** "As soon as practicable" and without unreasonable delay
- **Method:** Written notification
- **Attorney General + Director of Consumer Affairs:** Must be notified

**Implementation:**

```python
from src.aimedres.compliance.regulatory import StateBreachNotification

notifier = StateBreachNotification()

# Determine notification requirements by affected states
affected_states = ['CA', 'MA', 'NY', 'TX']
requirements = notifier.get_notification_requirements(affected_states)

# Generate and send notifications
notifier.send_notifications(
    affected_individuals=affected_individuals,
    incident_details=incident_details,
    requirements=requirements
)
```

---

## Adverse Outcome Management

### Clinical Incident Classification

**Near Miss:** Potential for patient harm but no harm occurred
**No Harm Event:** Incident reached patient but no harm
**Adverse Event:** Patient harm occurred
**Serious Adverse Event:** Death or serious injury

### Adverse Event Response Procedure

**Phase 1: Immediate Response (0-2 hours)**

1. **Ensure Patient Safety:**
   - Assess patient status
   - Provide immediate medical intervention if needed
   - Document current condition

2. **Incident Reporting:**
   ```python
   from src.aimedres.compliance.fda import FDAAdverseEventReporter
   
   reporter = FDAAdverseEventReporter()
   
   reporter.report_adverse_event({
       'event_id': 'AE-2024-001',
       'event_type': 'incorrect_prediction',
       'severity': 'serious',
       'patient_outcome': 'no_harm',
       'model_version': 'alzheimer_v1',
       'date_occurred': '2024-01-15',
       'description': 'Model provided false positive result',
       'clinical_action': 'Clinician identified error, ordered additional testing',
       'patient_impact': 'No harm - caught before treatment decision'
   })
   ```

3. **Alert Response Team:**
   - Notify Incident Commander
   - Activate clinical response team
   - Involve patient safety officer

**Phase 2: Investigation (2-48 hours)**

1. **Root Cause Analysis:**
   - Review model input data
   - Examine model prediction and confidence
   - Analyze clinician workflow
   - Interview involved staff
   - Review similar cases

2. **Clinical Review:**
   - Clinical champion reviews case
   - Determine if model performed within specifications
   - Assess if clinician followed protocols
   - Identify contributing factors

**Root Cause Analysis Template:**

```markdown
## Adverse Event Root Cause Analysis

**Event ID:** AE-2024-001
**Date:** 2024-01-15
**Model:** Alzheimer v1.0.0
**Severity:** Serious / No Harm

### Event Summary
[Brief description of what happened]

### Timeline
- [Time]: [Event]
- [Time]: [Event]

### Root Cause Investigation

**5 Whys Analysis:**
1. Why did the adverse event occur? [Answer]
2. Why did [answer from 1] happen? [Answer]
3. Why did [answer from 2] happen? [Answer]
4. Why did [answer from 3] happen? [Answer]
5. Why did [answer from 4] happen? [Answer - Root Cause]

**Contributing Factors:**
- Human factors: [Description]
- System factors: [Description]
- Model factors: [Description]
- Process factors: [Description]

### Root Cause
[Identified root cause]

### Corrective Actions
1. [Immediate action taken]
2. [Short-term corrective action]
3. [Long-term preventive action]

### Follow-up
- Responsible Party: [Name]
- Due Date: [Date]
- Verification Method: [How to verify effectiveness]
```

**Phase 3: Corrective Actions (1-4 weeks)**

Based on root cause, implement appropriate corrective actions:

1. **Model-Related:**
   - Retrain model with corrected data
   - Update model validation procedures
   - Enhance model documentation
   - Add edge case handling

2. **Workflow-Related:**
   - Update clinical protocols
   - Enhance user training
   - Improve UI/UX to prevent errors
   - Add additional safety checks

3. **System-Related:**
   - Fix software bugs
   - Improve error handling
   - Enhance logging and monitoring
   - Update alerts and warnings

**Phase 4: FDA Reporting (if applicable)**

**Reporting Timelines:**

- **Death or Serious Injury:** 30 calendar days
- **Malfunction:** 30 calendar days
- **5-Day Report:** Life-threatening events

**FDA MedWatch Reporting:**

```python
def submit_fda_medwatch_report(adverse_event):
    """Submit FDA MedWatch report for medical device adverse event."""
    
    report = {
        'form': '3500A',  # MedWatch form
        'submission_date': datetime.now().isoformat(),
        
        # Section A: Patient Information (de-identified)
        'patient': {
            'age': adverse_event['patient_age'],
            'gender': adverse_event['patient_gender'],
            'weight': adverse_event['patient_weight']
        },
        
        # Section B: Adverse Event/Product Problem
        'adverse_event': {
            'outcomes': adverse_event['outcomes'],  # death, disability, etc.
            'date_of_event': adverse_event['date'],
            'description': adverse_event['description']
        },
        
        # Section C: Suspect Medical Device
        'device': {
            'brand_name': 'AiMedRes Alzheimer Assessment',
            'model_number': adverse_event['model_version'],
            'manufacturer': 'AiMedRes',
            'operator': 'healthcare_professional'
        },
        
        # Section D: Suspect Medical Device (continued)
        'device_problem': {
            'device_available_for_evaluation': True,
            'concomitant_medical_products': adverse_event.get('other_devices', [])
        },
        
        # Section E: Reporter Information
        'reporter': {
            'name': 'AiMedRes Patient Safety',
            'contact': 'safety@aimedres.org',
            'health_professional': True,
            'occupation': 'Patient Safety Officer'
        }
    }
    
    # Submit to FDA via electronic gateway
    submit_to_fda_gateway(report)
    
    # Document submission
    log_fda_submission(adverse_event['event_id'], report)
```

---

## System Incident Management

### Service Outage Response

**Detection:**

- Automated health checks (every 30 seconds)
- Monitoring alerts (Prometheus/Grafana)
- User reports

**Response Procedure:**

```bash
#!/bin/bash
# Script: handle_service_outage.sh

INCIDENT_ID=$1

echo "=== SERVICE OUTAGE RESPONSE ==="
echo "Incident ID: $INCIDENT_ID"

# 1. Confirm outage
python3 /opt/aimedres/scripts/verify_outage.py

# 2. Alert team
python3 /opt/aimedres/scripts/alert_incident_team.py \
    --incident-id "$INCIDENT_ID" \
    --severity critical \
    --type outage

# 3. Activate incident response
python3 /opt/aimedres/scripts/create_incident_channel.py \
    --incident-id "$INCIDENT_ID"

# 4. Post status update
python3 /opt/aimedres/scripts/post_status_update.py \
    --status "investigating" \
    --message "We are currently investigating a service outage. Updates will be provided every 15 minutes."

# 5. Begin diagnostics
./diagnose_outage.sh "$INCIDENT_ID"

# 6. Attempt automatic recovery
./auto_recovery.sh "$INCIDENT_ID"

# 7. Monitor recovery
python3 /opt/aimedres/scripts/monitor_recovery.py --incident-id "$INCIDENT_ID"
```

**Downtime Communication:**

```python
def post_status_update(status, message, estimated_resolution=None):
    """Post status update to status page and notify users."""
    
    # Update status page
    update_status_page(
        component='aimedres-api',
        status=status,  # operational, degraded, partial_outage, major_outage
        message=message
    )
    
    # Send email notification to all active users
    if status in ['partial_outage', 'major_outage']:
        send_mass_notification(
            subject=f"AiMedRes Service {status.replace('_', ' ').title()}",
            message=message,
            estimated_resolution=estimated_resolution
        )
    
    # Post to Slack/Teams channels
    post_to_chat_channels(
        message=f"🚨 **{status.upper()}**: {message}",
        estimated_resolution=estimated_resolution
    )
    
    # Tweet from status account (if applicable)
    post_to_twitter(f"AiMedRes Status: {message}")
```

### Performance Degradation

**Escalation Thresholds:**

- **Latency > 2x normal:** P3 (Medium) - Investigate
- **Latency > 3x normal:** P2 (High) - Immediate action
- **Latency > 5x normal:** P1 (Critical) - Emergency response

**Response Actions:**

1. **Immediate:**
   - Enable caching
   - Scale up resources
   - Disable non-essential features
   - Rate limit requests

2. **Short-term:**
   - Optimize slow queries
   - Add database indexes
   - Improve caching strategy
   - Load balance traffic

3. **Long-term:**
   - Capacity planning
   - Architecture optimization
   - Database sharding
   - CDN implementation

---

## Communication Plans

### Internal Communications

**Incident War Room:**

- **Platform:** Dedicated Slack/Teams channel per incident
- **Naming:** `#incident-[ID]-[type]`
- **Participants:** Incident response team + stakeholders
- **Updates:** Every 15-30 minutes during active incident

**Incident Status Updates:**

```python
def send_incident_update(incident_id, status, message):
    """Send incident status update to stakeholders."""
    
    # Internal stakeholders
    internal_recipients = [
        'executive-team@hospital.org',
        'it-leadership@hospital.org',
        'clinical-leadership@hospital.org',
        'compliance-team@hospital.org'
    ]
    
    # External stakeholders (if needed)
    external_recipients = []
    if status in ['major_outage', 'data_breach']:
        external_recipients = get_affected_customers()
    
    # Send update
    send_email(
        recipients=internal_recipients + external_recipients,
        subject=f"Incident {incident_id} Update - {status}",
        body=generate_incident_update_email(incident_id, status, message)
    )
```

### External Communications

**Status Page:**

- **URL:** https://status.aimedres.hospital.org
- **Updates:** Real-time incident status
- **Subscriptions:** Email/SMS notifications available
- **History:** 90-day incident history

**User Notifications:**

- **Critical Incidents:** Immediate email + in-app notification
- **Scheduled Maintenance:** 48-hour advance notice
- **Service Degradation:** In-app banner + email

**Media Relations:**

For high-impact incidents (data breaches, patient safety):

1. **Prepare Statement:** Communications team drafts statement
2. **Executive Approval:** CEO/CISO reviews and approves
3. **Media Distribution:** Send to relevant media outlets
4. **FAQ Preparation:** Prepare responses to anticipated questions
5. **Media Monitoring:** Track media coverage and public response

### Communication Templates

**Scheduled Maintenance Notice:**

```
Subject: Scheduled Maintenance - AiMedRes - [Date]

Dear AiMedRes Users,

We will be performing scheduled maintenance on the AiMedRes platform:

Date: [Day, Date]
Time: [Start Time] - [End Time] [Timezone]
Duration: Approximately [X] hours
Impact: System will be unavailable during this time

What to Expect:
- All AiMedRes services will be offline
- No data access or model predictions during maintenance
- System will automatically resume after maintenance

What We're Doing:
[Brief description of maintenance activities]

What You Should Do:
- Plan accordingly and avoid scheduling critical assessments during this time
- Save any work before maintenance window
- Check status page for updates: https://status.aimedres.hospital.org

We apologize for any inconvenience. If you have questions, please contact support at support@aimedres.hospital.org.

Thank you for your patience.

AiMedRes Team
```

**Incident Notification:**

```
Subject: URGENT: AiMedRes Service Disruption

Dear AiMedRes Users,

We are currently experiencing a service disruption affecting the AiMedRes platform.

Status: [Investigating / Identified / Monitoring / Resolved]
Impact: [Description of impact]
Affected Services: [List of affected services]
Started At: [Time and date]

What We're Doing:
[Brief description of response actions]

Workaround (if available):
[Alternative procedures if any]

Next Update:
We will provide another update within [timeframe].

For real-time updates, visit: https://status.aimedres.hospital.org

For urgent assistance, contact: 1-800-AIMEDRES

We apologize for the disruption and are working to resolve this as quickly as possible.

AiMedRes Incident Response Team
```

---

## Post-Incident Review

### Post-Incident Review Process

**Timeline:** Within 5 business days of incident resolution

**Attendees:**

- Incident Commander
- Technical Lead
- Security Lead (if security incident)
- Clinical Lead (if adverse outcome)
- Compliance Lead
- Key responders
- Stakeholder representatives

**Agenda (90 minutes):**

1. **Incident Overview (10 min)**
   - What happened
   - Timeline of events
   - Impact assessment

2. **Response Review (20 min)**
   - What went well
   - What didn't go well
   - Response effectiveness

3. **Root Cause Analysis (20 min)**
   - Identified root causes
   - Contributing factors
   - Similar past incidents

4. **Action Items (30 min)**
   - Preventive measures
   - Process improvements
   - System enhancements
   - Training needs

5. **Follow-up (10 min)**
   - Assign action item owners
   - Set deadlines
   - Schedule follow-up review

### Post-Incident Report

**Report Template:**

```markdown
# Post-Incident Review Report

## Incident Summary

**Incident ID:** [ID]
**Date/Time:** [Date and time]
**Duration:** [Total duration]
**Severity:** [P1/P2/P3/P4]
**Type:** [Security/Breach/Adverse/System]

**Impact:**
- Users Affected: [Number]
- Patients Affected: [Number]
- Services Impacted: [List]
- Financial Impact: $[Amount]

## Timeline

| Time | Event | Action Taken |
|------|-------|--------------|
| [Time] | [Event] | [Action] |
| [Time] | [Event] | [Action] |

## Root Cause

**Primary Cause:** [Description]

**Contributing Factors:**
1. [Factor 1]
2. [Factor 2]

## Response Evaluation

**What Went Well:**
- [Item 1]
- [Item 2]

**What Didn't Go Well:**
- [Item 1]
- [Item 2]

**Lessons Learned:**
- [Lesson 1]
- [Lesson 2]

## Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action 1] | [Name] | [Date] | [ ] |
| [Action 2] | [Name] | [Date] | [ ] |

## Preventive Measures

**Immediate (0-7 days):**
- [Action 1]
- [Action 2]

**Short-term (1-4 weeks):**
- [Action 1]
- [Action 2]

**Long-term (1-6 months):**
- [Action 1]
- [Action 2]

## Sign-off

**Reviewed By:**
- Incident Commander: _________________ Date: _______
- Technical Lead: ____________________ Date: _______
- Executive Sponsor: _________________ Date: _______

**Report Distribution:**
- Executive Team
- IT Leadership
- Clinical Leadership
- Compliance Team
- All Incident Responders
```

### Action Item Tracking

```python
from datetime import datetime, timedelta

class ActionItemTracker:
    """Track post-incident action items."""
    
    def __init__(self):
        self.action_items = []
    
    def add_action_item(self, description, owner, due_date, priority='medium'):
        """Add new action item."""
        item = {
            'id': generate_unique_id(),
            'description': description,
            'owner': owner,
            'due_date': due_date,
            'priority': priority,
            'status': 'open',
            'created_date': datetime.now(),
            'completed_date': None
        }
        self.action_items.append(item)
        
        # Send notification to owner
        send_notification(
            recipient=owner,
            subject=f"Action Item Assigned: {description}",
            body=f"You have been assigned an action item from incident review.\n\n"
                 f"Description: {description}\n"
                 f"Due Date: {due_date}\n"
                 f"Priority: {priority}"
        )
    
    def check_overdue_items(self):
        """Check for overdue action items and send reminders."""
        today = datetime.now()
        
        for item in self.action_items:
            if item['status'] == 'open' and item['due_date'] < today:
                # Send overdue notification
                send_notification(
                    recipient=item['owner'],
                    subject=f"OVERDUE: Action Item - {item['description']}",
                    body=f"This action item is overdue.\n\n"
                         f"Due Date: {item['due_date']}\n"
                         f"Days Overdue: {(today - item['due_date']).days}"
                )
                
                # Escalate if > 7 days overdue
                if (today - item['due_date']).days > 7:
                    escalate_overdue_item(item)
```

### Continuous Improvement

**Incident Metrics Dashboard:**

Track and analyze incidents over time:

- Incident volume by type and severity
- Mean time to detect (MTTD)
- Mean time to respond (MTTR)
- Mean time to resolve (MTTR)
- Action item completion rate
- Recurring incident analysis

**Monthly Incident Review:**

- Review all incidents from past month
- Identify trends and patterns
- Assess response effectiveness
- Update incident response procedures
- Conduct training on lessons learned

---

## Summary

This Incident Management Guide provides:

1. **Incident Classification** with clear severity levels and response times
2. **Incident Response Team** structure with defined roles and contacts
3. **Security Incident Management** procedures with containment and eradication
4. **Data Breach Response** with HIPAA-compliant notification processes
5. **Adverse Outcome Management** with clinical investigation and FDA reporting
6. **System Incident Management** for outages and performance issues
7. **Communication Plans** for internal and external stakeholders
8. **Post-Incident Review** process with action item tracking

**Key SOPs Established:**
- ✅ Security event response (15-minute initial response)
- ✅ Data breach assessment and notification (60-day timeline)
- ✅ Adverse event reporting and investigation
- ✅ Service outage management and communication
- ✅ Post-incident review and continuous improvement

**Emergency Contacts:**
- **Incident Hotline:** 1-800-AIMEDRES (24/7)
- **Security Hotline:** 1-800-SECURITY (24/7)
- **Clinical Safety:** clinical-safety@hospital.org
- **Status Page:** https://status.aimedres.hospital.org

**Next Steps:**
1. Conduct incident response drill
2. Train team on incident procedures
3. Test communication templates
4. Establish on-call rotation
5. Review and update quarterly
