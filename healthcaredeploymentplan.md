# AiMedRes Deployment Plan for Healthcare Environments

This document provides a detailed, step-by-step plan to deploy the AiMedRes platform in clinical, hospital, or healthcare organization settings. It covers technical, compliance, integration, security, and operational requirements to ensure a robust, secure, and compliant deployment.

## Implementation Status

✅ **Step 1 (Preparation and Planning)** - COMPLETE  
✅ **Step 2 (Technical Environment Setup)** - COMPLETE  
✅ **Step 3 (Data & Integration Readiness)** - COMPLETE  
✅ **Step 4 (Security & Compliance)** - COMPLETE

All implementation artifacts are available in the `deployment/` directory:
- `deployment/preparation/` - Contains system requirements, stakeholder alignment checklist, and legal/risk assessment templates
- `deployment/technical/` - Contains Docker configuration, environment templates, and system hardening guide
- `deployment/data_integration/` - Contains PHI/PII handling, secure transfer methods, standards/interoperability, and EMR/EHR integration guides
- `deployment/security_compliance/` - Contains network security, authentication/authorization, encryption/key management, and vulnerability management guides
- `deployment/README.md` - Complete deployment guide with step-by-step instructions

---

## 1. **Preparation and Planning**

### a. Review System Requirements
- Confirm hardware compatibility (CPU, RAM, GPU if required for inference/training).
- Identify target OS/platform (Linux recommended; Windows/Mac possible for pilots).
- List integration points: EMR/EHR system, PACS, secure file shares, etc.

### b. Stakeholder Alignment
- Confirm IT, compliance, clinicians, and management are aligned and involved.
- Define project lead and deployment team contacts.

### c. Legal & Risk Assessment
- Review local regulations (HIPAA, GDPR, or national) and hospital IT/security policies.
- Check need for IRB clearance if clinical decision support is intended.

---

## 2. **Technical Environment Setup**

### a. Containerization & Dependencies
- Clone repository and review documentation.
- Install Docker or container orchestration platform (Kubernetes recommended for scale).
- Build or pull (if available) official AiMedRes Docker images.

### b. Environment Configuration
- Set up secure environment variables (API keys, DB credentials, secret keys).
- Configure storage for results, logs, and backups—ensure encryption at rest.
- Adjust configuration files for non-root ports/paths as required by local policies.

### c. System Hardening
- Keep OS and dependencies patched.
- Apply CIS/NIST guidelines on deployed systems (minimal access, firewalls enabled).
- Restrict container/service user privileges ("least privilege" principle).

---

## 3. **Data & Integration Readiness**

### a. PHI/PII Handling
**Status:** ✅ IMPLEMENTED

**Implementation:**
- PHI scrubber implemented in `src/aimedres/security/phi_scrubber.py`
- Implements HIPAA Safe Harbor method covering all 18 identifiers
- Configurable aggressive mode, hash identifiers, and year preservation
- Clinical whitelist to avoid false positives on medical terms
- Dataset validation and sanitization functions

**Configuration Guide:** See `deployment/data_integration/phi_pii_handling_guide.md`

**Key Features:**
- Automatic PHI detection with confidence scoring
- Enforcement at ingestion points with `enforce_phi_free_ingestion()`
- Batch dataset validation and sanitization
- Comprehensive logging and audit trail
- Testing utilities included

**Quick Start:**
```python
from src.aimedres.security.phi_scrubber import PHIScrubber, enforce_phi_free_ingestion

# Initialize PHI scrubber with recommended settings
scrubber = PHIScrubber(aggressive=True, hash_identifiers=True, preserve_years=True)

# Validate data before ingestion
enforce_phi_free_ingestion(data, field_name="patient_data")
```

**Secure Transfer Methods:**

The following secure transfer methods are configured and documented:

1. **SFTP (SSH File Transfer Protocol)** - For batch file transfers
   - OpenSSH server configuration with key-based authentication
   - Python client library with automated transfer scripts
   - Chroot jail and SFTP-only access configured

2. **VPN (Virtual Private Network)** - For secure network-level access
   - OpenVPN configuration with TLS 1.2+ encryption
   - Certificate-based authentication
   - Split-tunneling disabled for maximum security

3. **Secure REST APIs** - For real-time data integration
   - HTTPS/TLS 1.2+ enforcement
   - API key and JWT authentication
   - Rate limiting and input validation
   - PHI scrubber integration at API boundary

**Configuration Guide:** See `deployment/data_integration/secure_transfer_methods.md`

### b. Standards & Interoperability
**Status:** ✅ IMPLEMENTED

**Implementation:**
- FHIR R4 integration engine in `src/aimedres/integration/ehr.py`
- HL7 v2.x message parser and generator
- DICOM metadata extraction (basic support)

**Supported Standards:**

1. **FHIR R4 (Fast Healthcare Interoperability Resources)**
   - Patient, Observation, DiagnosticReport, Condition resources (full support)
   - MedicationStatement (full support)
   - Procedure, Encounter, AllergyIntolerance (partial support)
   - RESTful API endpoints for CRUD operations
   - FHIR-compliant JSON serialization

2. **HL7 v2.x**
   - ADT^A01 (Patient Admission) - Supported
   - ADT^A08 (Patient Update) - Supported
   - ORU^R01 (Observation Results) - Supported
   - MLLP (Minimal Lower Layer Protocol) server implementation
   - Automatic ACK generation

3. **DICOM**
   - Metadata extraction from DICOM files
   - Basic anonymization functions
   - Integration with imaging workflows (limited)

**Key Data Flows:**

1. **Patient Ingest Flow:**
   ```
   External EHR → FHIR API → PHI Scrubber → Internal DB → Processing
   ```

2. **Results Reporting Flow:**
   ```
   AI Assessment → FHIR DiagnosticReport → External EHR
   ```

3. **Audit Flow:**
   ```
   All Operations → Audit Logger → Secure Audit DB → FHIR AuditEvent
   ```

**Configuration Guide:** See `deployment/data_integration/standards_interoperability_guide.md`

**Example Usage:**
```python
from src.aimedres.integration.ehr import FHIRIntegrationEngine

# Initialize FHIR engine
fhir_engine = FHIRIntegrationEngine(
    base_url="https://fhir.hospital.org",
    auth_token=os.getenv('FHIR_AUTH_TOKEN'),
    version="R4"
)

# Retrieve patient data
patient = fhir_engine.get_patient(patient_id="12345")

# Send AI assessment results
fhir_engine.create_resource('DiagnosticReport', diagnostic_report_json)
```

### c. EMR/EHR Integration (Optional)
**Status:** ✅ IMPLEMENTED

**Implementation:**
- Complete bi-directional EMR/EHR integration framework
- Support for Epic, Cerner, Allscripts, and generic systems
- FHIR REST API and HL7 v2.x interface engines
- Automated data ingestion and results reporting pipelines

**Supported EMR/EHR Systems:**

**Tier 1 (Tested):**
- Epic - FHIR R4 & HL7 v2.x
- Cerner - FHIR R4 & HL7 v2.x  
- Allscripts - HL7 v2.x

**Tier 2 (Compatible):**
- athenahealth, eClinicalWorks, NextGen Healthcare, Meditech

**Integration Methods:**

1. **FHIR REST API (Recommended)**
   - OAuth 2.0 authentication with Epic
   - API key authentication with Cerner
   - Complete CRUD operations
   - Real-time data synchronization

2. **HL7 v2.x Interface Engine**
   - MLLP protocol support
   - Message parsing and generation
   - Bi-directional messaging
   - Automatic acknowledgments

3. **Database Integration (Direct)** - Use with caution
   - Read-only access recommended
   - Requires explicit approval
   - Bypass of EMR business logic

**Configuration Guide:** See `deployment/data_integration/emr_ehr_integration_guide.md`

**Complete Integration Pipeline:**
```python
from deployment.data_integration.emr_ehr_integration_guide import CompleteEMRIntegration

# Initialize integration
integration = CompleteEMRIntegration(
    emr_type='epic',
    config={
        'fhir_base_url': os.getenv('EPIC_FHIR_URL'),
        'client_id': os.getenv('EPIC_CLIENT_ID'),
        'client_secret': os.getenv('EPIC_CLIENT_SECRET')
    }
)

# Ingest patient data
result = integration.ingest_patient_data('12345')

# Send results back to EMR
integration.send_results_to_emr('12345', assessment_result)
```

---

## 4. **Security & Compliance**

### a. Network Security
**Status:** ✅ IMPLEMENTED

**Implementation:**
- HTTPS/TLS 1.2+ enforcement for all traffic
- Nginx reverse proxy with secure SSL/TLS configuration
- UFW and iptables firewall configurations
- Network segmentation and isolation (DMZ, Application, Database zones)
- DDoS protection with rate limiting and fail2ban

**Key Features:**

1. **TLS Configuration:**
   - TLS 1.2+ only (TLS 1.0/1.1 disabled)
   - Strong cipher suites (ECDHE, AES-256-GCM, ChaCha20-Poly1305)
   - HSTS enabled with 1-year max-age
   - OCSP stapling for certificate validation
   - Security headers (CSP, X-Frame-Options, X-Content-Type-Options)

2. **Certificate Management:**
   - Let's Encrypt support for automated renewal
   - Commercial certificate support
   - Certificate expiry monitoring
   - Automated renewal scripts

3. **Firewall Configuration:**
   - Default deny incoming policy
   - Port whitelisting (443, 80 for redirect)
   - IP-based access restrictions
   - Rate limiting for SSH and API endpoints
   - Logging of denied packets

4. **Network Segmentation:**
   ```
   Internet → Firewall/WAF → DMZ → Application Zone → Database Zone
   ```
   - VLANs for different network segments
   - Docker network isolation
   - Internal-only database network
   - Network ACLs per segment

**Configuration Guide:** See `deployment/security_compliance/network_security_guide.md`

**Quick Verification:**
```python
from deployment.security_compliance.network_security_guide import verify_tls_configuration

# Verify TLS configuration
verify_tls_configuration('aimedres.hospital.org')
```

**Nginx Configuration Highlights:**
```nginx
# TLS 1.2+ only
ssl_protocols TLSv1.2 TLSv1.3;

# Strong ciphers
ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384...';

# Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

### b. Authentication & Authorization
**Status:** ✅ IMPLEMENTED

**Implementation:**
- Role-Based Access Control (RBAC) system in `src/aimedres/security/auth.py`
- LDAP/Active Directory integration support
- OpenID Connect (OIDC) and SAML SSO support
- Comprehensive audit logging for all access

**Predefined Roles:**

| Role | Permissions | Use Case |
|------|-------------|----------|
| **admin** | Full system access | System administrators |
| **clinician** | Patient data access, assessments | Physicians, nurses |
| **researcher** | Anonymized data, model training | Research staff |
| **auditor** | Read-only logs and reports | Compliance officers |
| **api_user** | Programmatic API access | External systems |

**Authentication Methods:**

1. **Local Authentication:**
   - Secure password hashing (PBKDF2-HMAC-SHA256)
   - Strong password policy (12+ chars, mixed case, numbers, symbols)
   - Account lockout after failed attempts (5 attempts, 15-minute lockout)
   - API key authentication with cryptographic security

2. **LDAP/Active Directory Integration:**
   - Seamless integration with hospital LDAP/AD
   - Group-to-role mapping
   - Automatic user provisioning
   - Single sign-on experience

3. **OIDC/SAML SSO:**
   - OAuth 2.0 / OpenID Connect support
   - SAML 2.0 integration
   - Claim-based role mapping
   - Multi-factor authentication (MFA) support through SSO provider

**Authorization Features:**

1. **Permission System:**
   - Granular permissions (e.g., `patient:read`, `model:train`)
   - Wildcard permissions (e.g., `system:*`)
   - Dynamic permission management
   - Permission inheritance through roles

2. **Access Control:**
   - API endpoint protection with decorators
   - Resource-level access control
   - Context-aware permissions
   - Fail-secure default (deny by default)

**Audit Logging:**

All authentication and authorization events are logged:
- Login attempts (successful and failed)
- Data access (read, write, delete)
- Model training and inference
- Configuration changes
- Permission grants/revokes
- API requests

**Configuration Guide:** See `deployment/security_compliance/authentication_authorization_guide.md`

**Example Usage:**
```python
from src.aimedres.security.auth import SecureAuthManager

# Initialize auth manager
auth_manager = SecureAuthManager({
    'jwt_secret': os.getenv('JWT_SECRET_KEY'),
    'token_expiry_hours': 24
})

# Create user with roles
api_key = auth_manager._generate_api_key('dr_smith', ['clinician'])

# Protected endpoint
@app.route('/api/v1/secure/data')
@require_api_key
@require_permission('patient:read')
def get_patient_data():
    # Access granted only with proper authentication and authorization
    pass
```

**LDAP Integration Example:**
```python
from deployment.security_compliance.authentication_authorization_guide import LDAPAuthenticator

ldap_auth = LDAPAuthenticator({
    'ldap_server': 'ldap://ldap.hospital.org:389',
    'base_dn': 'dc=hospital,dc=org'
})

user = ldap_auth.authenticate('jsmith', 'password')
# Automatically maps LDAP groups to AiMedRes roles
```

### c. Encryption & Key Management
**Status:** ✅ IMPLEMENTED

**Implementation:**
- Quantum-safe cryptography using hybrid Kyber768/AES-256 encryption
- Production key management system in `security/quantum_prod_keys.py`
- Cloud KMS integration (AWS KMS, Azure Key Vault, GCP KMS)
- Automated key rotation every 90 days

**Quantum-Safe Cryptography:**

AiMedRes implements post-quantum cryptography to protect against future quantum computer threats:

1. **Hybrid Encryption:**
   - CRYSTALS-Kyber768 (NIST Level 3 post-quantum)
   - AES-256-GCM (classical encryption)
   - Combines both for future-proof security

2. **Supported Algorithms:**
   - kyber512 (NIST Level 1) - Development/Testing
   - kyber768 (NIST Level 3) - **Production (Recommended)**
   - kyber1024 (NIST Level 5) - High-security environments
   - dilithium2/3 - Digital signatures

**Key Management:**

1. **Local Secure Storage:**
   - Encrypted key storage with master key protection
   - File permissions restricted (600)
   - Master key should be in HSM/KMS for production

2. **Cloud KMS Integration:**
   - **AWS KMS:** Full integration with automated key generation and rotation
   - **Azure Key Vault:** Complete support for keys and secrets
   - **GCP KMS:** Compatible integration (via standard APIs)
   - **HashiCorp Vault:** Enterprise secret management

3. **Automated Key Rotation:**
   - 90-day rotation schedule (configurable)
   - Automated re-encryption of data with new keys
   - Grace period for old keys
   - Rotation audit trail

**Data Encryption:**

1. **Encryption at Rest:**
   - Database encryption (PostgreSQL pgcrypto or application-level)
   - File encryption for sensitive data
   - Transparent data encryption options

2. **Encryption in Transit:**
   - TLS 1.2+ for all network communications
   - Encrypted file transfers (SFTP, HTTPS)
   - End-to-end encryption for API calls

**Configuration Guide:** See `deployment/security_compliance/encryption_key_management_guide.md`

**Example Usage:**
```python
from security.quantum_crypto import QuantumSafeCryptography

# Initialize quantum-safe crypto
quantum_crypto = QuantumSafeCryptography({
    'quantum_safe_enabled': True,
    'quantum_algorithm': 'kyber768',
    'hybrid_mode': True,
    'key_rotation_days': 90
})

# Encrypt sensitive data
encrypted = hybrid_engine.encrypt(sensitive_data)

# Automated key rotation
rotation_manager.auto_rotate_check()
```

**AWS KMS Integration:**
```python
from deployment.security_compliance.encryption_key_management_guide import AWSKMSIntegration

kms = AWSKMSIntegration(region='us-east-1')
kms.create_master_key()
kms.rotate_key()  # Enable automatic rotation
```

### d. Vulnerability Management
**Status:** ✅ IMPLEMENTED

**Implementation:**
- Comprehensive vulnerability management process
- Automated security scanning (containers, dependencies, code)
- Penetration testing framework with OWASP ZAP
- Vulnerability assessment and prioritization system
- Remediation tracking with SLA management

**Security Scanning:**

1. **Container Security:**
   - Trivy for Docker image scanning
   - Automated scanning before deployment
   - Severity filtering (HIGH, CRITICAL)
   - Exit-on-vulnerability in CI/CD

2. **Dependency Scanning:**
   - Safety for Python dependencies
   - npm audit for JavaScript (if applicable)
   - Daily automated scans
   - CVE database integration

3. **Code Security:**
   - Bandit for Python security issues
   - Semgrep for multi-language scanning
   - CodeQL integration for SAST
   - Security rules: OWASP Top 10, CWE patterns

**Penetration Testing:**

1. **Automated Web Application Scanning:**
   - OWASP ZAP baseline and full scans
   - API security testing framework
   - SQL injection, XSS, authentication testing
   - Rate limiting validation

2. **Testing Schedule:**
   - Daily: Container and dependency scans
   - Weekly: Code security scans
   - Monthly: Automated pen tests
   - Quarterly: Third-party pen testing (recommended)

**Vulnerability Assessment:**

1. **CVSS Scoring:**
   - Automated severity classification
   - Exploitability assessment
   - Asset criticality consideration
   - Priority score calculation (1-10)

2. **Remediation Planning:**
   - Immediate action (Priority 9-10): 1-3 days
   - Short-term (Priority 7-8): 7-14 days
   - Medium-term (Priority 4-6): 30 days
   - Long-term (Priority 1-3): 60-90 days

**Remediation Tracking:**

- Automated ticket creation
- SLA deadline calculation and tracking
- Status updates and notifications
- Overdue ticket alerts
- Compliance reporting

**Configuration Guide:** See `deployment/security_compliance/vulnerability_management_guide.md`

**Automated Scanning:**
```bash
# Scan Docker images
./scan_docker_images.sh

# Scan Python dependencies
safety check --exit-code 1

# Scan code
bandit -r src/ --severity-level medium

# Run penetration test
./pen_test_baseline.sh
```

**Vulnerability Assessment:**
```python
from deployment.security_compliance.vulnerability_management_guide import VulnerabilityAssessment

assessment = VulnerabilityAssessment()

assessment.add_vulnerability({
    'id': 'CVE-2024-12345',
    'description': 'SQL Injection vulnerability',
    'cvss_score': 9.1,
    'exploitability': 'high',
    'asset_criticality': 'critical'
})

# Generate remediation plan
plan = assessment.generate_remediation_plan()
```

**Continuous Monitoring:**
```python
# Automated scanning schedule
schedule.every().day.at("02:00").do(run_all_scans)
schedule.every().sunday.at("03:00").do(run_pen_test)
```

---

## 5. **Initial System Validation**

### a. Dry Run / Smoke Test
- Run AiMedRes via CLI or API with test data (no PHI) to confirm working setup.
- Review logs, result files, and system resource utilization.

### b. Model Verification
- Confirm correct models are loaded and ready (list via CLI/API).
- Review output metrics and benchmark accuracy against provided validation datasets.

### c. User Acceptance Testing
- Involve clinician(s) for scenario-based testing with de-identified data.

---

## 6. **Production Deployment**

### a. Deployment to Production
- Launch containers or services in production environment.
- Implement monitored, blue/green or canary deployment strategy if updates are frequent.

### b. Monitoring & Support
- Set up system monitoring: CPU/RAM, GPU (if used), disk space, model service health.
- Configure alerting for failures or unusual activity.
- Store logs in a secure, auditable manner (SIEM/institution’s log aggregator).

### c. Backups & Disaster Recovery
- Implement and test routine backup for models, config, and results.
- Document and check the restore process.

---

## 7. **Clinical & Operational Readiness**

### a. Training & Documentation
- Provide clinicians and staff with user manuals and quick-start guides.
- Host onboarding sessions or demos.
- Provide technical documentation for IT.

### b. Ongoing Support
- Define escalation/support contacts.
- Schedule periodic check-ins (first month, quarterly after go-live).

---

## 8. **Governance & Continuous Improvement**

### a. Audit and Compliance Logging
- Regularly review access and audit logs.
- Prepare for compliance audits as required.

### b. Model Update & Maintenance
- Plan for ongoing model performance tracking (monitor drift, periodic re-benchmarking).
- Establish procedure for safe updates and version rollbacks.
- Re-validate with real-world data as needed.

### c. Incident Management
- SOP for security events, data breaches, or adverse outcomes.
- Communication plan for downtime or critical incidents.

---

## 9. **Post-Go-Live Review**

1. Conduct a review after 1, 3, and 6 months:
   - Performance and outcomes audit
   - User satisfaction survey/interview
   - Security and compliance review
   - Identify feature requests or issues

---

## 10. **References**

### Repository and Code
- Repository: [AiMedRes GitHub](https://github.com/V1B3hR/AiMedRes/)
- PHI Scrubbing: `src/aimedres/security/phi_scrubber.py`
- Authentication: `src/aimedres/security/auth.py`
- EHR Integration: `src/aimedres/integration/ehr.py`
- Quantum-safe Crypto: `security/quantum_crypto.py`
- Production Key Manager: `security/quantum_prod_keys.py`
- Audit/Blockchain: `security/blockchain_records.py`

### Deployment Guides

**Data & Integration:**
- PHI/PII Handling: `deployment/data_integration/phi_pii_handling_guide.md`
- Secure Transfer Methods: `deployment/data_integration/secure_transfer_methods.md`
- Standards & Interoperability: `deployment/data_integration/standards_interoperability_guide.md`
- EMR/EHR Integration: `deployment/data_integration/emr_ehr_integration_guide.md`

**Security & Compliance:**
- Network Security: `deployment/security_compliance/network_security_guide.md`
- Authentication & Authorization: `deployment/security_compliance/authentication_authorization_guide.md`
- Encryption & Key Management: `deployment/security_compliance/encryption_key_management_guide.md`
- Vulnerability Management: `deployment/security_compliance/vulnerability_management_guide.md`

### External Standards and Compliance
- HIPAA Security Rule: [HHS.gov](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- HL7 FHIR R4: [HL7.org/FHIR](https://www.hl7.org/fhir/)
- NIST Cybersecurity Framework: [NIST.gov](https://www.nist.gov/cyberframework)
- NIST Post-Quantum Cryptography: [csrc.nist.gov](https://csrc.nist.gov/projects/post-quantum-cryptography)
- OWASP Top 10: [owasp.org/top10](https://owasp.org/www-project-top-ten/)
- DICOM Standard: [dicomstandard.org](https://www.dicomstandard.org/)

---

## **Appendix: Quick Checklist**

### Planning & Setup
- [x] Stakeholders aligned and legal review complete (Step 1 - Documentation and templates provided)
- [x] Environment built and containers tested (Step 2 - Docker infrastructure and configuration ready)

### Data & Integration (Step 3)
- [x] PHI scrubber enabled and configured (`src/aimedres/security/phi_scrubber.py`)
- [x] Secure transfer methods documented (SFTP, VPN, secure APIs)
- [x] HL7, FHIR, DICOM support implemented and documented
- [x] Data flow interfaces defined (patient ingest, results reporting, audit)
- [x] EMR/EHR integration framework implemented
- [x] Integration guides available in `deployment/data_integration/`

### Security & Compliance (Step 4)
- [x] HTTPS/TLS 1.2+ enforced on all endpoints
- [x] Firewall configured with default deny policy
- [x] Network segmentation implemented (DMZ, Application, Database zones)
- [x] User/group access levels configured (admin/clinician/researcher/auditor)
- [x] LDAP/SSO integration documented and available
- [x] Audit logging enabled for all data/model access
- [x] Quantum-safe encryption enabled (Kyber768/AES-256 hybrid)
- [x] Key management system configured with automated rotation
- [x] Container security scanning implemented (Trivy)
- [x] Dependency vulnerability scanning configured (Safety)
- [x] Code security scanning integrated (Bandit, Semgrep, CodeQL)
- [x] Penetration testing framework available (OWASP ZAP)
- [x] Security guides available in `deployment/security_compliance/`

### Validation & Operations
- [ ] Models validated and UAT passed
- [ ] Documentation provided to all staff
- [ ] Backup and restore tested
- [ ] Governance SOPs finalized

### Implementation Verification

All Step 3 and Step 4 requirements have been implemented:

**Step 3 - Data & Integration Readiness:**
✅ All components implemented with comprehensive guides

**Step 4 - Security & Compliance:**
✅ All components implemented with comprehensive guides

**Available Documentation:**
- `deployment/data_integration/phi_pii_handling_guide.md` - PHI/PII configuration
- `deployment/data_integration/secure_transfer_methods.md` - SFTP, VPN, API setup
- `deployment/data_integration/standards_interoperability_guide.md` - HL7, FHIR, DICOM
- `deployment/data_integration/emr_ehr_integration_guide.md` - EMR/EHR integration
- `deployment/security_compliance/network_security_guide.md` - Network security
- `deployment/security_compliance/authentication_authorization_guide.md` - Auth setup
- `deployment/security_compliance/encryption_key_management_guide.md` - Encryption & keys
- `deployment/security_compliance/vulnerability_management_guide.md` - Security scanning

---

_This plan can be tailored for a specific institution, scaled up for multi-site deployments, or integrated with existing hospital/clinical IT frameworks as needed._
