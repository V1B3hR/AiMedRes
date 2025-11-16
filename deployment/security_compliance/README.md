# Security & Compliance Guides

This directory contains comprehensive guides for implementing security and compliance measures in AiMedRes for healthcare deployment.

## Overview

These guides implement **Step 4 (Security & Compliance)** from the Healthcare Deployment Plan, covering:

- Network security and TLS enforcement
- Authentication and authorization
- Encryption and key management
- Vulnerability management

## Available Guides

### 1. Network Security Guide
**File:** `network_security_guide.md`

**Purpose:** Comprehensive network security configuration for HIPAA compliance

**Topics Covered:**
- HTTPS/TLS 1.2+ enforcement for all traffic
- SSL/TLS certificate setup (Let's Encrypt, commercial, internal CA)
- Nginx reverse proxy configuration with security headers
- Firewall configuration (UFW, iptables)
- Network segmentation and isolation (DMZ, Application, Database zones)
- DDoS protection (rate limiting, fail2ban)
- Network monitoring and alerting
- Security verification scripts

**Key Features:**
- ✅ TLS 1.2+ only (TLS 1.0/1.1 disabled)
- ✅ Strong cipher suites (ECDHE, AES-256-GCM)
- ✅ HSTS with 1-year max-age
- ✅ Security headers (CSP, X-Frame-Options, etc.)
- ✅ Default deny firewall policy
- ✅ Network segmentation with VLANs

**Quick Verification:**
```python
from deployment.security_compliance.network_security_guide import verify_tls_configuration
verify_tls_configuration('aimedres.hospital.org')
```

### 2. Authentication & Authorization Guide
**File:** `authentication_authorization_guide.md`

**Purpose:** Complete authentication and access control implementation

**Topics Covered:**
- Role-Based Access Control (RBAC) system
  - admin, clinician, researcher, auditor, api_user roles
  - Granular permissions system
- Local authentication with secure password policies
- LDAP/Active Directory integration
- OpenID Connect (OIDC) and SAML SSO
- API key and JWT token authentication
- Comprehensive audit logging
- Multi-factor authentication (MFA) support

**Predefined Roles:**
| Role | Permissions | Use Case |
|------|-------------|----------|
| admin | system:*, user:*, data:*, model:*, config:* | System administrators |
| clinician | patient:read/write, assessment:run/read | Physicians, nurses |
| researcher | data:read_anonymized, model:train/evaluate | Research staff |
| auditor | audit:read, logs:read, reports:read | Compliance officers |
| api_user | api:read/write | External systems |

**Security Features:**
- ✅ PBKDF2-HMAC-SHA256 password hashing
- ✅ 12+ character password policy
- ✅ Account lockout (5 attempts, 15-minute lockout)
- ✅ Cryptographically secure API keys
- ✅ JWT with configurable expiry
- ✅ All access audited and logged

**Example Usage:**
```python
from src.aimedres.security.auth import SecureAuthManager

auth_manager = SecureAuthManager({
    'jwt_secret': os.getenv('JWT_SECRET_KEY'),
    'token_expiry_hours': 24
})

# Protected endpoint
@require_api_key
@require_permission('patient:read')
def get_patient_data():
    pass
```

### 3. Encryption & Key Management Guide
**File:** `encryption_key_management_guide.md`

**Purpose:** Quantum-safe encryption and comprehensive key management

**Topics Covered:**
- Quantum-safe cryptography (CRYSTALS-Kyber)
  - Hybrid encryption (Kyber768 + AES-256)
  - Post-quantum algorithms
- Production key management system
- Cloud KMS integration
  - AWS KMS
  - Azure Key Vault
  - GCP KMS
  - HashiCorp Vault
- Automated key rotation (90-day schedule)
- Data encryption (at rest and in transit)
  - Database encryption
  - File encryption
  - Application-level encryption
- Key backup and recovery

**Quantum-Safe Algorithms:**
| Algorithm | Security Level | Key Size | Recommendation |
|-----------|---------------|----------|----------------|
| kyber512 | NIST Level 1 | 800 bytes | Development/Testing |
| kyber768 | NIST Level 3 | 1184 bytes | **Production** ✅ |
| kyber1024 | NIST Level 5 | 1568 bytes | High-security |

**Key Features:**
- ✅ Post-quantum cryptography (future-proof)
- ✅ Hybrid classical + quantum-safe encryption
- ✅ Automated 90-day key rotation
- ✅ Cloud KMS integration for enterprise deployments
- ✅ Secure key storage with encrypted at-rest keys
- ✅ Database and file encryption
- ✅ TLS 1.2+ for data in transit

**Example Usage:**
```python
from security.quantum_crypto import QuantumSafeCryptography

quantum_crypto = QuantumSafeCryptography({
    'quantum_safe_enabled': True,
    'quantum_algorithm': 'kyber768',
    'hybrid_mode': True,
    'key_rotation_days': 90
})

# Encrypt with quantum-safe algorithm
encrypted = hybrid_engine.encrypt(sensitive_data)
```

### 4. Vulnerability Management Guide
**File:** `vulnerability_management_guide.md`

**Purpose:** Comprehensive vulnerability scanning and remediation process

**Topics Covered:**
- Container security scanning (Trivy)
- Dependency vulnerability scanning (Safety, npm audit)
- Code security scanning (Bandit, Semgrep, CodeQL)
- Penetration testing with OWASP ZAP
- Vulnerability assessment with CVSS scoring
- Remediation tracking with SLA management
- Continuous monitoring and automated scans
- Security reporting

**Scanning Tools:**

1. **Container Security:**
   - Trivy for Docker images
   - Exit-on-vulnerability in CI/CD
   - Daily automated scans

2. **Dependency Scanning:**
   - Safety for Python packages
   - npm audit for JavaScript
   - CVE database integration

3. **Code Security:**
   - Bandit for Python
   - Semgrep for multi-language
   - CodeQL for SAST

4. **Penetration Testing:**
   - OWASP ZAP baseline and full scans
   - API security testing
   - SQL injection, XSS, auth testing

**Remediation SLAs:**
- Priority 9-10 (Critical): 1-3 days
- Priority 7-8 (High): 7-14 days
- Priority 4-6 (Medium): 30 days
- Priority 1-3 (Low): 60-90 days

**Automated Scanning:**
```bash
# Daily scans
./scan_docker_images.sh
safety check --exit-code 1
bandit -r src/ --severity-level medium

# Weekly pen test
./pen_test_baseline.sh
```

## Implementation Workflow

### Step 1: Network Security
1. Review `network_security_guide.md`
2. Obtain SSL/TLS certificates (Let's Encrypt recommended)
3. Configure Nginx with TLS 1.2+ and security headers
4. Set up firewall (UFW or iptables)
5. Implement network segmentation
6. Configure DDoS protection
7. Test TLS configuration
8. Enable monitoring

### Step 2: Authentication & Authorization
1. Review `authentication_authorization_guide.md`
2. Initialize SecureAuthManager
3. Define roles and permissions for your organization
4. Create initial users and API keys
5. Configure LDAP/OIDC integration (if needed)
6. Enable audit logging
7. Test authentication flows
8. Configure password policy

### Step 3: Encryption & Key Management
1. Review `encryption_key_management_guide.md`
2. Initialize QuantumSafeCryptography
3. Configure key storage (local or cloud KMS)
4. Set up automated key rotation
5. Enable database encryption
6. Configure file encryption
7. Test encryption/decryption
8. Verify TLS for data in transit

### Step 4: Vulnerability Management
1. Review `vulnerability_management_guide.md`
2. Install scanning tools (Trivy, Safety, Bandit)
3. Configure automated scanning schedule
4. Set up OWASP ZAP for pen testing
5. Implement vulnerability assessment process
6. Configure remediation tracking
7. Run initial security scans
8. Generate security report

## Configuration Files

Create environment variables for your deployment:

```bash
# Network Security
HTTPS_ENABLED=true
TLS_MIN_VERSION=1.2
SSL_CERT_PATH=/etc/ssl/certs/aimedres.crt
SSL_KEY_PATH=/etc/ssl/private/aimedres.key

# Authentication
JWT_SECRET_KEY=your_jwt_secret_key
TOKEN_EXPIRY_HOURS=24
MAX_FAILED_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15

# LDAP (if using)
LDAP_ENABLED=true
LDAP_SERVER=ldap://ldap.hospital.org:389
LDAP_BASE_DN=dc=hospital,dc=org

# Encryption
ENCRYPTION_ENABLED=true
QUANTUM_SAFE_ENABLED=true
QUANTUM_ALGORITHM=kyber768
KEY_ROTATION_DAYS=90
KMS_PROVIDER=aws  # or azure, gcp, vault, local

# Vulnerability Scanning
VULN_SCAN_ENABLED=true
VULN_SCAN_SCHEDULE=daily
CONTAINER_SCAN_ON_BUILD=true
```

## Security Checklist

Use this checklist to verify implementation:

### Network Security
- [ ] TLS 1.2+ enforced on all endpoints
- [ ] SSL certificates valid and not expiring soon
- [ ] HTTP redirects to HTTPS
- [ ] Strong cipher suites configured
- [ ] HSTS header enabled
- [ ] Security headers configured
- [ ] Firewall enabled with default deny
- [ ] Network segmentation implemented
- [ ] DDoS protection active
- [ ] Network monitoring configured

### Authentication & Authorization
- [ ] Strong password policy enforced
- [ ] API keys cryptographically secure
- [ ] JWT tokens with appropriate expiry
- [ ] RBAC implemented with defined roles
- [ ] LDAP/SSO integrated (if applicable)
- [ ] Audit logging for all access
- [ ] Failed login attempts tracked
- [ ] Account lockout configured
- [ ] MFA available (optional)

### Encryption & Key Management
- [ ] Quantum-safe cryptography enabled
- [ ] Hybrid encryption mode active
- [ ] Data encrypted at rest
- [ ] Data encrypted in transit
- [ ] Keys stored in HSM/KMS
- [ ] Automated key rotation configured
- [ ] Key backup procedures documented
- [ ] Encryption performance tested

### Vulnerability Management
- [ ] Container scanning automated
- [ ] Dependency scanning scheduled
- [ ] Code security scanning in CI/CD
- [ ] Penetration testing scheduled
- [ ] Vulnerability assessment process defined
- [ ] Remediation SLAs documented
- [ ] Security reports generated regularly
- [ ] Critical vulnerabilities patched promptly

## Testing and Validation

Each guide includes comprehensive testing procedures:

```python
# Network security
from deployment.security_compliance.network_security_guide import verify_network_compliance
verify_network_compliance()

# Authentication
from deployment.security_compliance.authentication_authorization_guide import verify_auth_compliance
verify_auth_compliance()

# Encryption
from deployment.security_compliance.encryption_key_management_guide import verify_encryption_compliance
verify_encryption_compliance()

# Run all compliance checks
python deployment/security_compliance/validate_compliance.py
```

## Compliance Standards

These implementations support compliance with:

- ✅ HIPAA Security Rule
  - Access Control (§164.312(a))
  - Audit Controls (§164.312(b))
  - Integrity (§164.312(c))
  - Transmission Security (§164.312(e))
- ✅ NIST Cybersecurity Framework
  - Identify, Protect, Detect, Respond, Recover
- ✅ NIST SP 800-52 (TLS Guidelines)
- ✅ NIST SP 800-53 (Security Controls)
- ✅ NIST Post-Quantum Cryptography
- ✅ OWASP Top 10
- ✅ PCI DSS (if applicable)
- ✅ GDPR (if applicable)

## Continuous Monitoring

Set up automated monitoring and alerting:

```python
import schedule

# Daily security scans
schedule.every().day.at("02:00").do(run_all_scans)

# Weekly penetration test
schedule.every().sunday.at("03:00").do(run_pen_test)

# Monthly security report
schedule.every().month.do(generate_security_report)
```

## Support and Resources

- **Main Documentation:** `healthcaredeploymentplan.md`
- **Code Implementation:**
  - `src/aimedres/security/auth.py`
  - `src/aimedres/security/encryption.py`
  - `security/quantum_crypto.py`
  - `security/quantum_prod_keys.py`
- **Tests:** `tests/security/`
- **Issue Tracker:** [GitHub Issues](https://github.com/V1B3hR/AiMedRes/issues)

## Quick Links

- [Network Security Guide](network_security_guide.md)
- [Authentication & Authorization Guide](authentication_authorization_guide.md)
- [Encryption & Key Management Guide](encryption_key_management_guide.md)
- [Vulnerability Management Guide](vulnerability_management_guide.md)
- [Main Deployment Plan](../README.md)
- [Data & Integration Guides](../data_integration/)

## Version History

- **v1.0.0** (2024-01) - Initial implementation of all Security & Compliance guides
  - Network security with TLS 1.2+ enforcement
  - RBAC with LDAP/SSO integration
  - Quantum-safe encryption (Kyber768)
  - Comprehensive vulnerability management
