# API Security, PHI Compliance, and Audit Logging Documentation

This document provides comprehensive details on how all AiMedRes API endpoints are secured with authentication, PHI compliance, and audit logging.

## Table of Contents
1. [Authentication & Authorization](#authentication--authorization)
2. [PHI Compliance](#phi-compliance)
3. [Audit Logging](#audit-logging)
4. [Endpoint Security Matrix](#endpoint-security-matrix)
5. [Security Testing](#security-testing)

---

## Authentication & Authorization

### Overview
All AiMedRes API endpoints implement enterprise-grade authentication using:
- **API Keys**: Cryptographically secure keys with PBKDF2 hashing
- **JWT Tokens**: Session management with expiration
- **Role-Based Access Control (RBAC)**: User and Admin roles
- **Rate Limiting**: Brute force protection

### Authentication Implementation

**Module**: `security/auth.py`

#### API Key Authentication
```python
from security.auth import require_auth, require_admin

@api_bp.route('/protected-endpoint', methods=['GET'])
@require_auth('user')  # Requires user role
def protected_endpoint():
    # User info available via request.user_info
    return jsonify({'message': 'Authenticated'})

@api_bp.route('/admin-endpoint', methods=['POST'])
@require_admin  # Requires admin role
def admin_endpoint():
    return jsonify({'message': 'Admin access granted'})
```

#### Security Features
1. **Secure Key Generation**: Uses `secrets.token_urlsafe(32)` for cryptographic randomness
2. **Key Hashing**: PBKDF2-HMAC-SHA256 with 100,000 iterations
3. **Salt Storage**: Unique salt per key for rainbow table protection
4. **Brute Force Protection**: Lockout after 5 failed attempts (15 min duration)
5. **Session Tracking**: Active session monitoring and invalidation

### Role-Based Access Control

**Roles**:
- `user`: Standard API access (training, cases, viewer)
- `admin`: Full system access (canary, quantum, system management)

**Role Verification**:
```python
def has_role(self, user_info: Dict, required_role: str) -> bool:
    """Check if user has required role."""
    roles = user_info.get('roles', [])
    return required_role in roles or 'admin' in roles
```

---

## PHI Compliance

### Overview
All endpoints handling Protected Health Information (PHI) implement HIPAA-compliant safeguards.

### PHI Protection Mechanisms

#### 1. PHI Scrubber
**Module**: `src/aimedres/security/phi_scrubber.py`

**Features**:
- Detects 18 HIPAA Safe Harbor PHI categories
- De-identifies data at ingestion
- Maintains clinical term whitelist
- Hash-based consistent anonymization

**Integration**:
```python
from aimedres.security.phi_scrubber import PHIScrubber

scrubber = PHIScrubber()
cleaned_data = scrubber.scrub(patient_data)
```

#### 2. Response Sanitization
All API responses are sanitized to remove PHI:
- Patient names → Hashed IDs
- Addresses → Geographic region only
- Dates → Year only (configurable)
- Contact info → Removed
- Medical record numbers → Anonymized

#### 3. Error Message Protection
Error messages never contain PHI:
```python
# BAD - Leaks PHI
return jsonify({'error': f'Patient {patient_name} not found'}), 404

# GOOD - PHI-safe
return jsonify({'error': 'Patient not found'}), 404
```

### PHI Compliance by Endpoint Category

#### Viewer Endpoints (`/api/viewer/*`)
- **Authentication**: Required for all endpoints
- **PHI Protection**: Patient IDs anonymized, DICOM metadata scrubbed
- **Audit**: All image access logged with user, timestamp, reason

#### Case Management (`/api/v1/cases/*`)
- **Authentication**: User role required
- **PHI Protection**: De-identified patient data, no PII in responses
- **Audit**: Case access, approvals, rejections logged

#### Training (`/api/v1/training/*`)
- **Authentication**: User role required
- **PHI Protection**: Training data pre-scrubbed, no PHI in logs
- **Audit**: Training job submissions, model access logged

#### Canary Deployment (`/api/v1/canary/*`)
- **Authentication**: Admin role required
- **PHI Protection**: No patient data in deployment metrics
- **Audit**: All deployment operations logged

#### Quantum Key Management (`/api/v1/quantum/*`)
- **Authentication**: Admin role required
- **PHI Protection**: N/A (infrastructure only)
- **Audit**: All key operations logged

---

## Audit Logging

### Overview
Comprehensive audit logging for compliance, security monitoring, and incident response.

### Audit Log Implementation

#### 1. Blockchain Audit Trail
**Module**: `security/blockchain_records.py`

**Features**:
- Immutable audit records
- Cryptographic chain integrity
- Tamper-evident logging
- HIPAA/GDPR compliant

**Usage**:
```python
from security.blockchain_records import BlockchainMedicalRecords

blockchain = BlockchainMedicalRecords()
blockchain.record_audit_event({
    'event_type': 'data_access',
    'user_id': user_id,
    'resource': 'patient_record',
    'action': 'read',
    'timestamp': datetime.now().isoformat(),
    'ip_address': request.remote_addr
})
```

#### 2. HIPAA Audit Logger
**Module**: `security/hipaa_audit.py`

**Features**:
- HIPAA-compliant audit records
- User access tracking
- Data modification logging
- Export for compliance reporting

#### 3. Security Monitor
**Module**: `security/monitoring.py`

**Features**:
- Real-time security event monitoring
- Intrusion detection
- Anomaly detection
- Alert generation

### Audited Events

#### Authentication Events
- Login attempts (success/failure)
- API key validation
- Token generation/expiration
- Role changes
- Account lockouts

#### Data Access Events
- Patient record access
- Medical image viewing
- Case retrievals
- Report generation

#### Data Modification Events
- Case approvals/rejections
- Clinical decision overrides
- Model predictions
- Training data ingestion

#### System Operations
- Model deployments (canary)
- Key rotations (quantum)
- Configuration changes
- System maintenance

#### Security Events
- Failed authentication attempts
- Unauthorized access attempts
- PHI scrubbing events
- Encryption/decryption operations

### Audit Log Format

```json
{
  "event_id": "evt_abc123",
  "timestamp": "2025-11-04T00:00:00Z",
  "event_type": "data_access",
  "user_id": "user_456",
  "user_role": "clinician",
  "resource_type": "patient_record",
  "resource_id": "pat_xyz789",
  "action": "read",
  "ip_address": "10.0.1.100",
  "user_agent": "AiMedRes-Frontend/1.0",
  "result": "success",
  "metadata": {
    "reason": "clinical_review",
    "session_id": "sess_def789"
  },
  "chain_hash": "0a1b2c3d...",
  "previous_hash": "9z8y7x6w..."
}
```

---

## Endpoint Security Matrix

### Training API (`/api/v1/training/*`)

| Endpoint | Method | Auth | Role | PHI | Audit | Notes |
|----------|--------|------|------|-----|-------|-------|
| `/training/submit` | POST | ✅ | user | ✅ | ✅ | Training data pre-scrubbed |
| `/training/status/<id>` | GET | ✅ | user | ✅ | ✅ | User can only see own jobs |
| `/training/jobs` | GET | ✅ | user | ✅ | ✅ | Filtered by user |
| `/training/cancel/<id>` | POST | ✅ | user | ✅ | ✅ | User can only cancel own |
| `/training/model/<id>` | GET | ✅ | user | ✅ | ✅ | Encrypted model download |
| `/admin/training/status` | GET | ✅ | admin | N/A | ✅ | System-wide stats |

### Viewer API (`/api/viewer/*`)

| Endpoint | Method | Auth | Role | PHI | Audit | Notes |
|----------|--------|------|------|-----|-------|-------|
| `/viewer/session` | POST | ✅ | user | ✅ | ✅ | Create viewer session |
| `/viewer/brain/<id>` | GET | ✅ | user | ✅ | ✅ | Patient ID anonymized |
| `/viewer/dicom/study/<id>` | GET | ✅ | user | ✅ | ✅ | DICOM metadata scrubbed |
| `/viewer/dicom/series/<id>` | GET | ✅ | user | ✅ | ✅ | Streaming with PHI removal |
| `/viewer/health` | GET | ❌ | public | N/A | ✅ | Health check only |

### Canary Deployment API (`/api/v1/canary/*`)

| Endpoint | Method | Auth | Role | PHI | Audit | Notes |
|----------|--------|------|------|-----|-------|-------|
| `/canary/deployments` | GET | ✅ | user | N/A | ✅ | Read-only for users |
| `/canary/deployments/<id>` | GET | ✅ | user | N/A | ✅ | Deployment details |
| `/canary/deployments/<id>/metrics` | GET | ✅ | user | N/A | ✅ | Real-time metrics |
| `/canary/deployments/<id>/rollback` | POST | ✅ | admin | N/A | ✅ | Admin action required |
| `/canary/deployments/<id>/promote` | POST | ✅ | admin | N/A | ✅ | Admin action required |
| `/canary/health` | GET | ❌ | public | N/A | ✅ | Health check only |

### Quantum Key Management API (`/api/v1/quantum/*`)

| Endpoint | Method | Auth | Role | PHI | Audit | Notes |
|----------|--------|------|------|-----|-------|-------|
| `/quantum/keys` | GET | ✅ | admin | N/A | ✅ | List all keys |
| `/quantum/keys/<id>` | GET | ✅ | admin | N/A | ✅ | Key details |
| `/quantum/stats` | GET | ✅ | admin | N/A | ✅ | System statistics |
| `/quantum/policy` | GET | ✅ | admin | N/A | ✅ | Rotation policy |
| `/quantum/policy` | PUT | ✅ | admin | N/A | ✅ | Update policy |
| `/quantum/keys/<id>/rotate` | POST | ✅ | admin | N/A | ✅ | Manual rotation |
| `/quantum/history` | GET | ✅ | admin | N/A | ✅ | Rotation history |
| `/quantum/health` | GET | ❌ | public | N/A | ✅ | Health check only |

### Cases API (`/api/v1/cases/*`)

| Endpoint | Method | Auth | Role | PHI | Audit | Notes |
|----------|--------|------|------|-----|-------|-------|
| `/cases` | GET | ✅ | user | ✅ | ✅ | De-identified patient data |
| `/cases/<id>` | GET | ✅ | user | ✅ | ✅ | Case details with explainability |
| `/cases/<id>/approve` | POST | ✅ | user | ✅ | ✅ | Human-in-loop approval |

---

## Security Testing

### Test Suite
**File**: `tests/test_api_security_compliance.py`

### Test Categories

#### 1. Authentication Tests
- Verify all endpoints require auth
- Test invalid credentials rejection
- Test role-based access control
- Test admin-only endpoint protection

#### 2. PHI Compliance Tests
- Verify no PHI in error messages
- Test de-identification in responses
- Verify audit log PHI protection
- Test data sanitization

#### 3. Audit Logging Tests
- Verify all operations logged
- Test log immutability
- Verify timestamp accuracy
- Test log export functionality

#### 4. Security Tests
- SQL injection protection
- XSS prevention
- Rate limiting
- Input validation

### Running Security Tests

```bash
# Run all security tests
pytest tests/test_api_security_compliance.py -v

# Run specific test class
pytest tests/test_api_security_compliance.py::TestAuthenticationRequired -v

# Run PHI compliance tests
pytest tests/test_api_security_compliance.py::TestPHICompliance -v

# Run audit logging tests
pytest tests/test_api_security_compliance.py::TestAuditLogging -v
```

### E2E Security Tests

```bash
# Run Cypress E2E tests
cd frontend
npm run test:e2e

# Run specific test suite
npx cypress run --spec cypress/e2e/canary-dashboard.cy.ts
```

---

## Compliance Certifications

### HIPAA Compliance
✅ **Administrative Safeguards**: User authentication, role-based access
✅ **Physical Safeguards**: Encrypted data at rest and in transit
✅ **Technical Safeguards**: PHI scrubbing, audit logging, access controls
✅ **Organizational Requirements**: Business associate agreements supported
✅ **Policies and Procedures**: Security policies documented
✅ **Documentation**: Audit logs, incident reports, compliance records

### GDPR Compliance
✅ **Data Protection**: Encryption, access controls
✅ **Right to Access**: Audit logs provide access history
✅ **Right to Erasure**: Data deletion with audit trail
✅ **Data Portability**: Export functionality
✅ **Privacy by Design**: PHI protection built-in
✅ **Data Breach Notification**: Security monitoring and alerting

---

## Security Best Practices

### For Developers
1. Always use `@require_auth` or `@require_admin` decorators
2. Never log or return PHI in responses
3. Use PHIScrubber for all patient data
4. Record all operations in audit log
5. Use HTTPS in production
6. Rotate API keys regularly
7. Implement rate limiting
8. Validate all inputs
9. Use prepared statements for DB queries
10. Keep dependencies updated

### For Administrators
1. Review audit logs regularly
2. Monitor failed authentication attempts
3. Enforce strong password policies
4. Enable MFA for admin accounts
5. Rotate quantum keys per policy
6. Monitor canary deployments
7. Review security alerts
8. Conduct regular security audits
9. Train staff on PHI handling
10. Maintain incident response plan

### For API Users
1. Store API keys securely
2. Use HTTPS for all requests
3. Never log API keys
4. Rotate keys if compromised
5. Report security issues immediately
6. Follow rate limits
7. Respect role permissions
8. Handle errors gracefully
9. Validate responses
10. Keep client libraries updated

---

## Security Contact

For security issues or vulnerability reports:
- Email: security@aimedres.example.com
- See: [SECURITY.md](../SECURITY.md)

---

## References

- [HIPAA Compliance Guide](https://www.hhs.gov/hipaa)
- [GDPR Documentation](https://gdpr.eu)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**Last Updated**: 2025-11-04
**Version**: 1.0.0
**Status**: ✅ All endpoints secured and compliant
