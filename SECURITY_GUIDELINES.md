# Security Guidelines for Contributors

## Overview

DuetMind Adaptive implements enterprise-grade security measures to protect medical data and ensure compliance with healthcare regulations (HIPAA, GDPR). All contributors must follow these security guidelines.

## Security Architecture

### 1. Authentication & Authorization
- **Secure API Keys**: Cryptographically generated keys with PBKDF2 hashing
- **Role-Based Access Control (RBAC)**: Admin, user, and custom role permissions
- **JWT Tokens**: Secure session management with configurable expiration
- **Multi-Factor Authentication**: Support for additional security layers

### 2. Data Protection
- **Encryption at Rest**: AES-256 encryption for all sensitive data
- **Encryption in Transit**: HTTPS/TLS for all API communications
- **Data Anonymization**: Automatic PII removal and hash-based de-identification
- **Secure Storage**: Isolated directories with proper access controls

### 3. Privacy Compliance
- **GDPR Compliance**: Right to erasure, data portability, consent management
- **HIPAA Compliance**: PHI protection, audit trails, minimum necessary access
- **Data Retention**: Automated cleanup based on configurable policies
- **Audit Logging**: Comprehensive trails for all data access and processing

## Development Security Requirements

### Code Security

#### 1. Input Validation
```python
# Always validate and sanitize user input
from security import InputValidator

validator = InputValidator()
is_valid, errors = validator.validate_medical_data(input_data)
if not is_valid:
    raise ValueError(f"Invalid input: {errors}")
```

#### 2. Secure Data Handling
```python
# Use secure medical data processor
from secure_medical_processor import SecureMedicalDataProcessor

processor = SecureMedicalDataProcessor(config)
dataset_id = processor.load_and_secure_dataset(
    dataset_source='kaggle',
    dataset_params=params,
    user_id=user_id,
    purpose='training'
)
```

#### 3. Authentication Requirements
```python
# Protect endpoints with authentication
from security import require_auth, require_admin

@app.route('/api/medical/process')
@require_auth()
def medical_endpoint():
    # Access user info via g.user_info
    user_id = g.user_info['user_id']
    # ... implementation
```

### Medical Data Security

#### 1. Data Classification
- **Public**: Non-sensitive system information
- **Internal**: Aggregated, anonymized statistics  
- **Confidential**: De-identified medical data
- **Restricted**: Raw medical data with PII (requires encryption)

#### 2. Data Handling Rules
- Never store raw PII in logs or temporary files
- Always use secure processors for medical data
- Implement data isolation between training and inference
- Apply anonymization before any data sharing

#### 3. Model Security
```python
# Encrypt model parameters
encrypted_model = data_encryption.encrypt_model_parameters(model_data)

# Secure model storage
model_path = secure_storage_dir / f"{model_id}.encrypted"
```

## Configuration Security

### Environment Variables
Set these environment variables for production:

```bash
# Master encryption key (64+ characters)
export DUETMIND_MASTER_KEY="your-secure-master-key-here"

# JWT secret (32+ characters)  
export DUETMIND_JWT_SECRET="your-jwt-secret-here"

# Database encryption key
export DUETMIND_DB_KEY="your-database-key-here"
```

### Security Configuration
```python
security_config = {
    'enable_security': True,
    'https_only': True,
    'secure_workspace': '/secure/medical/workspace',
    'privacy_compliance': True,
    'audit_logging': True,
    'retention_policy': {
        'medical_data_retention_days': 2555,  # 7 years
        'anonymize_after_days': 30,
        'enable_automatic_deletion': True
    },
    'rate_limiting': {
        'admin_rate_limit': 500,
        'user_rate_limit': 100,
        'burst_limit': 50
    }
}
```

## Security Testing

### 1. Input Validation Testing
```python
def test_input_validation():
    # Test SQL injection protection
    malicious_input = "'; DROP TABLE patients; --"
    validator = InputValidator()
    assert not validator.validate_sql_injection(malicious_input)
    
    # Test XSS protection
    xss_input = "<script>alert('xss')</script>"
    assert not validator.validate_xss(xss_input)
```

### 2. Authentication Testing
```python
def test_authentication():
    # Test with invalid API key
    response = client.get('/api/medical/data', 
                         headers={'X-API-Key': 'invalid'})
    assert response.status_code == 401
    
    # Test rate limiting
    for _ in range(101):  # Exceed rate limit
        response = client.get('/api/data')
    assert response.status_code == 429
```

### 3. Privacy Testing
```python
def test_privacy_protection():
    # Verify PII removal
    original_data = {'patient_id': '12345', 'name': 'John Doe'}
    anonymized = processor.anonymize_medical_data(original_data)
    assert 'patient_id' not in anonymized
    assert 'name' not in anonymized
```

## Incident Response

### 1. Security Event Detection
The system automatically detects:
- Failed authentication attempts
- Unusual API usage patterns
- Potential data breaches
- System intrusion attempts

### 2. Response Procedures
1. **Immediate**: Automatic rate limiting and IP blocking
2. **Alert**: Notify security team via configured webhooks
3. **Investigation**: Review security logs and audit trails
4. **Containment**: Isolate affected systems if necessary
5. **Recovery**: Restore services with enhanced monitoring

### 3. Reporting
All security incidents are:
- Logged with detailed context
- Stored in audit database
- Available via admin dashboard
- Reported to compliance officers

## Compliance Checklists

### HIPAA Compliance ✓
- [x] Administrative Safeguards
- [x] Physical Safeguards  
- [x] Technical Safeguards
- [x] Audit Controls
- [x] Data Integrity
- [x] Transmission Security

### GDPR Compliance ✓
- [x] Data Protection by Design
- [x] Right to Erasure
- [x] Data Portability
- [x] Consent Management
- [x] Breach Notification
- [x] Privacy Impact Assessment

## Emergency Contacts

### Security Team
- **Security Lead**: security@duetmind.ai
- **Privacy Officer**: privacy@duetmind.ai
- **Emergency Hotline**: +1-XXX-XXX-XXXX

### Compliance
- **HIPAA Officer**: hipaa@duetmind.ai
- **GDPR Officer**: gdpr@duetmind.ai
- **Legal Team**: legal@duetmind.ai

## Regular Security Tasks

### Daily
- Monitor security alerts
- Review access logs
- Check system health

### Weekly  
- Security scan reports
- Update threat intelligence
- Review user permissions

### Monthly
- Security audit review
- Compliance assessment
- Incident response drill

### Quarterly
- Security policy review
- Penetration testing
- Staff security training

---

**Remember**: Security is everyone's responsibility. When in doubt, err on the side of caution and consult the security team.