# Security Configuration Guide

**Version**: 1.0.0 | **Last Updated**: November 2025

## Overview

This guide provides comprehensive instructions for configuring AiMedRes's security features for production deployment. Follow these steps to ensure maximum security and compliance.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM
- **Storage**: Minimum 100GB for secure workspace
- **Network**: HTTPS/TLS 1.3 support

### Dependencies
```bash
# Install required security packages
pip install cryptography>=3.4.8
pip install pyjwt>=2.4.0
pip install bcrypt>=3.2.0
pip install pyotp>=2.6.0  # For MFA support
```

## Initial Security Setup

### 1. Environment Configuration

Create a secure environment file `.env.security`:

```bash
# Master encryption keys (generate with openssl rand -base64 64)
DUETMIND_MASTER_KEY="your-64-character-master-encryption-key-here"
DUETMIND_JWT_SECRET="your-32-character-jwt-secret-here"
DUETMIND_DB_ENCRYPTION_KEY="your-database-encryption-key-here"

# Security settings
DUETMIND_SECURITY_ENABLED=true
DUETMIND_HTTPS_ONLY=true
DUETMIND_SECURE_COOKIES=true

# Workspace configuration
DUETMIND_SECURE_WORKSPACE="/opt/duetmind/secure"
DUETMIND_LOG_LEVEL=INFO

# Privacy settings
DUETMIND_PRIVACY_COMPLIANCE=true
DUETMIND_AUDIT_LOGGING=true
DUETMIND_DATA_ANONYMIZATION=true

# Rate limiting
DUETMIND_RATE_LIMIT_ENABLED=true
DUETMIND_ADMIN_RATE_LIMIT=500
DUETMIND_USER_RATE_LIMIT=100

# Monitoring
DUETMIND_SECURITY_MONITORING=true
DUETMIND_INTRUSION_DETECTION=true
```

### 2. Generate Security Keys

```bash
#!/bin/bash
# generate_keys.sh - Generate secure keys for production

echo "Generating secure keys for DuetMind Adaptive..."

# Master encryption key (64 bytes = 512 bits)
MASTER_KEY=$(openssl rand -base64 64 | tr -d '\n')
echo "DUETMIND_MASTER_KEY=\"$MASTER_KEY\""

# JWT secret (32 bytes = 256 bits)  
JWT_SECRET=$(openssl rand -base64 32 | tr -d '\n')
echo "DUETMIND_JWT_SECRET=\"$JWT_SECRET\""

# Database encryption key
DB_KEY=$(openssl rand -base64 32 | tr -d '\n')
echo "DUETMIND_DB_ENCRYPTION_KEY=\"$DB_KEY\""

# API key salt
API_SALT=$(openssl rand -base64 16 | tr -d '\n')
echo "DUETMIND_API_SALT=\"$API_SALT\""

echo "Keys generated successfully. Add these to your .env.security file."
```

### 3. SSL/TLS Configuration

#### Generate SSL Certificate (Self-signed for development)
```bash
# Generate private key
openssl genrsa -out duetmind.key 4096

# Generate certificate signing request
openssl req -new -key duetmind.key -out duetmind.csr

# Generate self-signed certificate (development only)
openssl x509 -req -days 365 -in duetmind.csr -signkey duetmind.key -out duetmind.crt

# For production, use a certificate from a trusted CA
```

#### SSL Context Configuration
```python
import ssl

def create_ssl_context():
    """Create SSL context for HTTPS."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    
    # Load certificate and private key
    context.load_cert_chain('duetmind.crt', 'duetmind.key')
    
    # Security settings
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1
    context.options |= ssl.OP_SINGLE_DH_USE
    context.options |= ssl.OP_SINGLE_ECDH_USE
    
    return context
```

## Security Configuration

### 1. Authentication Configuration

```python
# config/security.py
SECURITY_CONFIG = {
    # Authentication settings
    'authentication': {
        'enable_mfa': True,
        'password_policy': {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True, 
            'require_numbers': True,
            'require_symbols': True,
            'password_history': 5,
            'max_age_days': 90
        },
        'session_management': {
            'timeout_minutes': 30,
            'absolute_timeout_hours': 8,
            'concurrent_sessions': 1
        },
        'lockout_policy': {
            'max_attempts': 5,
            'lockout_duration_minutes': 15,
            'progressive_delays': True
        }
    },
    
    # API security
    'api_security': {
        'rate_limiting': {
            'enabled': True,
            'admin_limit': 500,      # requests per minute
            'user_limit': 100,       # requests per minute
            'burst_limit': 50,       # burst allowance
            'window_size': 60        # seconds
        },
        'request_validation': {
            'max_request_size': 10485760,  # 10MB
            'max_json_depth': 10,
            'allowed_content_types': ['application/json'],
            'require_content_length': True
        }
    },
    
    # Data protection
    'data_protection': {
        'encryption': {
            'algorithm': 'AES-256-GCM',
            'key_derivation': 'PBKDF2',
            'iterations': 100000,
            'salt_length': 32
        },
        'anonymization': {
            'auto_anonymize': True,
            'k_anonymity': 5,
            'l_diversity': True,
            't_closeness': 0.2
        }
    }
}
```

### 2. Privacy Configuration

```python
# config/privacy.py
PRIVACY_CONFIG = {
    # GDPR compliance
    'gdpr': {
        'enabled': True,
        'data_controller': 'Your Organization',
        'dpo_contact': 'dpo@yourorg.com',
        'legal_basis': 'healthcare_research',
        'retention_periods': {
            'medical_data': 2555,      # 7 years in days
            'research_data': 3650,     # 10 years
            'audit_logs': 2555,        # 7 years
            'api_logs': 90             # 90 days
        }
    },
    
    # HIPAA compliance
    'hipaa': {
        'enabled': True,
        'covered_entity': True,
        'business_associate': False,
        'minimum_necessary': True,
        'safeguards': {
            'administrative': True,
            'physical': True,
            'technical': True
        }
    },
    
    # Data subject rights
    'data_rights': {
        'right_to_access': True,
        'right_to_rectification': True,
        'right_to_erasure': True,
        'right_to_portability': True,
        'right_to_object': True,
        'automated_responses': True
    }
}
```

### 3. Monitoring Configuration

```python
# config/monitoring.py
MONITORING_CONFIG = {
    # Security monitoring
    'security_monitoring': {
        'enabled': True,
        'real_time_alerts': True,
        'alert_thresholds': {
            'failed_auth_rate': 10,        # per minute
            'error_rate': 0.05,            # 5%
            'response_time_anomaly': 3.0,  # std deviations
            'concurrent_limit': 100
        },
        'log_retention_days': 90
    },
    
    # Intrusion detection
    'intrusion_detection': {
        'enabled': True,
        'patterns': [
            'sql_injection',
            'xss_attacks',
            'brute_force',
            'anomalous_requests'
        ],
        'block_suspicious_ips': True,
        'quarantine_duration_hours': 24
    },
    
    # Audit logging
    'audit_logging': {
        'enabled': True,
        'log_all_access': True,
        'include_request_data': False,  # Privacy protection
        'include_response_data': False,
        'log_level': 'INFO'
    }
}
```

## Production Deployment

### 1. Docker Security Configuration

```dockerfile
# Dockerfile.security
FROM python:3.9-slim

# Security hardening
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r duetmind && useradd -r -g duetmind duetmind

# Set secure working directory
WORKDIR /app
RUN chown duetmind:duetmind /app

# Copy application
COPY --chown=duetmind:duetmind . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create secure directories
RUN mkdir -p /opt/duetmind/secure && \
    chown duetmind:duetmind /opt/duetmind/secure && \
    chmod 700 /opt/duetmind/secure

# Switch to non-root user
USER duetmind

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f https://localhost:8443/health || exit 1

# Secure entry point
ENTRYPOINT ["python", "secure_main.py"]
```

### 2. Kubernetes Security Configuration

```yaml
# k8s/security-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: duetmind-security-config
data:
  security.conf: |
    security:
      enabled: true
      https_only: true
      audit_logging: true
    
    privacy:
      gdpr_enabled: true
      hipaa_enabled: true
      data_minimization: true

---
apiVersion: v1
kind: Secret
metadata:
  name: duetmind-secrets
type: Opaque
data:
  master-key: <base64-encoded-master-key>
  jwt-secret: <base64-encoded-jwt-secret>
  db-key: <base64-encoded-db-key>

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: duetmind-secure
spec:
  replicas: 3
  selector:
    matchLabels:
      app: duetmind-secure
  template:
    metadata:
      labels:
        app: duetmind-secure
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: duetmind
        image: duetmind:secure
        ports:
        - containerPort: 8443
          name: https
        env:
        - name: DUETMIND_MASTER_KEY
          valueFrom:
            secretKeyRef:
              name: duetmind-secrets
              key: master-key
        volumeMounts:
        - name: secure-workspace
          mountPath: /opt/duetmind/secure
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      volumes:
      - name: secure-workspace
        persistentVolumeClaim:
          claimName: duetmind-secure-pvc
```

### 3. Nginx Security Configuration

```nginx
# nginx/duetmind-secure.conf
server {
    listen 443 ssl http2;
    server_name duetmind.yourorg.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/duetmind.crt;
    ssl_certificate_key /etc/ssl/private/duetmind.key;
    ssl_protocols TLSv1.3 TLSv1.2;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Proxy to application
    location / {
        proxy_pass https://127.0.0.1:8443;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Security monitoring
    access_log /var/log/nginx/duetmind-access.log combined;
    error_log /var/log/nginx/duetmind-error.log warn;
}
```

## Security Testing

### 1. Automated Security Tests

```python
# tests/test_security.py
import pytest
from security import SecureAuthManager, InputValidator

def test_api_key_generation():
    """Test secure API key generation."""
    auth_manager = SecureAuthManager({})
    api_key = auth_manager._generate_api_key('test_user', ['user'])
    
    assert api_key.startswith('dmk_')
    assert len(api_key) > 40
    assert api_key in auth_manager.api_keys

def test_input_validation():
    """Test input validation against attacks."""
    validator = InputValidator()
    
    # Test SQL injection
    sql_attack = "'; DROP TABLE users; --"
    assert not validator.validate_sql_injection(sql_attack)
    
    # Test XSS
    xss_attack = "<script>alert('xss')</script>"
    assert not validator.validate_xss(xss_attack)

def test_rate_limiting():
    """Test rate limiting functionality."""
    # Implementation depends on your rate limiting logic
    pass

def test_encryption():
    """Test data encryption/decryption."""
    from security import DataEncryption
    
    encryption = DataEncryption()
    test_data = {'sensitive': 'medical_data'}
    
    encrypted = encryption.encrypt_data(test_data)
    decrypted = encryption.decrypt_data(encrypted)
    
    assert decrypted == test_data
```

### 2. Security Scan Commands

```bash
#!/bin/bash
# security_scan.sh - Run security scans

echo "Running security scans..."

# Dependency vulnerability scan
safety check

# Static code analysis
bandit -r . -f json -o security_report.json

# Secret detection
truffleHog --regex --entropy=False .

# OWASP ZAP API scan (if ZAP is installed)
if command -v zap-baseline.py &> /dev/null; then
    zap-baseline.py -t https://localhost:8443 -J zap_report.json
fi

echo "Security scans completed. Check reports for issues."
```

## Maintenance and Updates

### 1. Security Update Schedule

```bash
#!/bin/bash
# update_security.sh - Security maintenance script

# Update dependencies
pip install --upgrade cryptography pyjwt bcrypt

# Rotate keys (monthly)
if [ "$(date +%d)" = "01" ]; then
    echo "Rotating security keys..."
    # Add key rotation logic
fi

# Check for security updates
apt list --upgradable | grep -i security

# Run security tests
python -m pytest tests/test_security.py

echo "Security maintenance completed."
```

### 2. Monitoring and Alerting

```python
# monitoring/security_alerts.py
def setup_security_alerts():
    """Configure security alert notifications."""
    
    alert_config = {
        'webhook_url': os.getenv('SECURITY_WEBHOOK_URL'),
        'email_notifications': True,
        'alert_levels': {
            'critical': ['security_team@yourorg.com'],
            'warning': ['devops@yourorg.com'],
            'info': ['admin@yourorg.com']
        }
    }
    
    return alert_config
```

## Compliance Verification

### 1. HIPAA Compliance Checklist

- [ ] **Administrative Safeguards**
  - [ ] Security Officer assigned
  - [ ] Workforce training completed
  - [ ] Access procedures documented
  - [ ] Incident response plan active

- [ ] **Physical Safeguards**
  - [ ] Facility access controls
  - [ ] Workstation security
  - [ ] Media disposal procedures

- [ ] **Technical Safeguards**
  - [ ] Access controls implemented
  - [ ] Audit logging active
  - [ ] Data integrity controls
  - [ ] Transmission security (TLS)

### 2. GDPR Compliance Checklist

- [ ] **Legal Requirements**
  - [ ] Data Protection Impact Assessment
  - [ ] Privacy notices updated
  - [ ] Consent mechanisms implemented
  - [ ] Data Processing Agreement

- [ ] **Technical Measures**
  - [ ] Data minimization implemented
  - [ ] Pseudonymization/anonymization
  - [ ] Encryption at rest and in transit
  - [ ] Access controls and monitoring

---

**Security Note**: This configuration provides enterprise-grade security. Adjust settings based on your specific risk assessment and compliance requirements. Regularly review and update security configurations as threats evolve.