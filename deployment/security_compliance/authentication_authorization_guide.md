# Authentication & Authorization Configuration Guide

## Overview

This guide provides comprehensive instructions for configuring authentication and authorization in AiMedRes, including user/group access levels, hospital authentication integration (LDAP/SSO), and audit logging.

## Authentication Architecture

```
┌──────────────┐
│   User/CLI   │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Authentication      │
│  - LDAP/AD          │
│  - OIDC/SAML        │
│  - API Keys         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Authorization       │
│  - RBAC             │
│  - Permissions      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Audit Logging       │
│  - All Access       │
│  - All Changes      │
└──────────────────────┘
```

## Access Levels and Roles

### Predefined Roles

| Role | Permissions | Use Case |
|------|-------------|----------|
| **admin** | Full system access, user management, configuration | System administrators |
| **clinician** | Patient data access, run assessments, view results | Physicians, nurses |
| **researcher** | Anonymized data access, model training, analytics | Research staff |
| **auditor** | Read-only access to logs, reports, audit trails | Compliance officers |
| **api_user** | Programmatic access via API | External systems |

### Role Configuration

```python
# Role definitions in src/aimedres/security/auth.py
from src.aimedres.security.auth import SecureAuthManager

ROLES = {
    'admin': {
        'permissions': [
            'system:*',
            'user:*',
            'data:*',
            'model:*',
            'config:*'
        ],
        'description': 'Full system administrator'
    },
    'clinician': {
        'permissions': [
            'patient:read',
            'patient:write',
            'assessment:run',
            'assessment:read',
            'results:read'
        ],
        'description': 'Clinical staff with patient access'
    },
    'researcher': {
        'permissions': [
            'data:read_anonymized',
            'model:train',
            'model:evaluate',
            'analytics:run'
        ],
        'description': 'Research staff with anonymized data access'
    },
    'auditor': {
        'permissions': [
            'audit:read',
            'logs:read',
            'reports:read'
        ],
        'description': 'Compliance and audit staff'
    },
    'api_user': {
        'permissions': [
            'api:read',
            'api:write'
        ],
        'description': 'External system integration'
    }
}
```

## Local Authentication Setup

### Initialize Authentication Manager

```python
from src.aimedres.security.auth import SecureAuthManager
import os

# Initialize auth manager
auth_manager = SecureAuthManager({
    'jwt_secret': os.getenv('JWT_SECRET_KEY'),
    'token_expiry_hours': 24,
    'max_failed_attempts': 5,
    'lockout_duration_minutes': 15
})
```

### Create Users

```python
def create_user(username: str, password: str, roles: list):
    """Create a new user with specified roles"""
    
    # Generate secure API key
    api_key = auth_manager._generate_api_key(username, roles)
    
    # Hash password for web login
    password_hash = auth_manager._hash_password(password)
    
    # Store user
    user_data = {
        'username': username,
        'password_hash': password_hash,
        'api_key': api_key,
        'roles': roles,
        'created_at': datetime.now().isoformat(),
        'active': True
    }
    
    # Save to database
    save_user(user_data)
    
    print(f"✅ User created: {username}")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Roles: {', '.join(roles)}")
    
    return api_key

# Create example users
admin_key = create_user('admin', 'secure_password', ['admin'])
clinician_key = create_user('dr_smith', 'secure_password', ['clinician'])
researcher_key = create_user('researcher1', 'secure_password', ['researcher'])
auditor_key = create_user('auditor1', 'secure_password', ['auditor'])
```

### Password Policy

```python
import re

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password meets security requirements"""
    
    errors = []
    
    # Minimum length
    if len(password) < 12:
        errors.append("Password must be at least 12 characters")
    
    # Uppercase letter
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain an uppercase letter")
    
    # Lowercase letter
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain a lowercase letter")
    
    # Digit
    if not re.search(r'\d', password):
        errors.append("Password must contain a digit")
    
    # Special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain a special character")
    
    # Common passwords check
    common_passwords = ['password', '12345678', 'qwerty', 'admin']
    if password.lower() in common_passwords:
        errors.append("Password is too common")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "Password meets requirements"

# Usage
is_valid, message = validate_password("MySecure$Pass123")
print(message)
```

### API Key Authentication

```python
from flask import request, jsonify
from functools import wraps

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        # Validate API key
        user = auth_manager.validate_api_key(api_key)
        
        if not user:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Add user to request context
        request.user = user
        
        return f(*args, **kwargs)
    
    return decorated_function

# Protected endpoint
@app.route('/api/v1/secure/data', methods=['GET'])
@require_api_key
def get_secure_data():
    user = request.user
    return jsonify({
        'message': 'Secure data',
        'user': user['username'],
        'roles': user['roles']
    })
```

## LDAP/Active Directory Integration

### Install Dependencies

```bash
pip install python-ldap
```

### LDAP Configuration

```python
import ldap
from typing import Optional, Dict

class LDAPAuthenticator:
    """LDAP/Active Directory authentication"""
    
    def __init__(self, config: Dict):
        self.ldap_server = config['ldap_server']
        self.base_dn = config['base_dn']
        self.bind_dn = config.get('bind_dn')
        self.bind_password = config.get('bind_password')
        self.user_search_base = config.get('user_search_base', self.base_dn)
        self.group_search_base = config.get('group_search_base', self.base_dn)
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user against LDAP"""
        
        try:
            # Connect to LDAP server
            conn = ldap.initialize(self.ldap_server)
            conn.set_option(ldap.OPT_REFERRALS, 0)
            conn.set_option(ldap.OPT_PROTOCOL_VERSION, 3)
            conn.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)
            
            # Construct user DN
            user_dn = f"uid={username},{self.user_search_base}"
            
            # Try to bind with user credentials
            try:
                conn.simple_bind_s(user_dn, password)
            except ldap.INVALID_CREDENTIALS:
                print(f"❌ Invalid credentials for {username}")
                return None
            
            # Get user attributes
            result = conn.search_s(
                user_dn,
                ldap.SCOPE_BASE,
                '(objectClass=*)',
                ['cn', 'mail', 'memberOf']
            )
            
            if not result:
                return None
            
            dn, attrs = result[0]
            
            # Extract groups
            groups = []
            if 'memberOf' in attrs:
                for group_dn in attrs['memberOf']:
                    # Extract group name from DN
                    group_name = group_dn.decode('utf-8').split(',')[0].split('=')[1]
                    groups.append(group_name)
            
            user_info = {
                'username': username,
                'email': attrs.get('mail', [b''])[0].decode('utf-8'),
                'full_name': attrs.get('cn', [b''])[0].decode('utf-8'),
                'ldap_groups': groups,
                'roles': self._map_groups_to_roles(groups)
            }
            
            conn.unbind_s()
            
            print(f"✅ LDAP authentication successful: {username}")
            return user_info
            
        except ldap.LDAPError as e:
            print(f"❌ LDAP error: {e}")
            return None
    
    def _map_groups_to_roles(self, ldap_groups: list) -> list:
        """Map LDAP groups to AiMedRes roles"""
        
        group_role_mapping = {
            'Domain Admins': ['admin'],
            'IT Admins': ['admin'],
            'Physicians': ['clinician'],
            'Nurses': ['clinician'],
            'Research Staff': ['researcher'],
            'Compliance Officers': ['auditor'],
        }
        
        roles = set()
        for group in ldap_groups:
            if group in group_role_mapping:
                roles.update(group_role_mapping[group])
        
        return list(roles) if roles else ['api_user']

# Usage
ldap_auth = LDAPAuthenticator({
    'ldap_server': 'ldap://ldap.hospital.org:389',
    'base_dn': 'dc=hospital,dc=org',
    'user_search_base': 'ou=users,dc=hospital,dc=org',
    'group_search_base': 'ou=groups,dc=hospital,dc=org'
})

user = ldap_auth.authenticate('jsmith', 'password123')
if user:
    print(f"User: {user['username']}, Roles: {user['roles']}")
```

### LDAP Integration in Flask

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login endpoint with LDAP authentication"""
    
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    # Authenticate with LDAP
    user = ldap_auth.authenticate(username, password)
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Generate JWT token
    token = auth_manager.generate_token(user)
    
    # Log authentication
    security_logger.info(f"User logged in: {username}")
    
    return jsonify({
        'token': token,
        'user': {
            'username': user['username'],
            'roles': user['roles']
        }
    }), 200
```

## OIDC/SAML SSO Integration

### OpenID Connect (OIDC) Configuration

```python
from src.aimedres.security.oidc_auth import OIDCAuthenticator

# Initialize OIDC authenticator
oidc_auth = OIDCAuthenticator({
    'issuer': 'https://sso.hospital.org',
    'client_id': os.getenv('OIDC_CLIENT_ID'),
    'client_secret': os.getenv('OIDC_CLIENT_SECRET'),
    'redirect_uri': 'https://aimedres.hospital.org/auth/callback'
})

@app.route('/auth/login')
def oidc_login():
    """Redirect to OIDC provider for authentication"""
    auth_url = oidc_auth.get_authorization_url()
    return redirect(auth_url)

@app.route('/auth/callback')
def oidc_callback():
    """Handle OIDC callback"""
    code = request.args.get('code')
    
    if not code:
        return jsonify({'error': 'No authorization code'}), 400
    
    # Exchange code for token
    user_info = oidc_auth.authenticate(code)
    
    if not user_info:
        return jsonify({'error': 'Authentication failed'}), 401
    
    # Map OIDC claims to roles
    roles = map_oidc_claims_to_roles(user_info)
    
    # Generate internal JWT
    token = auth_manager.generate_token({
        'username': user_info['email'],
        'roles': roles
    })
    
    return jsonify({'token': token})

def map_oidc_claims_to_roles(user_info: Dict) -> list:
    """Map OIDC claims to AiMedRes roles"""
    
    roles = []
    
    # Check groups claim
    groups = user_info.get('groups', [])
    
    if 'admins' in groups:
        roles.append('admin')
    if 'clinicians' in groups:
        roles.append('clinician')
    if 'researchers' in groups:
        roles.append('researcher')
    
    return roles if roles else ['api_user']
```

## Role-Based Access Control (RBAC)

### Permission Checking

```python
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = request.user
            
            if not user:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Check if user has permission
            if not has_permission(user, permission):
                return jsonify({
                    'error': 'Insufficient permissions',
                    'required': permission
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def has_permission(user: Dict, required_permission: str) -> bool:
    """Check if user has required permission"""
    
    user_roles = user.get('roles', [])
    
    for role in user_roles:
        role_permissions = ROLES.get(role, {}).get('permissions', [])
        
        for permission in role_permissions:
            # Check for exact match
            if permission == required_permission:
                return True
            
            # Check for wildcard match (e.g., 'system:*' matches 'system:read')
            if permission.endswith(':*'):
                prefix = permission[:-1]  # Remove '*'
                if required_permission.startswith(prefix):
                    return True
    
    return False

# Protected endpoint with permission check
@app.route('/api/admin/users', methods=['GET'])
@require_api_key
@require_permission('user:read')
def list_users():
    users = get_all_users()
    return jsonify(users)

@app.route('/api/admin/users', methods=['POST'])
@require_api_key
@require_permission('user:create')
def create_user_endpoint():
    data = request.get_json()
    user = create_user(data['username'], data['password'], data['roles'])
    return jsonify(user), 201
```

### Dynamic Permission System

```python
class PermissionManager:
    """Dynamic permission management"""
    
    def __init__(self):
        self.permissions = {}
    
    def grant_permission(self, user_id: str, permission: str):
        """Grant permission to user"""
        if user_id not in self.permissions:
            self.permissions[user_id] = set()
        
        self.permissions[user_id].add(permission)
        audit_log('PERMISSION_GRANTED', user_id, permission)
    
    def revoke_permission(self, user_id: str, permission: str):
        """Revoke permission from user"""
        if user_id in self.permissions:
            self.permissions[user_id].discard(permission)
            audit_log('PERMISSION_REVOKED', user_id, permission)
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission"""
        user_perms = self.permissions.get(user_id, set())
        return permission in user_perms

# Usage
perm_manager = PermissionManager()
perm_manager.grant_permission('user123', 'patient:read')
```

## Audit Logging

### Audit Log Configuration

```python
import logging
from datetime import datetime
import json

# Configure audit logger
audit_logger = logging.getLogger('aimedres.audit')
audit_logger.setLevel(logging.INFO)

# File handler for audit log
audit_handler = logging.FileHandler('/var/log/aimedres/audit.log')
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(message)s'
))
audit_logger.addHandler(audit_handler)

def audit_log(event_type: str, user: str, resource: str, 
              action: str, result: str, details: Dict = None):
    """Log audit event"""
    
    event = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'user': user,
        'resource': resource,
        'action': action,
        'result': result,
        'details': details or {},
        'ip_address': request.remote_addr if request else None
    }
    
    audit_logger.info(json.dumps(event))

# Usage examples
audit_log('DATA_ACCESS', 'dr_smith', 'patient/12345', 'read', 'success')
audit_log('MODEL_TRAINING', 'researcher1', 'model/alzheimer', 'train', 'success')
audit_log('CONFIG_CHANGE', 'admin', 'system/config', 'update', 'success')
audit_log('LOGIN_ATTEMPT', 'unknown', 'auth/login', 'login', 'failed', 
          {'reason': 'invalid_credentials'})
```

### Audit Middleware

```python
from flask import Flask, request, g
import time

def audit_middleware():
    """Middleware to audit all API requests"""
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.user = getattr(request, 'user', None)
    
    @app.after_request
    def after_request(response):
        # Calculate request duration
        duration = time.time() - g.start_time
        
        # Get user info
        user = g.user['username'] if g.user else 'anonymous'
        
        # Log request
        audit_log(
            event_type='API_REQUEST',
            user=user,
            resource=request.path,
            action=request.method,
            result='success' if response.status_code < 400 else 'failed',
            details={
                'status_code': response.status_code,
                'duration_ms': round(duration * 1000, 2),
                'user_agent': request.headers.get('User-Agent')
            }
        )
        
        return response

# Initialize middleware
audit_middleware()
```

### Audit Query and Reporting

```python
import json
from datetime import datetime, timedelta

def query_audit_logs(user: str = None, event_type: str = None, 
                     start_date: datetime = None, end_date: datetime = None):
    """Query audit logs"""
    
    results = []
    
    with open('/var/log/aimedres/audit.log', 'r') as f:
        for line in f:
            try:
                # Parse timestamp and event
                parts = line.split(' - ', 1)
                if len(parts) != 2:
                    continue
                
                timestamp_str = parts[0]
                event_json = parts[1]
                
                event = json.loads(event_json)
                event_timestamp = datetime.fromisoformat(event['timestamp'])
                
                # Apply filters
                if user and event.get('user') != user:
                    continue
                
                if event_type and event.get('event_type') != event_type:
                    continue
                
                if start_date and event_timestamp < start_date:
                    continue
                
                if end_date and event_timestamp > end_date:
                    continue
                
                results.append(event)
                
            except Exception as e:
                print(f"Error parsing log line: {e}")
                continue
    
    return results

# Usage
# Get all failed login attempts in last 24 hours
failed_logins = query_audit_logs(
    event_type='LOGIN_ATTEMPT',
    start_date=datetime.now() - timedelta(days=1)
)
failed_logins = [l for l in failed_logins if l['result'] == 'failed']

print(f"Failed login attempts: {len(failed_logins)}")
```

## Environment Configuration

Add to `.env` file:

```bash
# Authentication Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here
TOKEN_EXPIRY_HOURS=24
MAX_FAILED_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15

# LDAP Configuration (if using)
LDAP_ENABLED=true
LDAP_SERVER=ldap://ldap.hospital.org:389
LDAP_BASE_DN=dc=hospital,dc=org
LDAP_USER_SEARCH_BASE=ou=users,dc=hospital,dc=org
LDAP_GROUP_SEARCH_BASE=ou=groups,dc=hospital,dc=org

# OIDC Configuration (if using)
OIDC_ENABLED=false
OIDC_ISSUER=https://sso.hospital.org
OIDC_CLIENT_ID=your_client_id
OIDC_CLIENT_SECRET=your_client_secret
OIDC_REDIRECT_URI=https://aimedres.hospital.org/auth/callback

# Audit Logging
AUDIT_ENABLED=true
AUDIT_LOG_PATH=/var/log/aimedres/audit.log
AUDIT_LOG_LEVEL=INFO
```

## Testing Authentication

### Authentication Test Suite

```python
import unittest

class TestAuthentication(unittest.TestCase):
    """Test authentication and authorization"""
    
    def setUp(self):
        self.auth_manager = SecureAuthManager({
            'jwt_secret': 'test_secret',
            'token_expiry_hours': 24
        })
    
    def test_api_key_authentication(self):
        """Test API key authentication"""
        api_key = self.auth_manager._generate_api_key('testuser', ['clinician'])
        user = self.auth_manager.validate_api_key(api_key)
        
        self.assertIsNotNone(user)
        self.assertEqual(user['user_id'], 'testuser')
        self.assertIn('clinician', user['roles'])
    
    def test_permission_check(self):
        """Test permission checking"""
        user = {'username': 'testuser', 'roles': ['clinician']}
        
        # Should have permission
        self.assertTrue(has_permission(user, 'patient:read'))
        
        # Should not have permission
        self.assertFalse(has_permission(user, 'system:config'))
    
    def test_password_validation(self):
        """Test password policy"""
        # Strong password
        valid, msg = validate_password('MySecure$Pass123')
        self.assertTrue(valid)
        
        # Weak password
        valid, msg = validate_password('weak')
        self.assertFalse(valid)
    
    def test_audit_logging(self):
        """Test audit logging"""
        audit_log('TEST_EVENT', 'testuser', 'test/resource', 'read', 'success')
        
        # Verify log was written
        logs = query_audit_logs(user='testuser', event_type='TEST_EVENT')
        self.assertGreater(len(logs), 0)

if __name__ == '__main__':
    unittest.main()
```

## Security Checklist

- [ ] Strong password policy enforced (12+ chars, mixed case, numbers, symbols)
- [ ] Multi-factor authentication available (optional)
- [ ] API keys use cryptographically secure generation
- [ ] JWT tokens have appropriate expiry (24 hours recommended)
- [ ] LDAP/AD integration configured (if applicable)
- [ ] SSO integration configured (if applicable)
- [ ] Role-based access control implemented
- [ ] All roles and permissions documented
- [ ] Audit logging enabled for all access
- [ ] Audit logs include: user, timestamp, resource, action, result
- [ ] Failed login attempts tracked and limited
- [ ] Account lockout after failed attempts
- [ ] Session timeout configured
- [ ] Secure credential storage (hashed passwords, encrypted keys)
- [ ] Regular access review process established

## Compliance Verification

```python
def verify_auth_compliance():
    """Verify authentication/authorization compliance"""
    
    checks = {
        'Password Policy': check_password_policy(),
        'API Key Security': check_api_key_security(),
        'Role-Based Access': check_rbac_configured(),
        'Audit Logging': check_audit_logging(),
        'Session Management': check_session_management(),
    }
    
    print("\n📋 Authentication/Authorization Compliance Report")
    print("=" * 60)
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n✅ All authentication checks passed")
    else:
        print("\n❌ Some authentication checks failed")
    
    return all_passed

if __name__ == '__main__':
    verify_auth_compliance()
```

## References

- NIST SP 800-63B: [Digital Identity Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)
- HIPAA Security Rule: [Access Control Standards](https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html)
- OWASP Authentication Cheat Sheet: [OWASP.org](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
