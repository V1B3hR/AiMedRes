"""
Advanced Security and Safety Testing
Comprehensive security validation, penetration testing, and compliance verification
"""

import pytest
import hashlib
import hmac
import time
import threading
import sys
import os
import tempfile
import json
import re
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import requests
from unittest.mock import Mock, patch, MagicMock
import sqlite3
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aimedres.clinical.decision_support import RiskStratificationEngine
# from regulatory_compliance import RegulatoryCompliance, HIPAAValidator, ValidationProtocol


class TestDataSecurity:
    """Test data security and encryption functionality"""
    
    def test_data_encryption_decryption(self):
        """Test data encryption and decryption processes"""
        from cryptography.fernet import Fernet
        
        # Generate key
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Test data
        sensitive_data = "Patient ID: 12345, Diagnosis: Alzheimer's"
        
        # Encrypt
        encrypted_data = cipher.encrypt(sensitive_data.encode())
        assert encrypted_data != sensitive_data.encode()
        
        # Decrypt
        decrypted_data = cipher.decrypt(encrypted_data).decode()
        assert decrypted_data == sensitive_data

    def test_password_hashing_security(self):
        """Test secure password hashing"""
        import bcrypt
        
        password = "SecurePassword123!"
        
        # Hash password
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Verify password
        assert bcrypt.checkpw(password.encode('utf-8'), hashed)
        assert not bcrypt.checkpw("WrongPassword".encode('utf-8'), hashed)

    def test_secure_random_generation(self):
        """Test cryptographically secure random number generation"""
        import secrets
        
        # Generate secure random tokens
        token1 = secrets.token_hex(32)
        token2 = secrets.token_hex(32)
        
        assert len(token1) == 64  # 32 bytes = 64 hex chars
        assert len(token2) == 64
        assert token1 != token2  # Should be different

    def test_data_masking_and_anonymization(self):
        """Test data masking for sensitive information"""
        def mask_patient_id(patient_id: str) -> str:
            """Mask patient ID for logging"""
            if len(patient_id) <= 4:
                return '*' * len(patient_id)
            return patient_id[:2] + '*' * (len(patient_id) - 4) + patient_id[-2:]
        
        def anonymize_age(age: int) -> str:
            """Anonymize age to age ranges"""
            if age < 18:
                return "0-17"
            elif age < 65:
                return "18-64"
            else:
                return "65+"
        
        # Test masking
        assert mask_patient_id("12345678") == "12****78"
        assert mask_patient_id("123") == "***"
        
        # Test anonymization
        assert anonymize_age(25) == "18-64"
        assert anonymize_age(70) == "65+"
        assert anonymize_age(15) == "0-17"

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention measures"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute('''
                CREATE TABLE patients (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    diagnosis TEXT
                )
            ''')
            
            # Insert test data
            cursor.execute(
                "INSERT INTO patients (name, diagnosis) VALUES (?, ?)",
                ("John Doe", "Healthy")
            )
            
            # Test parameterized query (safe)
            safe_query = "SELECT * FROM patients WHERE name = ?"
            cursor.execute(safe_query, ("John Doe",))
            results = cursor.fetchall()
            assert len(results) == 1
            
            # Test that malicious input doesn't work with parameterized queries
            malicious_input = "'; DROP TABLE patients; --"
            cursor.execute(safe_query, (malicious_input,))
            results = cursor.fetchall()
            assert len(results) == 0  # No results, but table still exists
            
            # Verify table still exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
            assert len(cursor.fetchall()) == 1
            
            conn.close()
        finally:
            os.unlink(db_path)

    def test_input_sanitization(self):
        """Test input sanitization and validation"""
        def sanitize_input(user_input: str) -> str:
            """Sanitize user input"""
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
            sanitized = user_input
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            return sanitized.strip()
        
        def validate_medical_id(medical_id: str) -> bool:
            """Validate medical ID format"""
            # Should be alphanumeric, 8-12 characters
            return bool(re.match(r'^[A-Za-z0-9]{8,12}$', medical_id))
        
        # Test sanitization
        malicious_input = "<script>alert('xss')</script>"
        sanitized = sanitize_input(malicious_input)
        assert '<' not in sanitized
        assert '>' not in sanitized
        assert 'script' in sanitized  # Content remains, but tags removed
        
        # Test validation
        assert validate_medical_id("ABC12345") is True
        assert validate_medical_id("123456789ABC") is True
        assert validate_medical_id("ABC") is False  # Too short
        assert validate_medical_id("ABC@123") is False  # Invalid character


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization mechanisms"""
    
    def test_token_based_authentication(self):
        """Test JWT-like token authentication"""
        import jwt
        import datetime
        
        secret_key = "test_secret_key_12345"
        
        # Create token
        payload = {
            'user_id': 'doctor123',
            'role': 'physician',
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }
        
        token = jwt.encode(payload, secret_key, algorithm='HS256')
        
        # Verify token
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        assert decoded['user_id'] == 'doctor123'
        assert decoded['role'] == 'physician'

    def test_role_based_access_control(self):
        """Test role-based access control (RBAC)"""
        class AccessController:
            def __init__(self):
                self.permissions = {
                    'admin': ['read', 'write', 'delete', 'admin'],
                    'physician': ['read', 'write'],
                    'nurse': ['read', 'write_limited'],
                    'patient': ['read_own']
                }
            
            def has_permission(self, role: str, action: str) -> bool:
                return action in self.permissions.get(role, [])
        
        controller = AccessController()
        
        # Test different role permissions
        assert controller.has_permission('admin', 'delete') is True
        assert controller.has_permission('physician', 'read') is True
        assert controller.has_permission('physician', 'delete') is False
        assert controller.has_permission('nurse', 'admin') is False
        assert controller.has_permission('patient', 'read_own') is True

    def test_session_management(self):
        """Test secure session management"""
        class SessionManager:
            def __init__(self):
                self.sessions = {}
                self.session_timeout = 3600  # 1 hour
            
            def create_session(self, user_id: str) -> str:
                import secrets
                session_id = secrets.token_urlsafe(32)
                self.sessions[session_id] = {
                    'user_id': user_id,
                    'created_at': time.time(),
                    'last_activity': time.time()
                }
                return session_id
            
            def validate_session(self, session_id: str) -> bool:
                if session_id not in self.sessions:
                    return False
                
                session = self.sessions[session_id]
                current_time = time.time()
                
                # Check timeout
                if current_time - session['last_activity'] > self.session_timeout:
                    del self.sessions[session_id]
                    return False
                
                # Update last activity
                session['last_activity'] = current_time
                return True
            
            def invalidate_session(self, session_id: str):
                if session_id in self.sessions:
                    del self.sessions[session_id]
        
        manager = SessionManager()
        
        # Create session
        session_id = manager.create_session('user123')
        assert manager.validate_session(session_id) is True
        
        # Invalidate session
        manager.invalidate_session(session_id)
        assert manager.validate_session(session_id) is False

    def test_multi_factor_authentication(self):
        """Test multi-factor authentication simulation"""
        class MFAValidator:
            def __init__(self):
                self.totp_secret = "JBSWY3DPEHPK3PXP"  # Base32 encoded secret
            
            def generate_totp(self, secret: str, timestamp: int = None) -> str:
                """Generate TOTP code"""
                if timestamp is None:
                    timestamp = int(time.time())
                
                # Simple TOTP implementation for testing
                time_step = timestamp // 30
                code = hashlib.sha1(f"{secret}{time_step}".encode()).hexdigest()[:6]
                return code
            
            def validate_totp(self, provided_code: str, secret: str) -> bool:
                """Validate TOTP code with time window"""
                current_time = int(time.time())
                
                # Check current time step and adjacent ones for clock drift
                for time_offset in [-1, 0, 1]:
                    timestamp = current_time + (time_offset * 30)
                    expected_code = self.generate_totp(secret, timestamp)
                    if provided_code == expected_code:
                        return True
                return False
        
        mfa = MFAValidator()
        
        # Generate and validate TOTP
        code = mfa.generate_totp(mfa.totp_secret)
        assert mfa.validate_totp(code, mfa.totp_secret) is True
        assert mfa.validate_totp("000000", mfa.totp_secret) is False


class TestAPISecurityTesting:
    """Test API security measures"""
    
    def test_rate_limiting(self):
        """Test API rate limiting functionality"""
        class RateLimiter:
            def __init__(self, max_requests: int = 100, time_window: int = 3600):
                self.max_requests = max_requests
                self.time_window = time_window
                self.requests = {}
            
            def is_allowed(self, client_id: str) -> bool:
                current_time = time.time()
                
                if client_id not in self.requests:
                    self.requests[client_id] = []
                
                # Clean old requests
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id]
                    if current_time - req_time < self.time_window
                ]
                
                # Check if under limit
                if len(self.requests[client_id]) >= self.max_requests:
                    return False
                
                # Record request
                self.requests[client_id].append(current_time)
                return True
        
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        # Test normal usage
        for _ in range(5):
            assert limiter.is_allowed("client1") is True
        
        # Test rate limit exceeded
        assert limiter.is_allowed("client1") is False
        
        # Test different client
        assert limiter.is_allowed("client2") is True

    def test_cors_security(self):
        """Test CORS security configuration"""
        def validate_cors_origin(origin: str, allowed_origins: List[str]) -> bool:
            """Validate CORS origin"""
            if '*' in allowed_origins:
                return True  # Allow all (insecure for production)
            return origin in allowed_origins
        
        allowed_origins = [
            'https://medical-app.example.com',
            'https://dashboard.medical.org'
        ]
        
        # Test valid origins
        assert validate_cors_origin('https://medical-app.example.com', allowed_origins) is True
        assert validate_cors_origin('https://dashboard.medical.org', allowed_origins) is True
        
        # Test invalid origins
        assert validate_cors_origin('https://malicious-site.com', allowed_origins) is False
        assert validate_cors_origin('http://medical-app.example.com', allowed_origins) is False  # HTTP not HTTPS

    def test_input_validation_api(self):
        """Test API input validation"""
        class APIValidator:
            @staticmethod
            def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
                """Validate patient data input"""
                errors = []
                
                # Required fields
                required_fields = ['patient_id', 'age', 'gender']
                for field in required_fields:
                    if field not in data:
                        errors.append(f"Missing required field: {field}")
                
                # Age validation
                if 'age' in data:
                    age = data['age']
                    if not isinstance(age, int) or age < 0 or age > 150:
                        errors.append("Age must be an integer between 0 and 150")
                
                # Gender validation
                if 'gender' in data:
                    if data['gender'] not in ['M', 'F', 'Other']:
                        errors.append("Gender must be 'M', 'F', or 'Other'")
                
                # Patient ID validation
                if 'patient_id' in data:
                    if not re.match(r'^[A-Za-z0-9]{6,12}$', str(data['patient_id'])):
                        errors.append("Patient ID must be 6-12 alphanumeric characters")
                
                return len(errors) == 0, errors
        
        validator = APIValidator()
        
        # Test valid data
        valid_data = {
            'patient_id': 'ABC123456',
            'age': 45,
            'gender': 'M'
        }
        is_valid, errors = validator.validate_patient_data(valid_data)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid data
        invalid_data = {
            'patient_id': '123',  # Too short
            'age': 200,  # Invalid age
            'gender': 'X'  # Invalid gender
        }
        is_valid, errors = validator.validate_patient_data(invalid_data)
        assert is_valid is False
        assert len(errors) > 0


class TestPrivacyAndCompliance:
    """Test privacy protection and regulatory compliance"""
    
    def test_hipaa_compliance_logging(self):
        """Test HIPAA compliance audit logging"""
        class HIPAAAuditLogger:
            def __init__(self):
                self.audit_log = []
            
            def log_phi_access(self, user_id: str, patient_id: str, action: str, 
                             purpose: str, timestamp: float = None):
                """Log PHI access for HIPAA compliance"""
                if timestamp is None:
                    timestamp = time.time()
                
                log_entry = {
                    'timestamp': timestamp,
                    'user_id': user_id,
                    'patient_id': self._hash_patient_id(patient_id),  # Hash for security
                    'action': action,
                    'purpose': purpose,
                    'ip_address': '192.168.1.100',  # Mock IP
                    'session_id': 'session_123'
                }
                self.audit_log.append(log_entry)
            
            def _hash_patient_id(self, patient_id: str) -> str:
                """Hash patient ID for audit logs"""
                return hashlib.sha256(patient_id.encode()).hexdigest()[:16]
            
            def get_audit_trail(self, patient_id: str) -> List[Dict]:
                """Get audit trail for specific patient"""
                hashed_id = self._hash_patient_id(patient_id)
                return [log for log in self.audit_log if log['patient_id'] == hashed_id]
        
        logger = HIPAAAuditLogger()
        
        # Log some accesses
        logger.log_phi_access('doctor123', 'patient456', 'VIEW', 'Treatment')
        logger.log_phi_access('nurse789', 'patient456', 'UPDATE', 'Care coordination')
        
        # Check audit trail
        trail = logger.get_audit_trail('patient456')
        assert len(trail) == 2
        assert trail[0]['action'] == 'VIEW'
        assert trail[1]['action'] == 'UPDATE'

    def test_data_retention_policy(self):
        """Test data retention policy implementation"""
        class DataRetentionManager:
            def __init__(self):
                self.retention_periods = {
                    'patient_records': 7 * 365 * 24 * 3600,  # 7 years
                    'audit_logs': 6 * 365 * 24 * 3600,  # 6 years
                    'temporary_data': 30 * 24 * 3600  # 30 days
                }
            
            def should_delete(self, data_type: str, creation_time: float) -> bool:
                """Check if data should be deleted based on retention policy"""
                if data_type not in self.retention_periods:
                    return False
                
                retention_period = self.retention_periods[data_type]
                current_time = time.time()
                
                return (current_time - creation_time) > retention_period
            
            def cleanup_expired_data(self, data_store: Dict[str, Dict]) -> int:
                """Clean up expired data and return count of deleted items"""
                deleted_count = 0
                keys_to_delete = []
                
                for key, data in data_store.items():
                    if self.should_delete(data['type'], data['created_at']):
                        keys_to_delete.append(key)
                        deleted_count += 1
                
                for key in keys_to_delete:
                    del data_store[key]
                
                return deleted_count
        
        manager = DataRetentionManager()
        
        # Test retention decision
        current_time = time.time()
        old_time = current_time - (8 * 365 * 24 * 3600)  # 8 years ago
        
        assert manager.should_delete('patient_records', old_time) is True
        assert manager.should_delete('patient_records', current_time) is False

    def test_data_minimization(self):
        """Test data minimization principles"""
        class DataMinimizer:
            @staticmethod
            def extract_necessary_fields(patient_data: Dict[str, Any], 
                                       purpose: str) -> Dict[str, Any]:
                """Extract only necessary fields based on purpose"""
                field_mappings = {
                    'diagnosis': ['age', 'gender', 'symptoms', 'medical_history'],
                    'billing': ['patient_id', 'insurance_info', 'services'],
                    'research': ['age_range', 'gender', 'diagnosis'],  # Anonymized
                    'emergency': ['patient_id', 'allergies', 'emergency_contacts']
                }
                
                if purpose not in field_mappings:
                    return {}
                
                necessary_fields = field_mappings[purpose]
                minimized_data = {}
                
                for field in necessary_fields:
                    if field in patient_data:
                        # Special handling for research data
                        if purpose == 'research' and field == 'age_range':
                            age = patient_data.get('age', 0)
                            minimized_data[field] = DataMinimizer._anonymize_age(age)
                        else:
                            minimized_data[field] = patient_data[field]
                
                return minimized_data
            
            @staticmethod
            def _anonymize_age(age: int) -> str:
                """Convert age to age range for anonymization"""
                if age < 18:
                    return "0-17"
                elif age < 65:
                    return "18-64"
                else:
                    return "65+"
        
        patient_data = {
            'patient_id': 'P12345',
            'name': 'John Doe',
            'age': 45,
            'gender': 'M',
            'symptoms': ['headache', 'fatigue'],
            'medical_history': ['hypertension'],
            'insurance_info': 'Blue Cross',
            'allergies': ['penicillin'],
            'emergency_contacts': ['Jane Doe: 555-1234']
        }
        
        # Test different purposes
        diagnosis_data = DataMinimizer.extract_necessary_fields(patient_data, 'diagnosis')
        research_data = DataMinimizer.extract_necessary_fields(patient_data, 'research')
        emergency_data = DataMinimizer.extract_necessary_fields(patient_data, 'emergency')
        
        # Verify data minimization
        assert 'name' not in diagnosis_data  # Not needed for diagnosis
        assert 'age_range' in research_data  # Anonymized age
        assert 'patient_id' in emergency_data  # Needed for emergency
        assert len(research_data) <= len(diagnosis_data)  # Research has less data


class TestSecurityMonitoring:
    """Test security monitoring and threat detection"""
    
    def test_intrusion_detection(self):
        """Test intrusion detection system"""
        class IntrusionDetector:
            def __init__(self):
                self.failed_attempts = {}
                self.max_failures = 5
                self.lockout_duration = 300  # 5 minutes
            
            def record_failed_login(self, user_id: str, ip_address: str):
                """Record failed login attempt"""
                key = f"{user_id}:{ip_address}"
                current_time = time.time()
                
                if key not in self.failed_attempts:
                    self.failed_attempts[key] = []
                
                self.failed_attempts[key].append(current_time)
                
                # Clean old attempts
                cutoff_time = current_time - self.lockout_duration
                self.failed_attempts[key] = [
                    attempt for attempt in self.failed_attempts[key]
                    if attempt > cutoff_time
                ]
            
            def is_locked_out(self, user_id: str, ip_address: str) -> bool:
                """Check if user/IP is locked out"""
                key = f"{user_id}:{ip_address}"
                if key not in self.failed_attempts:
                    return False
                
                return len(self.failed_attempts[key]) >= self.max_failures
            
            def detect_brute_force(self, user_id: str, ip_address: str) -> bool:
                """Detect potential brute force attack"""
                return self.is_locked_out(user_id, ip_address)
        
        detector = IntrusionDetector()
        
        # Simulate failed login attempts
        for _ in range(4):
            detector.record_failed_login('user123', '192.168.1.100')
            assert detector.is_locked_out('user123', '192.168.1.100') is False
        
        # One more failure should trigger lockout
        detector.record_failed_login('user123', '192.168.1.100')
        assert detector.is_locked_out('user123', '192.168.1.100') is True

    def test_anomaly_detection(self):
        """Test anomaly detection in user behavior"""
        class AnomalyDetector:
            def __init__(self):
                self.user_patterns = {}
            
            def record_activity(self, user_id: str, action: str, timestamp: float = None):
                """Record user activity"""
                if timestamp is None:
                    timestamp = time.time()
                
                if user_id not in self.user_patterns:
                    self.user_patterns[user_id] = {
                        'actions': [],
                        'login_times': [],
                        'ip_addresses': set()
                    }
                
                self.user_patterns[user_id]['actions'].append({
                    'action': action,
                    'timestamp': timestamp
                })
            
            def detect_unusual_activity(self, user_id: str, action: str) -> bool:
                """Detect unusual activity patterns"""
                if user_id not in self.user_patterns:
                    return False
                
                user_data = self.user_patterns[user_id]
                recent_actions = [
                    act for act in user_data['actions']
                    if time.time() - act['timestamp'] < 3600  # Last hour
                ]
                
                # Check for unusual frequency
                action_count = sum(1 for act in recent_actions if act['action'] == action)
                if action_count > 50:  # More than 50 of same action in an hour
                    return True
                
                return False
        
        detector = AnomalyDetector()
        
        # Normal activity
        for i in range(10):
            detector.record_activity('user123', 'view_patient')
        
        assert detector.detect_unusual_activity('user123', 'view_patient') is False
        
        # Unusual activity
        for i in range(45):
            detector.record_activity('user123', 'download_record')
        
        assert detector.detect_unusual_activity('user123', 'download_record') is True

    def test_security_alerting(self):
        """Test security alert system"""
        class SecurityAlerter:
            def __init__(self):
                self.alerts = []
                self.alert_thresholds = {
                    'failed_login': 5,
                    'data_access_anomaly': 3,
                    'privilege_escalation': 1
                }
            
            def check_and_alert(self, event_type: str, severity: str, details: Dict[str, Any]):
                """Check if alert should be triggered"""
                alert = {
                    'timestamp': time.time(),
                    'event_type': event_type,
                    'severity': severity,
                    'details': details,
                    'alert_id': len(self.alerts) + 1
                }
                
                # Critical events always trigger alerts
                if severity == 'critical':
                    self.alerts.append(alert)
                    return True
                
                # Count recent similar events
                recent_events = [
                    a for a in self.alerts
                    if (time.time() - a['timestamp'] < 300 and  # Last 5 minutes
                        a['event_type'] == event_type)
                ]
                
                threshold = self.alert_thresholds.get(event_type, 1)
                if len(recent_events) >= threshold:
                    self.alerts.append(alert)
                    return True
                
                return False
            
            def get_alerts(self, severity: str = None) -> List[Dict]:
                """Get alerts, optionally filtered by severity"""
                if severity is None:
                    return self.alerts
                return [alert for alert in self.alerts if alert['severity'] == severity]
        
        alerter = SecurityAlerter()
        
        # Test critical alert
        critical_triggered = alerter.check_and_alert(
            'privilege_escalation', 
            'critical',
            {'user_id': 'user123', 'attempted_role': 'admin'}
        )
        assert critical_triggered is True
        
        # Test threshold-based alerting
        for i in range(3):
            alerter.check_and_alert(
                'failed_login',
                'medium',
                {'user_id': 'user456', 'ip': '192.168.1.100'}
            )
        
        medium_alerts = alerter.get_alerts('medium')
        assert len(medium_alerts) >= 1


@pytest.mark.slow
class TestPenetrationTesting:
    """Simulated penetration testing scenarios"""
    
    def test_xss_prevention(self):
        """Test Cross-Site Scripting (XSS) prevention"""
        def sanitize_html_input(user_input: str) -> str:
            """Sanitize HTML input to prevent XSS"""
            import html
            return html.escape(user_input)
        
        def validate_no_script_tags(content: str) -> bool:
            """Validate that content contains no script tags"""
            return '<script' not in content.lower() and 'javascript:' not in content.lower()
        
        # Test XSS attempts
        xss_attempts = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
        ]
        
        for attempt in xss_attempts:
            sanitized = sanitize_html_input(attempt)
            assert validate_no_script_tags(sanitized) is True
            assert '<script>' not in sanitized
            assert 'javascript:' not in sanitized

    def test_csrf_protection(self):
        """Test Cross-Site Request Forgery (CSRF) protection"""
        import secrets
        
        class CSRFProtection:
            def __init__(self):
                self.tokens = {}
            
            def generate_token(self, session_id: str) -> str:
                """Generate CSRF token for session"""
                token = secrets.token_urlsafe(32)
                self.tokens[session_id] = token
                return token
            
            def validate_token(self, session_id: str, provided_token: str) -> bool:
                """Validate CSRF token"""
                expected_token = self.tokens.get(session_id)
                return expected_token == provided_token
        
        csrf = CSRFProtection()
        
        # Generate token
        session_id = "session_123"
        token = csrf.generate_token(session_id)
        
        # Valid token should pass
        assert csrf.validate_token(session_id, token) is True
        
        # Invalid token should fail
        assert csrf.validate_token(session_id, "invalid_token") is False
        assert csrf.validate_token("wrong_session", token) is False

    def test_directory_traversal_prevention(self):
        """Test directory traversal attack prevention"""
        import os
        from pathlib import Path
        
        def safe_file_access(base_path: str, user_path: str) -> bool:
            """Safely validate file access within base directory"""
            try:
                base = Path(base_path).resolve()
                target = (base / user_path).resolve()
                
                # Check if target is within base directory
                return str(target).startswith(str(base))
            except (OSError, ValueError):
                return False
        
        base_directory = "/var/medical_files"
        
        # Valid paths
        assert safe_file_access(base_directory, "patient_123.pdf") is True
        assert safe_file_access(base_directory, "reports/analysis.txt") is True
        
        # Directory traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "....//....//etc/shadow",
        ]
        
        for attempt in traversal_attempts:
            assert safe_file_access(base_directory, attempt) is False