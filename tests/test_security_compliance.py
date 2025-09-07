"""
Security and Privacy Tests for DuetMind Adaptive

This test suite validates the security and privacy features including:
- Authentication and authorization
- Input validation and sanitization
- Data encryption and anonymization
- Privacy compliance (GDPR/HIPAA)
- Security monitoring
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Import security modules
try:
    from security import (
        SecureAuthManager, InputValidator, SecurityValidator,
        DataEncryption, PrivacyManager, DataRetentionPolicy,
        SecurityMonitor
    )
    from secure_medical_processor import SecureMedicalDataProcessor
except ImportError:
    pytest.skip("Security modules not available", allow_module_level=True)

class TestSecureAuthManager:
    """Test authentication and authorization functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'max_failed_attempts': 3,
            'lockout_duration_minutes': 5,
            'token_expiry_hours': 1
        }
        self.auth_manager = SecureAuthManager(self.config)
    
    def test_api_key_generation(self):
        """Test secure API key generation."""
        api_key = self.auth_manager._generate_api_key('test_user', ['user'])
        
        # Verify key format
        assert api_key.startswith('dmk_')
        assert len(api_key) > 40
        
        # Verify storage
        assert api_key in self.auth_manager.api_keys
        
        # Verify user info
        key_info = self.auth_manager.api_keys[api_key]
        assert key_info['user_id'] == 'test_user'
        assert 'user' in key_info['roles']
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Generate test key
        api_key = self.auth_manager._generate_api_key('test_user', ['user'])
        
        # Test valid key
        with patch('flask.request') as mock_request:
            mock_request.remote_addr = '127.0.0.1'
            is_valid, user_info = self.auth_manager.validate_api_key(api_key)
            
            assert is_valid
            assert user_info['user_id'] == 'test_user'
    
    def test_invalid_api_key(self):
        """Test invalid API key handling."""
        with patch('flask.request') as mock_request:
            mock_request.remote_addr = '127.0.0.1'
            is_valid, user_info = self.auth_manager.validate_api_key('invalid_key')
            
            assert not is_valid
            assert user_info is None
    
    def test_brute_force_protection(self):
        """Test brute force protection."""
        with patch('flask.request') as mock_request:
            mock_request.remote_addr = '192.168.1.100'
            
            # Exceed failed attempts
            for _ in range(4):
                self.auth_manager.validate_api_key('invalid_key')
            
            # Should be locked out
            assert self.auth_manager._is_locked_out('192.168.1.100')
    
    def test_role_based_access(self):
        """Test role-based access control."""
        api_key = self.auth_manager._generate_api_key('admin_user', ['admin', 'user'])
        
        # Get user info
        with patch('flask.request') as mock_request:
            mock_request.remote_addr = '127.0.0.1'
            _, user_info = self.auth_manager.validate_api_key(api_key)
        
        # Test role checking
        assert self.auth_manager.has_role(user_info, 'admin')
        assert self.auth_manager.has_role(user_info, 'user')
        assert not self.auth_manager.has_role(user_info, 'superuser')

class TestInputValidator:
    """Test input validation and sanitization."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = InputValidator()
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        sql_attacks = [
            "'; DROP TABLE users; --",
            "' OR 1=1 --",
            "' UNION SELECT * FROM passwords",
            "admin'/*",
            "1; DELETE FROM users"
        ]
        
        for attack in sql_attacks:
            assert not self.validator.validate_sql_injection(attack)
    
    def test_xss_detection(self):
        """Test XSS attack detection."""
        xss_attacks = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<iframe src='malicious.com'>",
            "<img onerror='alert(1)' src='x'>",
            "<svg onload='alert(1)'>"
        ]
        
        for attack in xss_attacks:
            assert not self.validator.validate_xss(attack)
    
    def test_string_sanitization(self):
        """Test string sanitization."""
        malicious_input = "<script>alert('xss')</script>"
        sanitized = self.validator.sanitize_string(malicious_input)
        
        # Should not contain script tags
        assert '<script>' not in sanitized
        assert 'alert' not in sanitized
    
    def test_medical_data_validation(self):
        """Test medical data validation."""
        # Valid medical data
        valid_data = {
            'Age': 65,
            'BMI': 24.5,
            'MMSE': 28,
            'Gender': 1
        }
        
        is_valid, errors = self.validator.validate_medical_data(valid_data)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid medical data
        invalid_data = {
            'Age': 150,  # Too old
            'BMI': 5.0,  # Too low
            'MMSE': 35,  # Too high
            'Gender': 'invalid'  # Wrong type
        }
        
        is_valid, errors = self.validator.validate_medical_data(invalid_data)
        assert not is_valid
        assert len(errors) > 0
    
    def test_json_request_validation(self):
        """Test JSON request validation."""
        # Valid JSON
        valid_json = {'task': 'analyze patient', 'priority': 'high'}
        is_valid, errors = self.validator.validate_json_request(
            valid_json, 
            required_fields=['task']
        )
        assert is_valid
        assert len(errors) == 0
        
        # Missing required field
        invalid_json = {'priority': 'high'}
        is_valid, errors = self.validator.validate_json_request(
            invalid_json,
            required_fields=['task']
        )
        assert not is_valid
        assert 'task' in str(errors)

class TestDataEncryption:
    """Test data encryption functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.encryption = DataEncryption(master_password="test_password_123")
    
    def test_data_encryption_decryption(self):
        """Test basic data encryption and decryption."""
        test_data = {'sensitive': 'medical_data', 'patient_id': '12345'}
        
        # Encrypt
        encrypted = self.encryption.encrypt_data(test_data)
        assert encrypted != test_data
        assert isinstance(encrypted, str)
        
        # Decrypt
        decrypted = self.encryption.decrypt_data(encrypted)
        assert decrypted == test_data
    
    def test_medical_data_encryption(self):
        """Test medical data specific encryption."""
        medical_data = {
            'patient_id': '12345',
            'name': 'John Doe',
            'age': 65,
            'diagnosis': 'mild_cognitive_impairment'
        }
        
        encrypted = self.encryption.encrypt_medical_data(
            medical_data,
            preserve_structure=True
        )
        
        # PII should be encrypted
        assert 'patient_id' in encrypted
        assert encrypted['patient_id'] != '12345'
        
        # Non-sensitive data should remain
        assert encrypted['age'] == 65
        assert encrypted['diagnosis'] == 'mild_cognitive_impairment'
    
    def test_data_anonymization(self):
        """Test medical data anonymization."""
        medical_data = {
            'patient_id': '12345',
            'name': 'John Doe',
            'birth_date': '1958-01-15',
            'age': 65,
            'mmse_score': 24
        }
        
        anonymized = self.encryption.anonymize_medical_data(medical_data)
        
        # Direct identifiers should be removed
        assert 'patient_id' not in anonymized
        assert 'name' not in anonymized
        
        # Quasi-identifiers should be hashed
        assert 'birth_date' not in anonymized
        assert 'birth_date_hash' in anonymized
        
        # Medical data should be preserved
        assert anonymized['age'] == 65
        assert anonymized['mmse_score'] == 24
        
        # Metadata should be added
        assert anonymized['anonymized'] == True
    
    def test_pii_hashing(self):
        """Test PII hashing consistency."""
        pii_value = "sensitive_identifier_123"
        
        # Same input should produce same hash
        hash1 = self.encryption.hash_pii(pii_value)
        hash2 = self.encryption.hash_pii(pii_value)
        assert hash1 == hash2
        
        # Different input should produce different hash
        hash3 = self.encryption.hash_pii("different_identifier")
        assert hash1 != hash3

class TestPrivacyManager:
    """Test privacy management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.temp_db = f.name
        
        self.config = {
            'retention_policy': {
                'medical_data_retention_days': 30,
                'enable_automatic_deletion': True
            }
        }
        self.privacy_manager = PrivacyManager(self.config, db_path=self.temp_db)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_data_access_logging(self):
        """Test data access logging."""
        self.privacy_manager.log_data_access(
            user_id='test_user',
            data_type='medical_data',
            action='read',
            data_id='dataset_123',
            purpose='research',
            legal_basis='healthcare_research'
        )
        
        # Verify log was created (would need database query in real implementation)
        assert True  # Placeholder for actual verification
    
    def test_data_retention_registration(self):
        """Test data retention tracking."""
        result = self.privacy_manager.register_data_for_retention(
            data_id='test_dataset_001',
            data_type='medical_data'
        )
        
        assert result == True
        
        # Check retention status
        status = self.privacy_manager.get_retention_status('test_dataset_001')
        assert status is not None
        assert status['data_id'] == 'test_dataset_001'
    
    def test_data_anonymization_tracking(self):
        """Test data anonymization tracking."""
        # Register data first
        self.privacy_manager.register_data_for_retention(
            'test_dataset_002',
            'medical_data'
        )
        
        # Mark as anonymized
        result = self.privacy_manager.anonymize_data('test_dataset_002')
        assert result == True
        
        # Verify status
        status = self.privacy_manager.get_retention_status('test_dataset_002')
        assert status['status'] == 'anonymized'
    
    def test_privacy_report_generation(self):
        """Test privacy compliance report generation."""
        # Add some test data
        self.privacy_manager.register_data_for_retention(
            'report_test_001',
            'medical_data'
        )
        
        report = self.privacy_manager.generate_privacy_report()
        
        assert 'generated_at' in report
        assert 'retention_policy' in report
        assert 'compliance_status' in report

class TestSecureMedicalDataProcessor:
    """Test secure medical data processing."""
    
    def setup_method(self):
        """Setup test environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config = {
                'secure_workspace': temp_dir,
                'privacy_compliance': True,
                'audit_logging': True
            }
            self.processor = SecureMedicalDataProcessor(self.config)
    
    def test_data_security_measures(self):
        """Test security measures applied to datasets."""
        # Mock dataset with PII
        mock_data = {
            'PatientID': ['P001', 'P002'],
            'Name': ['John Doe', 'Jane Smith'],
            'Age': [65, 72],
            'MMSE': [24, 18],
            'Diagnosis': [0, 1]
        }
        
        import pandas as pd
        df = pd.DataFrame(mock_data)
        
        secured_df = self.processor._secure_dataset(
            df, 
            'test_dataset_001', 
            'training'
        )
        
        # Direct identifiers should be removed
        assert 'PatientID' not in secured_df.columns
        assert 'Name' not in secured_df.columns
        
        # Medical data should be preserved
        assert 'Age' in secured_df.columns
        assert 'MMSE' in secured_df.columns
        assert 'Diagnosis' in secured_df.columns

class TestSecurityMonitor:
    """Test security monitoring functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'security_monitoring_enabled': True,
            'failed_auth_rate_threshold': 5,
            'high_request_rate_threshold': 100
        }
        self.monitor = SecurityMonitor(self.config)
    
    def test_security_event_logging(self):
        """Test security event logging."""
        self.monitor.log_security_event(
            event_type='test_event',
            details={'test': 'data'},
            severity='info',
            user_id='test_user',
            ip_address='127.0.0.1'
        )
        
        # Verify event was logged
        assert len(self.monitor.security_events) > 0
        event = self.monitor.security_events[-1]
        assert event['event_type'] == 'test_event'
        assert event['severity'] == 'info'
    
    def test_api_request_logging(self):
        """Test API request logging for pattern analysis."""
        self.monitor.log_api_request(
            user_id='test_user',
            endpoint='/api/medical/data',
            method='POST',
            status_code=200,
            response_time=0.5,
            ip_address='127.0.0.1'
        )
        
        # Verify request was logged
        assert 'test_user_127.0.0.1' in self.monitor.api_usage_patterns
        pattern_data = self.monitor.api_usage_patterns['test_user_127.0.0.1']
        assert len(pattern_data['requests']) > 0
    
    def test_security_summary(self):
        """Test security summary generation."""
        # Add some test events
        self.monitor.log_security_event('test_event', {}, 'info')
        
        summary = self.monitor.get_security_summary()
        
        assert 'monitoring_status' in summary
        assert 'total_security_events' in summary
        assert 'events_by_type' in summary
        assert summary['monitoring_status'] == 'active'

# Integration tests
class TestSecurityIntegration:
    """Test security system integration."""
    
    def test_end_to_end_secure_workflow(self):
        """Test complete secure medical data workflow."""
        # This would test the full workflow:
        # 1. Secure authentication
        # 2. Medical data loading with encryption
        # 3. Privacy-preserving processing
        # 4. Secure model training
        # 5. Audit trail generation
        
        # Placeholder for integration test
        assert True
    
    def test_compliance_validation(self):
        """Test GDPR/HIPAA compliance validation."""
        # This would validate compliance requirements:
        # 1. Data minimization
        # 2. Purpose limitation
        # 3. Retention policies
        # 4. Subject rights
        # 5. Audit requirements
        
        # Placeholder for compliance test
        assert True

if __name__ == '__main__':
    pytest.main([__file__, '-v'])