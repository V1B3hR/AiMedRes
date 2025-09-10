#!/usr/bin/env python3
"""
Comprehensive test suite for remote secure training functionality.

Tests all components of the remote training system:
- Security and authentication
- Training job management
- Encryption and data protection
- API endpoints
- Error handling
"""

import unittest
import json
import time
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import components to test
from security.auth import SecureAuthManager
from security.encryption import DataEncryption
from remote_training_manager import RemoteTrainingManager, TrainingStatus
from training import AlzheimerTrainer

class TestSecureAuthentication(unittest.TestCase):
    """Test security and authentication components."""
    
    def setUp(self):
        self.auth_config = {
            'jwt_secret': 'test_secret',
            'max_failed_attempts': 3,
            'lockout_duration_minutes': 5
        }
        self.auth_manager = SecureAuthManager(self.auth_config)
    
    def test_api_key_generation(self):
        """Test secure API key generation."""
        # Check that API keys are generated during initialization
        self.assertGreater(len(self.auth_manager.api_keys), 0)
        
        # Check key format
        for key in self.auth_manager.api_keys.keys():
            self.assertTrue(key.startswith('dmk_'))
            self.assertGreater(len(key), 32)  # Should be cryptographically secure
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Get a valid key
        valid_key = list(self.auth_manager.api_keys.keys())[0]
        
        # Test valid key
        is_valid, user_info = self.auth_manager.validate_api_key(valid_key)
        self.assertTrue(is_valid)
        self.assertIsInstance(user_info, dict)
        self.assertIn('user_id', user_info)
        self.assertIn('roles', user_info)
        
        # Test invalid key
        is_valid, user_info = self.auth_manager.validate_api_key('invalid_key')
        self.assertFalse(is_valid)
        self.assertIsNone(user_info)
    
    def test_role_based_access(self):
        """Test role-based access control."""
        # Get admin and user keys
        admin_key = None
        user_key = None
        
        for key, info in self.auth_manager.api_keys.items():
            if 'admin' in info['roles']:
                admin_key = key
            elif 'admin' not in info['roles']:
                user_key = key
        
        # Test admin access
        if admin_key:
            is_valid, user_info = self.auth_manager.validate_api_key(admin_key)
            self.assertTrue(self.auth_manager.has_role(user_info, 'admin'))
            self.assertTrue(self.auth_manager.has_role(user_info, 'user'))
        
        # Test user access
        if user_key:
            is_valid, user_info = self.auth_manager.validate_api_key(user_key)
            self.assertFalse(self.auth_manager.has_role(user_info, 'admin'))
            self.assertTrue(self.auth_manager.has_role(user_info, 'user'))

class TestDataEncryption(unittest.TestCase):
    """Test encryption and data protection."""
    
    def setUp(self):
        self.encryption = DataEncryption()
    
    def test_symmetric_encryption(self):
        """Test symmetric data encryption."""
        test_data = {'sensitive': 'medical_data', 'patient_id': '12345'}
        
        # Encrypt data
        encrypted = self.encryption.encrypt_data(test_data)
        self.assertIsInstance(encrypted, str)
        self.assertNotEqual(encrypted, str(test_data))
        
        # Decrypt data
        decrypted = self.encryption.decrypt_data(encrypted)
        self.assertEqual(decrypted, test_data)
    
    def test_medical_data_encryption(self):
        """Test medical data encryption with structure preservation."""
        medical_data = {
            'patient_id': 'P12345',
            'name': 'John Doe',
            'age': 65,
            'diagnosis': 'MCI',
            'mmse_score': 24
        }
        
        encrypted = self.encryption.encrypt_medical_data(medical_data)
        
        # Check that sensitive fields are encrypted
        self.assertNotEqual(encrypted.get('patient_id'), medical_data['patient_id'])
        self.assertNotEqual(encrypted.get('name'), medical_data['name'])
        
        # Check that non-sensitive fields are preserved
        self.assertEqual(encrypted.get('age'), medical_data['age'])
        self.assertEqual(encrypted.get('diagnosis'), medical_data['diagnosis'])
    
    def test_data_anonymization(self):
        """Test medical data anonymization."""
        medical_data = {
            'patient_id': 'P12345',
            'name': 'John Doe',
            'birth_date': '1958-01-15',
            'age': 65,
            'diagnosis': 'MCI'
        }
        
        anonymized = self.encryption.anonymize_medical_data(medical_data)
        
        # Check identifiers are removed
        self.assertNotIn('patient_id', anonymized)
        self.assertNotIn('name', anonymized)
        
        # Check quasi-identifiers are hashed
        self.assertIn('birth_date_hash', anonymized)
        self.assertNotIn('birth_date', anonymized)
        
        # Check medical data is preserved
        self.assertEqual(anonymized['age'], medical_data['age'])
        self.assertEqual(anonymized['diagnosis'], medical_data['diagnosis'])
        
        # Check anonymization metadata
        self.assertTrue(anonymized['anonymized'])
        self.assertIn('anonymization_version', anonymized)

class TestRemoteTrainingManager(unittest.TestCase):
    """Test remote training management."""
    
    def setUp(self):
        auth_config = {'jwt_secret': 'test_secret'}
        self.auth_manager = SecureAuthManager(auth_config)
        self.encryption = DataEncryption()
        self.training_manager = RemoteTrainingManager(self.auth_manager, self.encryption)
    
    def test_job_submission(self):
        """Test training job submission."""
        config = {
            'model_type': 'alzheimer_classifier',
            'dataset_source': 'synthetic_test_data'
        }
        
        job_id = self.training_manager.submit_training_job('test_user', config)
        
        self.assertIsInstance(job_id, str)
        self.assertIn(job_id, self.training_manager.training_jobs)
        
        job = self.training_manager.training_jobs[job_id]
        self.assertEqual(job.user_id, 'test_user')
        self.assertEqual(job.config, config)
    
    def test_job_status_tracking(self):
        """Test job status tracking and progress monitoring."""
        config = {
            'model_type': 'alzheimer_classifier',
            'dataset_source': 'synthetic_test_data'
        }
        
        job_id = self.training_manager.submit_training_job('test_user', config)
        
        # Wait for training to complete
        time.sleep(3)
        
        status = self.training_manager.get_job_status(job_id, 'test_user')
        
        self.assertIn('status', status)
        self.assertIn('progress', status)
        self.assertIn('job_id', status)
        self.assertEqual(status['job_id'], job_id)
    
    def test_user_authorization(self):
        """Test user authorization for job access."""
        config = {
            'model_type': 'alzheimer_classifier',
            'dataset_source': 'synthetic_test_data'
        }
        
        job_id = self.training_manager.submit_training_job('user1', config)
        
        # User should be able to access their own job
        status = self.training_manager.get_job_status(job_id, 'user1')
        self.assertEqual(status['job_id'], job_id)
        
        # Different user should not be able to access the job
        with self.assertRaises(PermissionError):
            self.training_manager.get_job_status(job_id, 'user2')
    
    def test_system_capacity_management(self):
        """Test system capacity and resource management."""
        initial_status = self.training_manager.get_system_status()
        self.assertIn('max_concurrent_jobs', initial_status)
        self.assertIn('capacity_available', initial_status)
        
        # Submit multiple jobs to test capacity
        config = {
            'model_type': 'alzheimer_classifier',
            'dataset_source': 'synthetic_test_data'
        }
        
        job_ids = []
        for i in range(2):  # Submit fewer than max capacity
            job_id = self.training_manager.submit_training_job(f'user{i}', config)
            job_ids.append(job_id)
        
        # Check that capacity is being tracked
        status = self.training_manager.get_system_status()
        self.assertGreaterEqual(status['max_concurrent_jobs'], len(job_ids))

class TestTrainingFunctionality(unittest.TestCase):
    """Test core training functionality."""
    
    def setUp(self):
        self.trainer = AlzheimerTrainer()
    
    def test_data_loading(self):
        """Test data loading functionality."""
        data = self.trainer.load_data()
        
        self.assertIsInstance(data, __import__('pandas').DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('diagnosis', data.columns)
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        data = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(data)
        
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(self.trainer.feature_columns), 0)
    
    def test_model_training(self):
        """Test model training."""
        data = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(data)
        model, metrics = self.trainer.train_model(X, y)
        
        self.assertIsNotNone(model)
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_model_parameters_extraction(self):
        """Test model parameter extraction for secure transmission."""
        data = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(data)
        model, metrics = self.trainer.train_model(X, y)
        
        params = self.trainer.get_model_parameters()
        
        self.assertIn('model_state', params)
        self.assertIn('feature_columns', params)
        self.assertIn('model_type', params)
        self.assertIn('preprocessing', params)

class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow."""
    
    def setUp(self):
        auth_config = {'jwt_secret': 'test_secret'}
        self.auth_manager = SecureAuthManager(auth_config)
        self.encryption = DataEncryption()
        self.training_manager = RemoteTrainingManager(self.auth_manager, self.encryption)
    
    def test_complete_training_workflow(self):
        """Test complete training workflow from submission to model retrieval."""
        # 1. Submit training job
        config = {
            'model_type': 'alzheimer_classifier',
            'dataset_source': 'synthetic_test_data'
        }
        
        job_id = self.training_manager.submit_training_job('test_user', config)
        self.assertIsInstance(job_id, str)
        
        # 2. Wait for training to complete
        max_wait = 10  # seconds
        waited = 0
        while waited < max_wait:
            status = self.training_manager.get_job_status(job_id, 'test_user')
            if status['status'] == 'completed':
                break
            time.sleep(1)
            waited += 1
        
        # 3. Check final status
        final_status = self.training_manager.get_job_status(job_id, 'test_user')
        self.assertEqual(final_status['status'], 'completed')
        self.assertEqual(final_status['progress'], 1.0)
        self.assertIn('results', final_status)
        
        # 4. Retrieve encrypted model
        encrypted_model = self.training_manager.get_encrypted_model(job_id, 'test_user')
        self.assertIsInstance(encrypted_model, str)
        self.assertGreater(len(encrypted_model), 0)
        
        # 5. Verify model can be decrypted
        decrypted_model = self.encryption.decrypt_data(encrypted_model)
        self.assertIn('model_params', decrypted_model)
        self.assertIn('metrics', decrypted_model)

def run_tests():
    """Run all test suites."""
    print("🧪 Running Remote Secure Training Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestSecureAuthentication,
        TestDataEncryption,
        TestRemoteTrainingManager,
        TestTrainingFunctionality,
        TestIntegrationWorkflow
    ]
    
    total_tests = 0
    total_failures = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        print("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Summary: {total_tests - total_failures}/{total_tests} tests passed")
    
    if total_failures == 0:
        print("✅ All tests passed! Remote secure training system is working correctly.")
        return True
    else:
        print(f"❌ {total_failures} test(s) failed. Please review the failures above.")
        return False

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)