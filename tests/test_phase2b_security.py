"""
Tests for Phase 2B Enhanced Clinical Security features.

Tests for:
- Zero-Trust Architecture
- Quantum-Safe Cryptography
- Blockchain Medical Records
"""

import pytest
import time
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'security'))

from zero_trust import ZeroTrustArchitecture
from quantum_crypto import QuantumSafeCryptography
from blockchain_records import BlockchainMedicalRecords, SmartContract


class TestZeroTrustArchitecture:
    """Test Zero-Trust Architecture implementation."""
    
    def test_initialization(self):
        """Test zero-trust system initialization."""
        config = {'zero_trust_enabled': True}
        zt = ZeroTrustArchitecture(config)
        
        assert zt.enabled is True
        assert len(zt.network_segments) == 4
        assert 'clinical' in zt.network_segments
    
    def test_session_creation(self):
        """Test zero-trust session creation."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        user_info = {
            'roles': ['clinician'],
            'mfa_verified': True,
            'new_device': False
        }
        
        session_id = zt.create_session('user123', user_info)
        assert session_id is not None
        assert session_id in zt.active_sessions
        assert zt.session_contexts[session_id]['network_segment'] == 'clinical'
    
    def test_continuous_authentication_valid(self):
        """Test continuous authentication with valid session."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        user_info = {'roles': ['clinician'], 'mfa_verified': True}
        session_id = zt.create_session('user123', user_info)
        
        is_valid, reason = zt.verify_continuous_authentication(session_id)
        assert is_valid is True
        assert reason is None
    
    def test_continuous_authentication_expired(self):
        """Test continuous authentication with expired session."""
        config = {'reauthentication_interval': 1}  # 1 second
        zt = ZeroTrustArchitecture(config)
        
        user_info = {'roles': ['clinician'], 'mfa_verified': True}
        session_id = zt.create_session('user123', user_info)
        
        # Wait for expiration
        time.sleep(2)
        
        is_valid, reason = zt.verify_continuous_authentication(session_id)
        assert is_valid is False
        assert 'Reauthentication required' in reason
    
    def test_policy_enforcement_allowed(self):
        """Test policy enforcement for allowed access."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        user_info = {
            'roles': ['clinician', 'physician'],
            'mfa_verified': True
        }
        session_id = zt.create_session('user123', user_info)
        
        # Manually set correct segment
        zt.session_contexts[session_id]['network_segment'] = 'critical'
        zt.session_contexts[session_id]['risk_score'] = 10
        
        allowed, reason = zt.enforce_policy(session_id, 'clinical_decision', 'execute')
        assert allowed is True
    
    def test_policy_enforcement_denied_role(self):
        """Test policy enforcement denied due to role."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        user_info = {
            'roles': ['user'],  # Not clinician
            'mfa_verified': True
        }
        session_id = zt.create_session('user123', user_info)
        
        allowed, reason = zt.enforce_policy(session_id, 'patient_data_access', 'read')
        assert allowed is False
        assert 'role' in reason.lower()
    
    def test_policy_enforcement_denied_risk(self):
        """Test policy enforcement denied due to high risk."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        user_info = {
            'roles': ['clinician'],
            'mfa_verified': True
        }
        session_id = zt.create_session('user123', user_info)
        
        # Set high risk score
        zt.session_contexts[session_id]['risk_score'] = 90
        
        allowed, reason = zt.enforce_policy(session_id, 'patient_data_access', 'read')
        assert allowed is False
        assert 'risk' in reason.lower()
    
    def test_micro_segmentation(self):
        """Test micro-segmentation implementation."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        user_info = {
            'roles': ['physician'],
            'mfa_verified': True
        }
        session_id = zt.create_session('user123', user_info)
        
        # Move to critical segment
        success, reason = zt.implement_micro_segmentation(session_id, 'critical')
        assert success is True
        assert zt.session_contexts[session_id]['network_segment'] == 'critical'
    
    def test_micro_segmentation_denied(self):
        """Test micro-segmentation denied for insufficient privileges."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        user_info = {
            'roles': ['user'],  # Not privileged
            'mfa_verified': True
        }
        session_id = zt.create_session('user123', user_info)
        
        # Try to move to critical segment
        success, reason = zt.implement_micro_segmentation(session_id, 'critical')
        assert success is False
        assert 'privileges' in reason.lower()
    
    def test_penetration_testing(self):
        """Test penetration testing validation."""
        config = {}
        zt = ZeroTrustArchitecture(config)
        
        results = zt.validate_with_penetration_testing()
        
        assert 'tests_passed' in results
        assert 'tests_failed' in results
        assert 'vulnerabilities' in results
        assert results['tests_passed'] >= 3  # Should pass most tests


class TestQuantumSafeCryptography:
    """Test Quantum-Safe Cryptography implementation."""
    
    def test_initialization(self):
        """Test quantum-safe crypto initialization."""
        config = {'quantum_algorithm': 'kyber768'}
        qsc = QuantumSafeCryptography(config)
        
        assert qsc.enabled is True
        assert qsc.selected_algorithm == 'kyber768'
        assert qsc.hybrid_mode is True
    
    def test_algorithm_evaluation(self):
        """Test post-quantum algorithm evaluation."""
        config = {}
        qsc = QuantumSafeCryptography(config)
        
        results = qsc.evaluate_algorithms()
        
        assert 'evaluated_algorithms' in results
        assert len(results['evaluated_algorithms']) >= 4
        assert results['recommended_algorithm'] == 'kyber768'
    
    def test_keypair_generation(self):
        """Test quantum keypair generation."""
        config = {}
        qsc = QuantumSafeCryptography(config)
        
        public_key, private_key = qsc.generate_quantum_keypair()
        
        assert public_key is not None
        assert private_key is not None
        assert len(private_key) > 0
        assert len(public_key) > 0
    
    def test_hybrid_encryption_decryption(self):
        """Test hybrid encryption and decryption."""
        config = {}
        qsc = QuantumSafeCryptography(config)
        
        # Generate keypair
        public_key, private_key = qsc.generate_quantum_keypair()
        
        # Test data
        original_data = b"Sensitive medical data for patient 12345"
        
        # Encrypt
        encrypted = qsc.hybrid_encrypt(original_data, public_key)
        
        assert 'ciphertext' in encrypted
        assert 'quantum_encrypted_key' in encrypted
        assert encrypted['hybrid_mode'] is True
        
        # Decrypt
        decrypted = qsc.hybrid_decrypt(encrypted, private_key)
        
        assert decrypted == original_data
    
    def test_quantum_key_exchange(self):
        """Test quantum-resistant key exchange."""
        config = {}
        qsc = QuantumSafeCryptography(config)
        
        # Generate keypairs for both parties
        public1, private1 = qsc.generate_quantum_keypair()
        public2, private2 = qsc.generate_quantum_keypair()
        
        # Perform key exchange
        shared_secret1 = qsc.quantum_key_exchange(private1, public2)
        
        assert shared_secret1 is not None
        assert len(shared_secret1) == 32  # 256-bit shared secret
    
    def test_performance_impact(self):
        """Test quantum-safe encryption performance impact."""
        config = {}
        qsc = QuantumSafeCryptography(config)
        
        results = qsc.test_performance_impact(test_data_size=1024)
        
        assert 'measurements' in results
        assert 'encryption' in results['measurements']
        assert 'decryption' in results['measurements']
        assert 'key_exchange' in results['measurements']
        
        # Performance should be acceptable
        assert results['measurements']['encryption']['average_ms'] < 1000
    
    def test_migration_path_documentation(self):
        """Test migration path documentation."""
        config = {}
        qsc = QuantumSafeCryptography(config)
        
        migration_doc = qsc.document_migration_path()
        
        assert 'current_state' in migration_doc
        assert 'target_state' in migration_doc
        assert 'migration_phases' in migration_doc
        assert len(migration_doc['migration_phases']) == 4
        assert 'security_benefits' in migration_doc
        assert 'compliance_mapping' in migration_doc
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        config = {}
        qsc = QuantumSafeCryptography(config)
        
        # Perform some operations
        public_key, private_key = qsc.generate_quantum_keypair()
        data = b"test data"
        encrypted = qsc.hybrid_encrypt(data, public_key)
        decrypted = qsc.hybrid_decrypt(encrypted, private_key)
        
        metrics = qsc.get_performance_metrics()
        
        assert 'encryption' in metrics
        assert 'decryption' in metrics
        assert metrics['encryption']['count'] > 0


class TestBlockchainMedicalRecords:
    """Test Blockchain Medical Records implementation."""
    
    def test_initialization(self):
        """Test blockchain initialization."""
        config = {'blockchain_enabled': True}
        blockchain = BlockchainMedicalRecords(config)
        
        assert blockchain.enabled is True
        assert len(blockchain.chain) == 1  # Genesis block
        assert blockchain.chain[0].index == 0
    
    def test_block_creation(self):
        """Test adding blocks to blockchain."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        data = {'type': 'test', 'message': 'Test block'}
        block = blockchain.add_block(data)
        
        assert block.index == 1
        assert block.data == data
        assert block.previous_hash == blockchain.chain[0].hash
    
    def test_chain_integrity(self):
        """Test blockchain integrity verification."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        # Add some blocks
        blockchain.add_block({'type': 'test1'})
        blockchain.add_block({'type': 'test2'})
        blockchain.add_block({'type': 'test3'})
        
        is_valid, error = blockchain.verify_chain_integrity()
        assert is_valid is True
        assert error is None
    
    def test_audit_trail_recording(self):
        """Test immutable audit trail recording."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        block = blockchain.record_audit_trail(
            event_type='data_access',
            patient_id='patient123',
            user_id='doctor456',
            action='view_record',
            details={'ip_address': '192.168.1.1'}
        )
        
        assert block.data['type'] == 'audit_trail'
        assert block.data['patient_id'] == 'patient123'
        assert block.data['user_id'] == 'doctor456'
    
    def test_patient_consent_management(self):
        """Test patient consent management on blockchain."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        scope = {
            'allowed_purposes': ['treatment', 'research'],
            'authorized_parties': ['doctor456', 'researcher789'],
            'expiration': (datetime.now() + timedelta(days=365)).isoformat()
        }
        
        block = blockchain.manage_patient_consent(
            patient_id='patient123',
            consent_type='data_sharing',
            granted=True,
            scope=scope
        )
        
        assert block.data['type'] == 'consent_management'
        assert block.data['granted'] is True
        assert 'patient123' in blockchain.consent_registry
    
    def test_consent_verification_granted(self):
        """Test consent verification for granted consent."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        scope = {
            'allowed_purposes': ['treatment'],
            'authorized_parties': ['doctor456']
        }
        
        blockchain.manage_patient_consent(
            patient_id='patient123',
            consent_type='data_sharing',
            granted=True,
            scope=scope
        )
        
        context = {
            'purpose': 'treatment',
            'requester_id': 'doctor456'
        }
        
        is_granted, reason = blockchain.verify_consent('patient123', 'data_sharing', context)
        assert is_granted is True
    
    def test_consent_verification_denied(self):
        """Test consent verification for denied consent."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        scope = {
            'allowed_purposes': ['treatment'],
            'authorized_parties': ['doctor456']
        }
        
        blockchain.manage_patient_consent(
            patient_id='patient123',
            consent_type='data_sharing',
            granted=True,
            scope=scope
        )
        
        # Wrong purpose
        context = {
            'purpose': 'marketing',
            'requester_id': 'doctor456'
        }
        
        is_granted, reason = blockchain.verify_consent('patient123', 'data_sharing', context)
        assert is_granted is False
        assert 'purpose' in reason.lower()
    
    def test_smart_contract_creation(self):
        """Test smart contract creation."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        terms = {
            'allowed_actions': ['read', 'view'],
            'authorized_parties': ['doctor456'],
            'expiration': (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        contract = blockchain.create_smart_contract(
            contract_id='contract123',
            owner_id='patient123',
            terms=terms
        )
        
        assert contract.contract_id == 'contract123'
        assert contract.status == 'active'
        assert 'contract123' in blockchain.smart_contracts
    
    def test_smart_contract_execution_allowed(self):
        """Test smart contract execution with allowed access."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        terms = {
            'allowed_actions': ['read'],
            'authorized_parties': ['doctor456']
        }
        
        blockchain.create_smart_contract(
            contract_id='contract123',
            owner_id='patient123',
            terms=terms
        )
        
        context = {
            'requester_id': 'doctor456',
            'action': 'read'
        }
        
        allowed, reason = blockchain.execute_smart_contract('contract123', context)
        assert allowed is True
    
    def test_smart_contract_execution_denied(self):
        """Test smart contract execution with denied access."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        terms = {
            'allowed_actions': ['read'],
            'authorized_parties': ['doctor456']
        }
        
        blockchain.create_smart_contract(
            contract_id='contract123',
            owner_id='patient123',
            terms=terms
        )
        
        # Wrong action
        context = {
            'requester_id': 'doctor456',
            'action': 'delete'
        }
        
        allowed, reason = blockchain.execute_smart_contract('contract123', context)
        assert allowed is False
    
    def test_ehr_integration(self):
        """Test EHR system integration."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        integration_config = {
            'sync_enabled': True,
            'sync_interval': 300,
            'api_endpoint': 'https://ehr.example.com/api'
        }
        
        integration = blockchain.integrate_ehr_system(
            ehr_system_id='epic_system',
            integration_config=integration_config
        )
        
        assert integration['status'] == 'active'
        assert integration['sync_enabled'] is True
        assert 'epic_system' in blockchain.ehr_integrations
    
    def test_patient_audit_trail_retrieval(self):
        """Test retrieving patient audit trail."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        # Record multiple events
        blockchain.record_audit_trail('access', 'patient123', 'doctor456', 'view', {})
        blockchain.record_audit_trail('modification', 'patient123', 'doctor456', 'update', {})
        blockchain.record_audit_trail('access', 'patient123', 'nurse789', 'view', {})
        
        audit_trail = blockchain.get_patient_audit_trail('patient123')
        
        assert len(audit_trail) == 3
        assert all(entry['data']['patient_id'] == 'patient123' for entry in audit_trail)
    
    def test_compliance_review(self):
        """Test compliance review for blockchain-based records."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        # Add some data
        blockchain.record_audit_trail('access', 'patient123', 'doctor456', 'view', {})
        blockchain.manage_patient_consent('patient123', 'data_sharing', True, {})
        blockchain.create_smart_contract('contract123', 'patient123', {})
        
        review = blockchain.conduct_compliance_review()
        
        assert 'blockchain_status' in review
        assert 'hipaa_compliance' in review
        assert 'gdpr_compliance' in review
        assert review['blockchain_status']['integrity_verified'] is True
        assert review['hipaa_compliance']['compliant'] is True
        assert review['gdpr_compliance']['compliant'] is True
    
    def test_blockchain_export(self):
        """Test blockchain export functionality."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        # Add some blocks
        blockchain.add_block({'type': 'test1'})
        blockchain.add_block({'type': 'test2'})
        
        exported = blockchain.export_blockchain()
        
        assert len(exported) == 3  # Genesis + 2 blocks
        assert all('hash' in block for block in exported)
        assert all('index' in block for block in exported)
    
    def test_statistics(self):
        """Test blockchain statistics."""
        config = {}
        blockchain = BlockchainMedicalRecords(config)
        
        # Add data
        blockchain.record_audit_trail('access', 'patient123', 'doctor456', 'view', {})
        blockchain.manage_patient_consent('patient123', 'data_sharing', True, {})
        blockchain.create_smart_contract('contract123', 'patient123', {})
        
        stats = blockchain.get_statistics()
        
        assert stats['total_blocks'] >= 3
        assert stats['total_patients'] >= 1
        assert stats['total_consents'] >= 1
        assert stats['total_smart_contracts'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
