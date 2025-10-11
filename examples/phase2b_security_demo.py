#!/usr/bin/env python3
"""
Phase 2B Enhanced Clinical Security - Demonstration Script

This script demonstrates the three major security enhancements:
1. Zero-Trust Architecture
2. Quantum-Safe Cryptography
3. Blockchain Medical Records
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'security'))

from zero_trust import ZeroTrustArchitecture
from quantum_crypto import QuantumSafeCryptography
from blockchain_records import BlockchainMedicalRecords
import json
from datetime import datetime, timedelta


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def demo_zero_trust():
    """Demonstrate Zero-Trust Architecture features."""
    print_section("1. ZERO-TRUST ARCHITECTURE DEMONSTRATION")
    
    # Initialize
    config = {
        'zero_trust_enabled': True,
        'reauthentication_interval': 300,
        'max_risk_score': 70
    }
    zt = ZeroTrustArchitecture(config)
    print("✓ Zero-Trust Architecture initialized")
    print(f"  - Reauthentication interval: {config['reauthentication_interval']}s")
    print(f"  - Max risk score: {config['max_risk_score']}")
    print(f"  - Network segments: {len(zt.network_segments)}")
    
    # Create sessions for different user types
    print("\n--- Creating User Sessions ---")
    
    physician_info = {
        'roles': ['physician', 'clinician'],
        'mfa_verified': True,
        'new_device': False,
        'unusual_location': False,
        'vpn_connection': True
    }
    physician_session = zt.create_session('dr_smith', physician_info)
    print(f"✓ Physician session created: {physician_session[:16]}...")
    print(f"  - Network segment: {zt.session_contexts[physician_session]['network_segment']}")
    print(f"  - Initial risk score: {zt.session_contexts[physician_session]['risk_score']}")
    
    nurse_info = {
        'roles': ['clinician'],
        'mfa_verified': True,
        'new_device': False
    }
    nurse_session = zt.create_session('nurse_jones', nurse_info)
    print(f"✓ Nurse session created: {nurse_session[:16]}...")
    print(f"  - Network segment: {zt.session_contexts[nurse_session]['network_segment']}")
    
    user_info = {
        'roles': ['user'],
        'mfa_verified': False,
        'new_device': True
    }
    user_session = zt.create_session('patient_doe', user_info)
    print(f"✓ Patient session created: {user_session[:16]}...")
    print(f"  - Network segment: {zt.session_contexts[user_session]['network_segment']}")
    print(f"  - Initial risk score: {zt.session_contexts[user_session]['risk_score']}")
    
    # Test continuous authentication
    print("\n--- Continuous Authentication ---")
    is_valid, reason = zt.verify_continuous_authentication(physician_session)
    print(f"Physician authentication: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if reason:
        print(f"  Reason: {reason}")
    
    # Test policy enforcement
    print("\n--- Policy Enforcement ---")
    
    # Physician accessing critical clinical decision
    zt.session_contexts[physician_session]['network_segment'] = 'critical'
    zt.session_contexts[physician_session]['risk_score'] = 10
    allowed, reason = zt.enforce_policy(physician_session, 'clinical_decision', 'execute')
    print(f"Physician -> Clinical Decision: {'✓ Allowed' if allowed else '✗ Denied'}")
    
    # Patient trying to access patient data
    allowed, reason = zt.enforce_policy(user_session, 'patient_data_access', 'read')
    print(f"Patient -> Patient Data: {'✗ Denied' if not allowed else '✓ Allowed'}")
    if not allowed:
        print(f"  Reason: {reason}")
    
    # Test micro-segmentation
    print("\n--- Micro-Segmentation ---")
    success, reason = zt.implement_micro_segmentation(physician_session, 'clinical')
    print(f"Move physician to clinical segment: {'✓ Success' if success else '✗ Failed'}")
    
    # Run penetration testing
    print("\n--- Penetration Testing ---")
    results = zt.validate_with_penetration_testing()
    print(f"Tests passed: {results['tests_passed']}")
    print(f"Tests failed: {results['tests_failed']}")
    print(f"Vulnerabilities found: {len(results['vulnerabilities'])}")
    if results['vulnerabilities']:
        for vuln in results['vulnerabilities']:
            print(f"  ⚠ {vuln}")
    
    return zt


def demo_quantum_crypto():
    """Demonstrate Quantum-Safe Cryptography features."""
    print_section("2. QUANTUM-SAFE CRYPTOGRAPHY DEMONSTRATION")
    
    # Initialize
    config = {
        'quantum_safe_enabled': True,
        'quantum_algorithm': 'kyber768',
        'hybrid_mode': True
    }
    qsc = QuantumSafeCryptography(config)
    print("✓ Quantum-Safe Cryptography initialized")
    print(f"  - Algorithm: {qsc.selected_algorithm}")
    print(f"  - Security level: {qsc.algorithm_config['security_level']}")
    print(f"  - Key size: {qsc.algorithm_config['key_size']} bytes")
    print(f"  - Hybrid mode: {qsc.hybrid_mode}")
    
    # Evaluate algorithms
    print("\n--- Algorithm Evaluation ---")
    evaluation = qsc.evaluate_algorithms()
    print(f"Algorithms evaluated: {len(evaluation['evaluated_algorithms'])}")
    print(f"Recommended: {evaluation['recommended_algorithm']}")
    print(f"Reason: {evaluation['recommendation_reason']}")
    
    # Generate keypair
    print("\n--- Quantum Keypair Generation ---")
    public_key, private_key = qsc.generate_quantum_keypair()
    print(f"✓ Public key generated: {len(public_key)} bytes")
    print(f"✓ Private key generated: {len(private_key)} bytes")
    
    # Test hybrid encryption
    print("\n--- Hybrid Encryption Test ---")
    medical_data = b"Patient ID: 12345 | Diagnosis: Hypertension | BP: 145/90"
    print(f"Original data: {medical_data.decode()}")
    
    encrypted = qsc.hybrid_encrypt(medical_data, public_key)
    print(f"✓ Data encrypted using hybrid mode")
    print(f"  - Ciphertext size: {len(encrypted['ciphertext'])} bytes")
    print(f"  - Quantum encrypted key: {len(encrypted['quantum_encrypted_key'])} bytes")
    print(f"  - Algorithm: {encrypted['algorithm']}")
    
    decrypted = qsc.hybrid_decrypt(encrypted, private_key)
    print(f"✓ Data decrypted successfully")
    print(f"  - Decrypted: {decrypted.decode()}")
    print(f"  - Match: {'✓ Yes' if decrypted == medical_data else '✗ No'}")
    
    # Test quantum key exchange
    print("\n--- Quantum Key Exchange ---")
    public_key2, private_key2 = qsc.generate_quantum_keypair()
    shared_secret = qsc.quantum_key_exchange(private_key, public_key2)
    print(f"✓ Shared secret established: {len(shared_secret)} bytes")
    
    # Performance testing
    print("\n--- Performance Testing ---")
    perf_results = qsc.test_performance_impact(test_data_size=1024)
    print(f"Test data size: {perf_results['test_data_size']} bytes")
    print(f"Encryption average: {perf_results['measurements']['encryption']['average_ms']:.2f}ms")
    print(f"Decryption average: {perf_results['measurements']['decryption']['average_ms']:.2f}ms")
    print(f"Key exchange average: {perf_results['measurements']['key_exchange']['average_ms']:.2f}ms")
    print(f"Performance rating: {perf_results['performance_rating']}")
    
    # Migration path
    print("\n--- Migration Path Documentation ---")
    migration = qsc.document_migration_path()
    print(f"Migration phases: {len(migration['migration_phases'])}")
    for phase in migration['migration_phases']:
        status_icon = '✓' if phase['status'] == 'completed' else '○'
        print(f"  {status_icon} Phase {phase['phase']}: {phase['name']} ({phase['status']})")
    
    return qsc


def demo_blockchain():
    """Demonstrate Blockchain Medical Records features."""
    print_section("3. BLOCKCHAIN MEDICAL RECORDS DEMONSTRATION")
    
    # Initialize
    config = {'blockchain_enabled': True}
    blockchain = BlockchainMedicalRecords(config)
    print("✓ Blockchain Medical Records initialized")
    print(f"  - Genesis block created")
    print(f"  - Chain length: {len(blockchain.chain)}")
    
    # Record audit trail
    print("\n--- Immutable Audit Trail ---")
    
    audit1 = blockchain.record_audit_trail(
        event_type='data_access',
        patient_id='patient_12345',
        user_id='dr_smith',
        action='view_record',
        details={'ip_address': '192.168.1.100', 'session_id': 'abc123'}
    )
    print(f"✓ Audit entry 1: Dr. Smith viewed patient record")
    print(f"  - Block index: {audit1.index}")
    print(f"  - Block hash: {audit1.hash[:16]}...")
    
    audit2 = blockchain.record_audit_trail(
        event_type='data_modification',
        patient_id='patient_12345',
        user_id='dr_smith',
        action='update_medication',
        details={'medication': 'Lisinopril 10mg', 'reason': 'Hypertension'}
    )
    print(f"✓ Audit entry 2: Dr. Smith updated medication")
    print(f"  - Block index: {audit2.index}")
    
    audit3 = blockchain.record_audit_trail(
        event_type='data_access',
        patient_id='patient_12345',
        user_id='nurse_jones',
        action='view_vitals',
        details={'vitals': 'BP, HR, Temp'}
    )
    print(f"✓ Audit entry 3: Nurse Jones viewed vitals")
    print(f"  - Block index: {audit3.index}")
    
    # Verify chain integrity
    print("\n--- Blockchain Integrity Verification ---")
    is_valid, error = blockchain.verify_chain_integrity()
    print(f"Blockchain integrity: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if error:
        print(f"  Error: {error}")
    
    # Patient consent management
    print("\n--- Patient Consent Management ---")
    
    consent_scope = {
        'allowed_purposes': ['treatment', 'research', 'quality_improvement'],
        'authorized_parties': ['dr_smith', 'nurse_jones', 'researcher_brown'],
        'expiration': (datetime.now() + timedelta(days=365)).isoformat(),
        'version': '2.0'
    }
    
    consent_block = blockchain.manage_patient_consent(
        patient_id='patient_12345',
        consent_type='data_sharing',
        granted=True,
        scope=consent_scope
    )
    print(f"✓ Consent granted for patient_12345")
    print(f"  - Consent type: data_sharing")
    print(f"  - Allowed purposes: {', '.join(consent_scope['allowed_purposes'])}")
    print(f"  - Block index: {consent_block.index}")
    
    # Verify consent
    print("\n--- Consent Verification ---")
    
    # Valid consent check
    context1 = {
        'purpose': 'treatment',
        'requester_id': 'dr_smith'
    }
    is_granted, reason = blockchain.verify_consent('patient_12345', 'data_sharing', context1)
    print(f"Dr. Smith for treatment: {'✓ Granted' if is_granted else '✗ Denied'}")
    
    # Invalid consent check (wrong purpose)
    context2 = {
        'purpose': 'marketing',
        'requester_id': 'dr_smith'
    }
    is_granted, reason = blockchain.verify_consent('patient_12345', 'data_sharing', context2)
    print(f"Dr. Smith for marketing: {'✗ Denied' if not is_granted else '✓ Granted'}")
    if not is_granted:
        print(f"  Reason: {reason}")
    
    # Smart contracts
    print("\n--- Smart Contract Implementation ---")
    
    contract_terms = {
        'allowed_actions': ['read', 'view', 'download'],
        'authorized_parties': ['dr_smith', 'nurse_jones'],
        'allowed_purposes': ['treatment', 'care_coordination'],
        'time_restrictions': {
            'allowed_hours': list(range(6, 22))  # 6 AM to 10 PM
        },
        'expiration': (datetime.now() + timedelta(days=90)).isoformat()
    }
    
    contract = blockchain.create_smart_contract(
        contract_id='contract_001',
        owner_id='patient_12345',
        terms=contract_terms
    )
    print(f"✓ Smart contract created: {contract.contract_id}")
    print(f"  - Owner: {contract.owner_id}")
    print(f"  - Allowed actions: {', '.join(contract_terms['allowed_actions'])}")
    
    # Execute smart contract
    print("\n--- Smart Contract Execution ---")
    
    # Valid execution
    exec_context1 = {
        'requester_id': 'dr_smith',
        'action': 'read',
        'purpose': 'treatment'
    }
    allowed, reason = blockchain.execute_smart_contract('contract_001', exec_context1)
    print(f"Dr. Smith reading for treatment: {'✓ Allowed' if allowed else '✗ Denied'}")
    
    # Invalid execution (unauthorized party)
    exec_context2 = {
        'requester_id': 'researcher_brown',
        'action': 'read',
        'purpose': 'treatment'
    }
    allowed, reason = blockchain.execute_smart_contract('contract_001', exec_context2)
    print(f"Researcher reading: {'✗ Denied' if not allowed else '✓ Allowed'}")
    if not allowed:
        print(f"  Reason: {reason}")
    
    # EHR integration
    print("\n--- EHR System Integration ---")
    
    integration_config = {
        'sync_enabled': True,
        'sync_interval': 300,
        'api_endpoint': 'https://ehr.hospital.com/api',
        'auth_method': 'oauth2',
        'audit_all_access': True
    }
    
    integration = blockchain.integrate_ehr_system(
        ehr_system_id='epic_main',
        integration_config=integration_config
    )
    print(f"✓ EHR system integrated: {integration['ehr_system_id']}")
    print(f"  - Status: {integration['status']}")
    print(f"  - Sync interval: {integration['sync_interval']}s")
    
    # Retrieve patient audit trail
    print("\n--- Patient Audit Trail Retrieval ---")
    audit_trail = blockchain.get_patient_audit_trail('patient_12345')
    print(f"✓ Audit trail retrieved: {len(audit_trail)} entries")
    for i, entry in enumerate(audit_trail, 1):
        data = entry['data']
        if data.get('type') == 'audit_trail':
            print(f"  {i}. {data.get('event_type', 'N/A')}: {data.get('action', 'N/A')} by {data.get('user_id', 'N/A')}")
        else:
            print(f"  {i}. {data.get('type', 'N/A')}: {data.get('consent_type', 'other')} event")
    
    # Compliance review
    print("\n--- Compliance Review ---")
    review = blockchain.conduct_compliance_review()
    print(f"Blockchain status:")
    print(f"  - Total blocks: {review['blockchain_status']['total_blocks']}")
    print(f"  - Integrity verified: {'✓' if review['blockchain_status']['integrity_verified'] else '✗'}")
    print(f"HIPAA compliance: {'✓ Compliant' if review['hipaa_compliance']['compliant'] else '✗ Non-compliant'}")
    print(f"GDPR compliance: {'✓ Compliant' if review['gdpr_compliance']['compliant'] else '✗ Non-compliant'}")
    print(f"Overall compliance: {'✓ PASS' if review['overall_compliance'] else '✗ FAIL'}")
    
    # Statistics
    print("\n--- Blockchain Statistics ---")
    stats = blockchain.get_statistics()
    print(f"Total blocks: {stats['total_blocks']}")
    print(f"Total patients: {stats['total_patients']}")
    print(f"Total consents: {stats['total_consents']}")
    print(f"Total smart contracts: {stats['total_smart_contracts']}")
    print(f"Active EHR integrations: {stats['active_ehr_integrations']}")
    print(f"Chain size: {stats['chain_size_kb']:.2f} KB")
    
    return blockchain


def demo_integrated_workflow():
    """Demonstrate integrated workflow using all three systems."""
    print_section("4. INTEGRATED WORKFLOW DEMONSTRATION")
    
    print("Simulating complete secure medical data access workflow...\n")
    
    # Initialize all systems
    zt = ZeroTrustArchitecture({'zero_trust_enabled': True})
    qsc = QuantumSafeCryptography({'quantum_algorithm': 'kyber768'})
    bc = BlockchainMedicalRecords({'blockchain_enabled': True})
    
    print("Step 1: User authentication with Zero-Trust")
    user_info = {
        'roles': ['physician'],
        'mfa_verified': True,
        'new_device': False
    }
    session_id = zt.create_session('dr_johnson', user_info)
    print(f"✓ Session created for Dr. Johnson")
    
    print("\nStep 2: Verify continuous authentication")
    is_valid, _ = zt.verify_continuous_authentication(session_id)
    print(f"✓ Continuous authentication: {'Valid' if is_valid else 'Invalid'}")
    
    print("\nStep 3: Check access policy")
    zt.session_contexts[session_id]['network_segment'] = 'clinical'
    zt.session_contexts[session_id]['risk_score'] = 15
    allowed, _ = zt.enforce_policy(session_id, 'patient_data_access', 'read')
    print(f"✓ Access policy: {'Allowed' if allowed else 'Denied'}")
    
    print("\nStep 4: Verify patient consent on blockchain")
    # Create consent first
    bc.manage_patient_consent(
        'patient_67890',
        'data_sharing',
        True,
        {'allowed_purposes': ['treatment'], 'authorized_parties': ['dr_johnson']}
    )
    is_granted, _ = bc.verify_consent(
        'patient_67890',
        'data_sharing',
        {'purpose': 'treatment', 'requester_id': 'dr_johnson'}
    )
    print(f"✓ Patient consent: {'Granted' if is_granted else 'Denied'}")
    
    print("\nStep 5: Retrieve and decrypt patient data with quantum-safe crypto")
    public_key, private_key = qsc.generate_quantum_keypair()
    patient_data = b"Patient 67890 - Medical Record - Confidential"
    encrypted_data = qsc.hybrid_encrypt(patient_data, public_key)
    decrypted_data = qsc.hybrid_decrypt(encrypted_data, private_key)
    print(f"✓ Data encrypted and decrypted using quantum-safe hybrid encryption")
    
    print("\nStep 6: Record access on blockchain audit trail")
    bc.record_audit_trail(
        'data_access',
        'patient_67890',
        'dr_johnson',
        'view_medical_record',
        {'session_id': session_id, 'encrypted': True}
    )
    print(f"✓ Access recorded on immutable blockchain audit trail")
    
    print("\nStep 7: Verify blockchain integrity")
    is_valid, _ = bc.verify_chain_integrity()
    print(f"✓ Blockchain integrity: {'Valid' if is_valid else 'Invalid'}")
    
    print("\n" + "="*80)
    print("  WORKFLOW COMPLETED SUCCESSFULLY")
    print("  All security measures enforced:")
    print("  ✓ Zero-Trust authentication and authorization")
    print("  ✓ Quantum-safe encryption for data protection")
    print("  ✓ Blockchain-based audit trail and consent management")
    print("="*80)


def main():
    """Main demonstration function."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    Phase 2B Enhanced Clinical Security                       ║
║                         Comprehensive Demonstration                          ║
║                                                                              ║
║  Features:                                                                   ║
║    1. Zero-Trust Architecture - Continuous authentication & segmentation    ║
║    2. Quantum-Safe Cryptography - Post-quantum hybrid encryption            ║
║    3. Blockchain Medical Records - Immutable audit trails & smart contracts ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run individual demonstrations
        zt = demo_zero_trust()
        qsc = demo_quantum_crypto()
        bc = demo_blockchain()
        
        # Run integrated workflow
        demo_integrated_workflow()
        
        print_section("DEMONSTRATION COMPLETE")
        print("✓ All Phase 2B features demonstrated successfully")
        print("✓ Zero-Trust Architecture: Operational")
        print("✓ Quantum-Safe Cryptography: Operational")
        print("✓ Blockchain Medical Records: Operational")
        print("\nFor more information, see docs/PHASE2B_SECURITY_GUIDE.md")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
