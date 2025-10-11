"""
Security module for DuetMind Adaptive system.

This module provides comprehensive security features including:
- API authentication and authorization
- Input validation and sanitization  
- Data encryption and privacy protection
- Security monitoring and logging
- Advanced safety monitoring with domains
- Pluggable safety checks
- Compliance with healthcare data regulations (HIPAA, GDPR)
- Zero-Trust Architecture (Phase 2B)
- Quantum-Safe Cryptography (Phase 2B)
- Blockchain Medical Records (Phase 2B)
"""

from .auth import SecureAuthManager, require_auth, require_admin
from .validation import InputValidator, SecurityValidator
from .encryption import DataEncryption
from .privacy import PrivacyManager, DataRetentionPolicy
from .monitoring import SecurityMonitor

# Import new advanced safety components
try:
    from .safety_monitor import SafetyMonitor, SafetyDomain, SafetyFinding, ISafetyCheck
    from .safety_checks import (
        SystemSafetyCheck, 
        DataSafetyCheck, 
        ModelSafetyCheck, 
        InteractionSafetyCheck, 
        ClinicalSafetyCheck
    )
    
    # Import Phase 2B Enhanced Security components
    from .zero_trust import ZeroTrustArchitecture
    from .quantum_crypto import QuantumSafeCryptography
    from .blockchain_records import BlockchainMedicalRecords
    
    __all__ = [
        'SecureAuthManager',
        'require_auth',
        'require_admin',
        'InputValidator', 
        'SecurityValidator',
        'DataEncryption',
        'PrivacyManager',
        'DataRetentionPolicy',
        'SecurityMonitor',
        'SafetyMonitor',
        'SafetyDomain', 
        'SafetyFinding',
        'ISafetyCheck',
        'SystemSafetyCheck',
        'DataSafetyCheck', 
        'ModelSafetyCheck',
        'InteractionSafetyCheck',
        'ClinicalSafetyCheck',
        'ZeroTrustArchitecture',
        'QuantumSafeCryptography',
        'BlockchainMedicalRecords'
    ]
except ImportError as e:
    # Fallback if new safety components not yet implemented
    __all__ = [
        'SecureAuthManager',
        'require_auth',
        'require_admin',
        'InputValidator', 
        'SecurityValidator',
        'DataEncryption',
        'PrivacyManager',
        'DataRetentionPolicy',
        'SecurityMonitor'
    ]