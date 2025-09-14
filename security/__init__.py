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
        'ClinicalSafetyCheck'
    ]
except ImportError:
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