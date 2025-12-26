"""
Security module for DuetMind Adaptive system.

This module provides comprehensive security features including:
- API authentication and authorization
- Input validation and sanitization
- Data encryption and privacy protection
- Security monitoring and logging
- Compliance with healthcare data regulations (HIPAA, GDPR)
"""

from .auth import SecureAuthManager, require_admin, require_auth
from .encryption import DataEncryption
from .monitoring import SecurityMonitor
from .privacy import DataRetentionPolicy, PrivacyManager
from .validation import InputValidator, SecurityValidator

__all__ = [
    "SecureAuthManager",
    "require_auth",
    "require_admin",
    "InputValidator",
    "SecurityValidator",
    "DataEncryption",
    "PrivacyManager",
    "DataRetentionPolicy",
    "SecurityMonitor",
]
