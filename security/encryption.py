"""
Data encryption and privacy protection utilities.

Provides enterprise-grade encryption for:
- Sensitive data at rest
- API communications
- Medical data protection
- Personal identification information (PII)
- Model parameter protection
"""

import os
import base64
import hashlib
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from typing import Dict, Any, Optional, Union
import json
import logging

security_logger = logging.getLogger('duetmind.security')

class DataEncryption:
    """
    Enterprise-grade encryption for sensitive data.
    
    Features:
    - AES-256 encryption for data at rest
    - RSA encryption for key exchange
    - Password-based key derivation (PBKDF2)
    - Medical data anonymization
    - Secure key management
    - PII protection
    """
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or os.environ.get('DUETMIND_MASTER_KEY')
        if not self.master_password:
            # Generate secure master password if none provided
            self.master_password = base64.urlsafe_b64encode(os.urandom(32)).decode()
            security_logger.warning("Generated temporary master password. Set DUETMIND_MASTER_KEY env var for production.")
        
        self.fernet = self._initialize_encryption()
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._initialize_rsa_keys()
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize Fernet encryption with derived key."""
        # Derive key from master password
        password = self.master_password.encode()
        salt = b'duetmind_salt_v1'  # In production, use unique salt per installation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def _initialize_rsa_keys(self):
        """Initialize RSA key pair for asymmetric encryption."""
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
    
    def encrypt_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """
        Encrypt sensitive data using AES-256.
        
        Args:
            data: Data to encrypt (string or dict)
            
        Returns:
            Base64-encoded encrypted data
        """
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            security_logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """
        Decrypt data encrypted with encrypt_data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
        except Exception as e:
            security_logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_medical_data(self, medical_data: Dict[str, Any], 
                           preserve_structure: bool = True) -> Dict[str, Any]:
        """
        Encrypt medical data while preserving structure for ML training.
        
        Args:
            medical_data: Medical data dictionary
            preserve_structure: Whether to preserve dict structure
            
        Returns:
            Encrypted medical data
        """
        if not preserve_structure:
            return {'encrypted_data': self.encrypt_data(medical_data)}
        
        # Encrypt sensitive fields while preserving structure
        sensitive_fields = [
            'patient_id', 'name', 'ssn', 'address', 'phone', 'email',
            'medical_record_number', 'insurance_id'
        ]
        
        encrypted_data = medical_data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt_data(str(encrypted_data[field]))
        
        return encrypted_data
    
    def anonymize_medical_data(self, medical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize medical data for training while preserving ML utility.
        
        Args:
            medical_data: Medical data dictionary
            
        Returns:
            Anonymized medical data
        """
        anonymized = medical_data.copy()
        
        # Remove direct identifiers
        identifiers_to_remove = [
            'patient_id', 'name', 'ssn', 'address', 'phone', 'email',
            'medical_record_number', 'insurance_id', 'doctor_name'
        ]
        
        for identifier in identifiers_to_remove:
            if identifier in anonymized:
                del anonymized[identifier]
        
        # Hash quasi-identifiers for consistency while preserving privacy
        quasi_identifiers = ['birth_date', 'zip_code']
        
        for qi in quasi_identifiers:
            if qi in anonymized:
                # Create consistent hash for grouping while removing identifiability
                hashed_value = hashlib.sha256(str(anonymized[qi]).encode()).hexdigest()[:8]
                anonymized[f"{qi}_hash"] = hashed_value
                del anonymized[qi]
        
        # Add anonymization metadata
        anonymized['anonymized'] = True
        anonymized['anonymization_version'] = 'v1.0'
        
        return anonymized
    
    def encrypt_model_parameters(self, model_params: Dict[str, Any]) -> str:
        """
        Encrypt ML model parameters for secure storage.
        
        Args:
            model_params: Model parameters dictionary
            
        Returns:
            Encrypted model parameters
        """
        return self.encrypt_data(model_params)
    
    def hash_pii(self, pii_value: str, salt: Optional[str] = None) -> str:
        """
        Create one-way hash of PII for consistent anonymization.
        
        Args:
            pii_value: Personal information to hash
            salt: Optional salt for hashing
            
        Returns:
            Hashed value
        """
        if salt is None:
            salt = "duetmind_pii_salt_v1"
        
        combined = f"{pii_value}_{salt}".encode()
        return hashlib.sha256(combined).hexdigest()
    
    def get_public_key_pem(self) -> str:
        """Get RSA public key in PEM format for client-side encryption."""
        pem = self.rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode()
    
    def rsa_encrypt(self, data: str) -> str:
        """
        Encrypt data using RSA public key (for small data like API keys).
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64-encoded encrypted data
        """
        encrypted = self.rsa_public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def rsa_decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt RSA-encrypted data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.rsa_private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode()
    
    def encrypt_phi_data(self, phi_data: Dict[str, Any], 
                        patient_id: str = None,
                        purpose: str = "treatment",
                        audit_log: bool = True) -> Dict[str, Any]:
        """
        Encrypt Protected Health Information (PHI) data with HIPAA compliance.
        
        Args:
            phi_data: PHI data to encrypt
            patient_id: Patient identifier for audit purposes
            purpose: Purpose of encryption for HIPAA compliance
            audit_log: Whether to log this encryption event
            
        Returns:
            Encrypted PHI data with metadata
        """
        if audit_log and patient_id:
            # Import here to avoid circular imports
            try:
                from .hipaa_audit import audit_phi_access, AccessType
                audit_phi_access(
                    user_id="system_encryption",
                    patient_id=patient_id,
                    action="WRITE",
                    purpose=f"data_encryption_{purpose}",
                    resource="phi_data",
                    user_role="system"
                )
            except ImportError:
                security_logger.warning("HIPAA audit logging not available")
        
        # Enhanced PHI field identification
        phi_fields = [
            'patient_id', 'name', 'first_name', 'last_name', 'ssn', 'social_security',
            'address', 'street_address', 'city', 'state', 'zip', 'zipcode', 'postal_code',
            'phone', 'phone_number', 'mobile', 'telephone', 'email', 'email_address',
            'medical_record_number', 'mrn', 'insurance_id', 'insurance_number',
            'date_of_birth', 'dob', 'birth_date', 'drivers_license', 'passport_number',
            'biometric_data', 'photo', 'fingerprint', 'voice_print',
            'account_number', 'certificate_number', 'license_number',
            'vehicle_identifier', 'device_identifier', 'web_url', 'ip_address',
            'full_face_photo', 'comparable_image'
        ]
        
        encrypted_data = phi_data.copy()
        encryption_metadata = {
            'encrypted_fields': [],
            'encryption_timestamp': time.time(),
            'encryption_version': '2.0_hipaa_compliant',
            'patient_id_hash': None,
            'purpose': purpose
        }
        
        # Hash patient ID for tracking
        if patient_id:
            encryption_metadata['patient_id_hash'] = hashlib.sha256(
                patient_id.encode()
            ).hexdigest()[:16]
        
        # Encrypt all PHI fields
        for field in phi_fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                try:
                    original_value = str(encrypted_data[field])
                    encrypted_value = self.encrypt_data(original_value)
                    encrypted_data[field] = encrypted_value
                    encryption_metadata['encrypted_fields'].append(field)
                except Exception as e:
                    security_logger.error(f"Failed to encrypt field {field}: {e}")
        
        # Add metadata
        encrypted_data['_encryption_metadata'] = encryption_metadata
        
        return encrypted_data
    
    def decrypt_phi_data(self, encrypted_phi_data: Dict[str, Any],
                        patient_id: str = None,
                        purpose: str = "treatment",
                        user_id: str = "unknown",
                        audit_log: bool = True) -> Dict[str, Any]:
        """
        Decrypt PHI data with HIPAA audit logging.
        
        Args:
            encrypted_phi_data: Encrypted PHI data
            patient_id: Patient identifier for audit purposes
            purpose: Purpose of decryption for HIPAA compliance
            user_id: User requesting decryption
            audit_log: Whether to log this decryption event
            
        Returns:
            Decrypted PHI data
        """
        if audit_log and patient_id:
            try:
                from .hipaa_audit import audit_phi_access
                audit_phi_access(
                    user_id=user_id,
                    patient_id=patient_id,
                    action="READ",
                    purpose=f"data_decryption_{purpose}",
                    resource="phi_data",
                    user_role="medical_professional"
                )
            except ImportError:
                security_logger.warning("HIPAA audit logging not available")
        
        decrypted_data = encrypted_phi_data.copy()
        
        # Get encryption metadata
        metadata = decrypted_data.get('_encryption_metadata', {})
        encrypted_fields = metadata.get('encrypted_fields', [])
        
        # Decrypt all encrypted fields
        for field in encrypted_fields:
            if field in decrypted_data:
                try:
                    encrypted_value = decrypted_data[field]
                    decrypted_value = self.decrypt_data(encrypted_value)
                    decrypted_data[field] = decrypted_value
                except Exception as e:
                    security_logger.error(f"Failed to decrypt field {field}: {e}")
        
        # Remove encryption metadata from output
        decrypted_data.pop('_encryption_metadata', None)
        
        return decrypted_data
    
    def validate_data_integrity(self, data: Dict[str, Any],
                               expected_hash: str = None) -> bool:
        """
        Validate integrity of encrypted medical data.
        
        Args:
            data: Data to validate
            expected_hash: Expected integrity hash
            
        Returns:
            True if data integrity is valid
        """
        try:
            # Calculate current hash
            data_str = json.dumps(data, sort_keys=True, default=str)
            current_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            if expected_hash:
                return current_hash == expected_hash
            
            # If no expected hash, check if data structure is valid
            metadata = data.get('_encryption_metadata', {})
            return 'encryption_timestamp' in metadata and 'encryption_version' in metadata
            
        except Exception as e:
            security_logger.error(f"Data integrity validation failed: {e}")
            return False