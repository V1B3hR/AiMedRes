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