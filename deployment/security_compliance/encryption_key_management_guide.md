# Encryption & Key Management Configuration Guide

## Overview

This guide provides comprehensive instructions for setting up encryption and quantum-safe key management in AiMedRes, ensuring all sensitive data is encrypted at rest and in transit.

## Encryption Architecture

```
┌──────────────────────────────────────┐
│     Data Encryption Layers           │
├──────────────────────────────────────┤
│  1. Transport Layer (TLS 1.2+)       │
│  2. Application Layer (AES-256)      │
│  3. Database Layer (Transparent DE)   │
│  4. File System Layer (LUKS/dm-crypt)│
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│    Quantum-Safe Key Management       │
├──────────────────────────────────────┤
│  - Hybrid Kyber768/AES-256           │
│  - Automated Key Rotation            │
│  - Secure Key Storage (HSM/KMS)      │
│  - Key Backup & Recovery             │
└──────────────────────────────────────┘
```

## Quantum-Safe Cryptography

### Overview

AiMedRes implements post-quantum cryptography using hybrid encryption (classical + post-quantum algorithms) to ensure future-proof security.

### Initialize Quantum-Safe Crypto

```python
from security.quantum_crypto import QuantumSafeCryptography

# Initialize quantum-safe cryptography system
quantum_crypto = QuantumSafeCryptography({
    'quantum_safe_enabled': True,
    'quantum_algorithm': 'kyber768',  # NIST Level 3 security
    'hybrid_mode': True,              # Combine with classical crypto
    'classical_algorithm': 'aes256',
    'key_rotation_days': 90
})

# Evaluate algorithm selection
evaluation = quantum_crypto.evaluate_algorithms()
print(f"Selected algorithm: {evaluation['recommended_algorithm']}")
print(f"Security level: {evaluation['security_assessment']}")
```

### Supported Quantum-Safe Algorithms

| Algorithm | Security Level | Key Size | Performance | Recommendation |
|-----------|---------------|----------|-------------|----------------|
| kyber512 | NIST Level 1 | 800 bytes | Fast | Development/Testing |
| kyber768 | NIST Level 3 | 1184 bytes | Medium | **Production (Recommended)** |
| kyber1024 | NIST Level 5 | 1568 bytes | Slow | High-security environments |
| dilithium2 | NIST Level 2 | 1312 bytes | Fast | Digital signatures |
| dilithium3 | NIST Level 3 | 1952 bytes | Medium | Digital signatures |

### Hybrid Encryption Implementation

```python
class HybridEncryptionEngine:
    """Hybrid classical + post-quantum encryption"""
    
    def __init__(self, quantum_crypto):
        self.quantum_crypto = quantum_crypto
        self.aes_key_size = 32  # 256 bits
    
    def encrypt(self, plaintext: bytes) -> dict:
        """Encrypt data using hybrid approach"""
        
        # Generate ephemeral AES key
        aes_key = os.urandom(self.aes_key_size)
        
        # Encrypt data with AES-256-GCM
        cipher = Cipher(
            algorithms.AES(aes_key),
            modes.GCM(os.urandom(12))
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Encrypt AES key with quantum-safe algorithm
        encrypted_key_quantum = self.quantum_crypto.encapsulate_key(aes_key)
        
        # Also encrypt with RSA for backward compatibility
        encrypted_key_classical = self._rsa_encrypt(aes_key)
        
        return {
            'ciphertext': ciphertext,
            'tag': encryptor.tag,
            'nonce': cipher.mode.nonce,
            'encrypted_key_quantum': encrypted_key_quantum,
            'encrypted_key_classical': encrypted_key_classical,
            'algorithm': 'hybrid-kyber768-aes256'
        }
    
    def decrypt(self, encrypted_data: dict) -> bytes:
        """Decrypt data using hybrid approach"""
        
        # Try quantum-safe decryption first
        try:
            aes_key = self.quantum_crypto.decapsulate_key(
                encrypted_data['encrypted_key_quantum']
            )
        except Exception:
            # Fallback to classical decryption
            aes_key = self._rsa_decrypt(
                encrypted_data['encrypted_key_classical']
            )
        
        # Decrypt data with AES-256-GCM
        cipher = Cipher(
            algorithms.AES(aes_key),
            modes.GCM(encrypted_data['nonce'], encrypted_data['tag'])
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(encrypted_data['ciphertext']) + decryptor.finalize()
        
        return plaintext

# Usage
hybrid_engine = HybridEncryptionEngine(quantum_crypto)

# Encrypt sensitive data
sensitive_data = b"Patient PHI data"
encrypted = hybrid_engine.encrypt(sensitive_data)

# Decrypt when needed
decrypted = hybrid_engine.decrypt(encrypted)
assert decrypted == sensitive_data
```

## Key Management System

### Key Generation

```python
from security.quantum_prod_keys import ProductionKeyManager

# Initialize production key manager
key_manager = ProductionKeyManager({
    'key_storage_path': '/etc/aimedres/keys',
    'backup_path': '/backup/aimedres/keys',
    'rotation_enabled': True,
    'rotation_days': 90,
    'kms_integration': True,
    'kms_provider': 'aws'  # or 'azure', 'gcp', 'vault'
})

# Generate master key
master_key = key_manager.generate_master_key()
print(f"Master key generated: {master_key['key_id']}")

# Generate data encryption keys (DEK)
dek = key_manager.generate_data_key()
print(f"Data encryption key generated: {dek['key_id']}")
```

### Key Storage

#### Local Secure Storage

```python
import os
import json
from cryptography.fernet import Fernet
from pathlib import Path

class SecureKeyStorage:
    """Secure local key storage with encryption"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Master encryption key (should be in HSM/KMS in production)
        self.master_key = self._load_or_create_master_key()
        self.cipher = Fernet(self.master_key)
    
    def _load_or_create_master_key(self):
        """Load or create master encryption key"""
        key_file = self.storage_path / '.master_key'
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            master_key = Fernet.generate_key()
            
            # Store with restricted permissions
            with open(key_file, 'wb') as f:
                f.write(master_key)
            
            os.chmod(key_file, 0o600)
            return master_key
    
    def store_key(self, key_id: str, key_data: dict):
        """Store encryption key securely"""
        
        # Serialize key data
        key_json = json.dumps(key_data).encode('utf-8')
        
        # Encrypt key data
        encrypted = self.cipher.encrypt(key_json)
        
        # Store encrypted key
        key_file = self.storage_path / f"{key_id}.key"
        with open(key_file, 'wb') as f:
            f.write(encrypted)
        
        os.chmod(key_file, 0o600)
        print(f"✅ Key stored securely: {key_id}")
    
    def load_key(self, key_id: str) -> dict:
        """Load encryption key"""
        
        key_file = self.storage_path / f"{key_id}.key"
        
        if not key_file.exists():
            raise ValueError(f"Key not found: {key_id}")
        
        # Read encrypted key
        with open(key_file, 'rb') as f:
            encrypted = f.read()
        
        # Decrypt key data
        decrypted = self.cipher.decrypt(encrypted)
        key_data = json.loads(decrypted.decode('utf-8'))
        
        return key_data
    
    def list_keys(self) -> list:
        """List all stored keys"""
        return [
            f.stem for f in self.storage_path.glob('*.key')
            if f.name != '.master_key'
        ]
    
    def delete_key(self, key_id: str):
        """Securely delete key"""
        key_file = self.storage_path / f"{key_id}.key"
        
        if key_file.exists():
            # Overwrite before deletion (simple version)
            with open(key_file, 'wb') as f:
                f.write(os.urandom(key_file.stat().st_size))
            
            key_file.unlink()
            print(f"✅ Key deleted: {key_id}")

# Usage
key_storage = SecureKeyStorage('/etc/aimedres/keys')

# Store a key
key_storage.store_key('data-key-001', {
    'algorithm': 'aes-256-gcm',
    'key': 'base64_encoded_key_here',
    'created_at': '2024-01-15T10:00:00Z'
})

# Load a key
key_data = key_storage.load_key('data-key-001')
```

#### Cloud KMS Integration

##### AWS KMS

```python
import boto3
import base64

class AWSKMSIntegration:
    """AWS Key Management Service integration"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.kms_client = boto3.client('kms', region_name=region)
        self.master_key_id = None
    
    def create_master_key(self, description: str = 'AiMedRes Master Key'):
        """Create KMS master key"""
        
        response = self.kms_client.create_key(
            Description=description,
            KeyUsage='ENCRYPT_DECRYPT',
            Origin='AWS_KMS',
            MultiRegion=False,
            Tags=[
                {'TagKey': 'Application', 'TagValue': 'AiMedRes'},
                {'TagKey': 'Purpose', 'TagValue': 'DataEncryption'}
            ]
        )
        
        self.master_key_id = response['KeyMetadata']['KeyId']
        
        # Create alias
        self.kms_client.create_alias(
            AliasName='alias/aimedres-master',
            TargetKeyId=self.master_key_id
        )
        
        print(f"✅ KMS master key created: {self.master_key_id}")
        return self.master_key_id
    
    def generate_data_key(self) -> dict:
        """Generate data encryption key"""
        
        response = self.kms_client.generate_data_key(
            KeyId=self.master_key_id,
            KeySpec='AES_256'
        )
        
        return {
            'plaintext_key': response['Plaintext'],
            'encrypted_key': response['CiphertextBlob']
        }
    
    def encrypt_data(self, plaintext: bytes) -> dict:
        """Encrypt data using KMS"""
        
        response = self.kms_client.encrypt(
            KeyId=self.master_key_id,
            Plaintext=plaintext
        )
        
        return {
            'ciphertext': response['CiphertextBlob'],
            'key_id': response['KeyId']
        }
    
    def decrypt_data(self, ciphertext: bytes) -> bytes:
        """Decrypt data using KMS"""
        
        response = self.kms_client.decrypt(
            CiphertextBlob=ciphertext
        )
        
        return response['Plaintext']
    
    def rotate_key(self):
        """Enable automatic key rotation"""
        
        self.kms_client.enable_key_rotation(
            KeyId=self.master_key_id
        )
        
        print(f"✅ Automatic key rotation enabled")

# Usage
kms = AWSKMSIntegration(region='us-east-1')
kms.create_master_key()

# Generate data key
data_key = kms.generate_data_key()

# Use plaintext key for encryption, store encrypted key
plaintext_key = data_key['plaintext_key']
encrypted_key = data_key['encrypted_key']
```

##### Azure Key Vault

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.keys import KeyClient
from azure.keyvault.secrets import SecretClient

class AzureKeyVaultIntegration:
    """Azure Key Vault integration"""
    
    def __init__(self, vault_url: str):
        credential = DefaultAzureCredential()
        self.key_client = KeyClient(vault_url, credential)
        self.secret_client = SecretClient(vault_url, credential)
    
    def create_key(self, key_name: str):
        """Create key in Azure Key Vault"""
        
        key = self.key_client.create_rsa_key(
            key_name,
            size=4096,
            tags={'application': 'aimedres'}
        )
        
        print(f"✅ Azure Key Vault key created: {key.name}")
        return key
    
    def store_secret(self, secret_name: str, secret_value: str):
        """Store secret in Azure Key Vault"""
        
        secret = self.secret_client.set_secret(
            secret_name,
            secret_value,
            tags={'application': 'aimedres'}
        )
        
        print(f"✅ Secret stored: {secret.name}")
        return secret
    
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Azure Key Vault"""
        
        secret = self.secret_client.get_secret(secret_name)
        return secret.value

# Usage
akv = AzureKeyVaultIntegration('https://aimedres-vault.vault.azure.net/')
akv.create_key('aimedres-master-key')
akv.store_secret('database-password', 'secure_password_here')
```

### Key Rotation

```python
from datetime import datetime, timedelta

class KeyRotationManager:
    """Automated key rotation management"""
    
    def __init__(self, key_storage, rotation_days: int = 90):
        self.key_storage = key_storage
        self.rotation_days = rotation_days
        self.rotation_log = []
    
    def check_key_rotation_needed(self, key_id: str) -> bool:
        """Check if key needs rotation"""
        
        key_data = self.key_storage.load_key(key_id)
        created_at = datetime.fromisoformat(key_data['created_at'])
        age_days = (datetime.now() - created_at).days
        
        return age_days >= self.rotation_days
    
    def rotate_key(self, old_key_id: str) -> str:
        """Rotate encryption key"""
        
        print(f"🔄 Rotating key: {old_key_id}")
        
        # Generate new key
        new_key_id = f"{old_key_id}-{datetime.now().strftime('%Y%m%d')}"
        new_key_data = {
            'algorithm': 'aes-256-gcm',
            'key': base64.b64encode(os.urandom(32)).decode('utf-8'),
            'created_at': datetime.now().isoformat(),
            'rotated_from': old_key_id
        }
        
        # Store new key
        self.key_storage.store_key(new_key_id, new_key_data)
        
        # Log rotation
        self.rotation_log.append({
            'timestamp': datetime.now().isoformat(),
            'old_key_id': old_key_id,
            'new_key_id': new_key_id,
            'reason': 'scheduled_rotation'
        })
        
        print(f"✅ Key rotated: {old_key_id} → {new_key_id}")
        
        # Note: Old key should be kept for a grace period to decrypt old data
        # then securely deleted after re-encryption is complete
        
        return new_key_id
    
    def re_encrypt_data(self, data: bytes, old_key_id: str, new_key_id: str) -> bytes:
        """Re-encrypt data with new key"""
        
        # Load keys
        old_key = self.key_storage.load_key(old_key_id)
        new_key = self.key_storage.load_key(new_key_id)
        
        # Decrypt with old key
        # (Implementation depends on encryption method)
        plaintext = decrypt_with_key(data, old_key)
        
        # Encrypt with new key
        ciphertext = encrypt_with_key(plaintext, new_key)
        
        return ciphertext
    
    def auto_rotate_check(self):
        """Check all keys and rotate if needed"""
        
        print("🔍 Checking keys for rotation...")
        
        for key_id in self.key_storage.list_keys():
            if self.check_key_rotation_needed(key_id):
                self.rotate_key(key_id)

# Usage
rotation_manager = KeyRotationManager(key_storage, rotation_days=90)

# Run periodic rotation check
rotation_manager.auto_rotate_check()
```

## Data Encryption

### Database Encryption

#### PostgreSQL Encryption

```sql
-- Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create encrypted column
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50),
    encrypted_data BYTEA,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Encrypt data on insert
INSERT INTO patients (patient_id, encrypted_data)
VALUES (
    'PATIENT-001',
    pgp_sym_encrypt('sensitive patient data', 'encryption_key')
);

-- Decrypt data on select
SELECT 
    patient_id,
    pgp_sym_decrypt(encrypted_data, 'encryption_key') as decrypted_data
FROM patients;
```

#### Application-Level Encryption

```python
from cryptography.fernet import Fernet
import psycopg2

class EncryptedDatabaseConnection:
    """Database connection with automatic encryption"""
    
    def __init__(self, connection_string: str, encryption_key: bytes):
        self.conn = psycopg2.connect(connection_string)
        self.cipher = Fernet(encryption_key)
    
    def insert_encrypted(self, table: str, data: dict):
        """Insert data with automatic encryption"""
        
        # Encrypt sensitive fields
        encrypted_data = {}
        for key, value in data.items():
            if isinstance(value, str) and self._is_sensitive(key):
                encrypted_data[key] = self.cipher.encrypt(value.encode())
            else:
                encrypted_data[key] = value
        
        # Insert into database
        cursor = self.conn.cursor()
        columns = ', '.join(encrypted_data.keys())
        placeholders = ', '.join(['%s'] * len(encrypted_data))
        
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor.execute(query, list(encrypted_data.values()))
        self.conn.commit()
    
    def select_encrypted(self, table: str, condition: str = '1=1'):
        """Select data with automatic decryption"""
        
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE {condition}")
        
        rows = []
        for row in cursor.fetchall():
            decrypted_row = {}
            for idx, value in enumerate(row):
                column_name = cursor.description[idx][0]
                
                if self._is_sensitive(column_name) and isinstance(value, bytes):
                    decrypted_row[column_name] = self.cipher.decrypt(value).decode()
                else:
                    decrypted_row[column_name] = value
            
            rows.append(decrypted_row)
        
        return rows
    
    def _is_sensitive(self, field_name: str) -> bool:
        """Check if field contains sensitive data"""
        sensitive_fields = ['ssn', 'name', 'address', 'phone', 'email']
        return any(s in field_name.lower() for s in sensitive_fields)

# Usage
encryption_key = Fernet.generate_key()
db = EncryptedDatabaseConnection(
    'postgresql://user:pass@localhost/aimedres',
    encryption_key
)

# Insert with encryption
db.insert_encrypted('patients', {
    'patient_id': 'P001',
    'name': 'John Doe',  # Will be encrypted
    'age': 45            # Will not be encrypted
})

# Select with decryption
patients = db.select_encrypted('patients', "patient_id = 'P001'")
```

### File Encryption

```python
from cryptography.fernet import Fernet
from pathlib import Path

class FileEncryption:
    """Encrypt/decrypt files"""
    
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)
    
    def encrypt_file(self, input_path: str, output_path: str = None):
        """Encrypt file"""
        
        input_file = Path(input_path)
        output_file = Path(output_path or f"{input_path}.encrypted")
        
        # Read plaintext
        with open(input_file, 'rb') as f:
            plaintext = f.read()
        
        # Encrypt
        ciphertext = self.cipher.encrypt(plaintext)
        
        # Write encrypted file
        with open(output_file, 'wb') as f:
            f.write(ciphertext)
        
        # Set secure permissions
        output_file.chmod(0o600)
        
        print(f"✅ File encrypted: {output_file}")
        return str(output_file)
    
    def decrypt_file(self, input_path: str, output_path: str = None):
        """Decrypt file"""
        
        input_file = Path(input_path)
        output_file = Path(output_path or input_path.replace('.encrypted', ''))
        
        # Read ciphertext
        with open(input_file, 'rb') as f:
            ciphertext = f.read()
        
        # Decrypt
        plaintext = self.cipher.decrypt(ciphertext)
        
        # Write decrypted file
        with open(output_file, 'wb') as f:
            f.write(plaintext)
        
        print(f"✅ File decrypted: {output_file}")
        return str(output_file)

# Usage
file_encryption = FileEncryption(encryption_key)

# Encrypt patient data file
file_encryption.encrypt_file('patient_data.csv', 'patient_data.csv.encrypted')

# Decrypt when needed
file_encryption.decrypt_file('patient_data.csv.encrypted', 'patient_data.csv')
```

## Environment Configuration

Add to `.env` file:

```bash
# Encryption Configuration
ENCRYPTION_ENABLED=true
ENCRYPTION_ALGORITHM=aes-256-gcm
ENCRYPTION_KEY_PATH=/etc/aimedres/keys

# Quantum-Safe Crypto
QUANTUM_SAFE_ENABLED=true
QUANTUM_ALGORITHM=kyber768
HYBRID_MODE=true

# Key Management
KEY_ROTATION_ENABLED=true
KEY_ROTATION_DAYS=90
KMS_PROVIDER=aws  # or azure, gcp, vault, local
KMS_REGION=us-east-1

# AWS KMS (if using)
AWS_KMS_MASTER_KEY_ID=your_kms_key_id
AWS_KMS_REGION=us-east-1

# Azure Key Vault (if using)
AZURE_KEY_VAULT_URL=https://aimedres-vault.vault.azure.net/
```

## Security Checklist

- [ ] Quantum-safe cryptography enabled
- [ ] Hybrid encryption mode active
- [ ] Data encrypted at rest (database, files)
- [ ] Data encrypted in transit (TLS 1.2+)
- [ ] Master keys stored in HSM/KMS
- [ ] Automated key rotation configured
- [ ] Key backup and recovery procedures documented
- [ ] Encryption keys never logged or exposed
- [ ] Strong encryption algorithms (AES-256, Kyber768)
- [ ] Key access audited and logged
- [ ] Encryption performance impact measured
- [ ] Disaster recovery plan includes key recovery

## Compliance Verification

```python
def verify_encryption_compliance():
    """Verify encryption compliance"""
    
    checks = {
        'Quantum-Safe Enabled': check_quantum_safe_enabled(),
        'Data at Rest Encrypted': check_data_encryption(),
        'TLS Configured': check_tls_encryption(),
        'Key Rotation Active': check_key_rotation(),
        'KMS Integration': check_kms_integration(),
    }
    
    print("\n📋 Encryption & Key Management Compliance Report")
    print("=" * 60)
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n✅ All encryption checks passed")
    else:
        print("\n❌ Some encryption checks failed")
    
    return all_passed

if __name__ == '__main__':
    verify_encryption_compliance()
```

## References

- NIST Post-Quantum Cryptography: [csrc.nist.gov/projects/post-quantum-cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- HIPAA Encryption Standards: [HHS.gov](https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html)
- CRYSTALS-Kyber: [pq-crystals.org/kyber](https://pq-crystals.org/kyber/)
