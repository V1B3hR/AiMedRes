"""
Tests for P3-2: Quantum-Safe Production Key Management
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.quantum_prod_keys import (
    QuantumProductionKeyManager,
    KeyType,
    KeyStatus,
    KeyRotationPolicy,
    create_quantum_key_manager
)


@pytest.fixture
def temp_storage():
    """Create temporary storage for keys."""
    temp_dir = tempfile.mkdtemp(prefix='test_keys_')
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def key_manager(temp_storage):
    """Create key manager instance for testing."""
    config = {
        'quantum_algorithm': 'kyber768',
        'kms_enabled': False,
        'key_storage_path': temp_storage
    }
    
    rotation_policy = KeyRotationPolicy(
        enabled=True,
        rotation_interval_days=90,
        automatic_rotation=False  # Manual for tests
    )
    
    manager = create_quantum_key_manager(config, rotation_policy)
    yield manager
    
    # Cleanup
    if manager.rotation_running:
        manager.stop_rotation_scheduler()


class TestKeyGeneration:
    """Tests for key generation."""
    
    def test_generate_data_encryption_key(self, key_manager):
        """Test generating data encryption key."""
        key = key_manager.generate_key(
            key_type=KeyType.DATA_ENCRYPTION,
            metadata={'purpose': 'patient_data'},
            expires_in_days=180
        )
        
        assert key is not None
        assert key.key_type == KeyType.DATA_ENCRYPTION
        assert key.status == KeyStatus.ACTIVE
        assert key.encrypted_key_material is not None
        assert key.expires_at is not None
        assert (key.expires_at - datetime.now()).days <= 180
    
    def test_generate_session_key(self, key_manager):
        """Test generating session key."""
        key = key_manager.generate_key(
            key_type=KeyType.SESSION,
            expires_in_days=1
        )
        
        assert key.key_type == KeyType.SESSION
        assert key.status == KeyStatus.ACTIVE
        assert (key.expires_at - datetime.now()).days <= 1
    
    def test_generate_multiple_keys(self, key_manager):
        """Test generating multiple keys of different types."""
        keys = []
        
        for key_type in [KeyType.DATA_ENCRYPTION, KeyType.SESSION, KeyType.API]:
            key = key_manager.generate_key(key_type=key_type)
            keys.append(key)
        
        assert len(keys) == 3
        assert all(k.status == KeyStatus.ACTIVE for k in keys)
        assert len(set(k.key_id for k in keys)) == 3  # All unique


class TestKeyRotation:
    """Tests for key rotation."""
    
    def test_rotate_key(self, key_manager):
        """Test basic key rotation."""
        # Generate initial key
        old_key = key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        old_key_id = old_key.key_id
        
        # Rotate key
        new_key = key_manager.rotate_key(old_key_id, force=True)
        
        assert new_key.key_id != old_key_id
        assert new_key.key_type == old_key.key_type
        assert new_key.rotation_count == 1
        assert new_key.status == KeyStatus.ACTIVE
        
        # Check old key status
        old_key_updated = key_manager.get_key(old_key_id)
        assert old_key_updated.status == KeyStatus.DEPRECATED
    
    def test_rotation_count_increments(self, key_manager):
        """Test rotation count increments correctly."""
        key = key_manager.generate_key(KeyType.SESSION)
        
        for i in range(3):
            key = key_manager.rotate_key(key.key_id, force=True)
            assert key.rotation_count == i + 1
    
    def test_rotation_not_needed(self, key_manager):
        """Test that rotation is skipped when not needed."""
        key = key_manager.generate_key(KeyType.API)
        
        # Try to rotate without force (should skip)
        result = key_manager.rotate_key(key.key_id, force=False)
        
        # Should return same key since rotation not needed
        assert result.key_id == key.key_id


class TestKeyRetrieval:
    """Tests for key retrieval and listing."""
    
    def test_get_key_by_id(self, key_manager):
        """Test retrieving key by ID."""
        key = key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        
        retrieved = key_manager.get_key(key.key_id)
        
        assert retrieved is not None
        assert retrieved.key_id == key.key_id
        assert retrieved.key_type == key.key_type
    
    def test_get_active_key(self, key_manager):
        """Test getting active key of specific type."""
        # Generate multiple keys of same type
        key1 = key_manager.generate_key(KeyType.SESSION)
        key2 = key_manager.generate_key(KeyType.SESSION)
        
        active = key_manager.get_active_key(KeyType.SESSION)
        
        assert active is not None
        assert active.key_type == KeyType.SESSION
        assert active.status == KeyStatus.ACTIVE
        # Should be most recent
        assert active.key_id == key2.key_id
    
    def test_list_keys(self, key_manager):
        """Test listing all keys."""
        # Generate keys of different types
        key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        key_manager.generate_key(KeyType.SESSION)
        key_manager.generate_key(KeyType.API)
        
        all_keys = key_manager.list_keys()
        
        assert len(all_keys) == 3
    
    def test_list_keys_filtered_by_type(self, key_manager):
        """Test listing keys filtered by type."""
        key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        key_manager.generate_key(KeyType.SESSION)
        
        data_keys = key_manager.list_keys(key_type=KeyType.DATA_ENCRYPTION)
        
        assert len(data_keys) == 2
        assert all(k.key_type == KeyType.DATA_ENCRYPTION for k in data_keys)
    
    def test_list_keys_filtered_by_status(self, key_manager):
        """Test listing keys filtered by status."""
        key1 = key_manager.generate_key(KeyType.API)
        key_manager.rotate_key(key1.key_id, force=True)
        
        active_keys = key_manager.list_keys(status=KeyStatus.ACTIVE)
        deprecated_keys = key_manager.list_keys(status=KeyStatus.DEPRECATED)
        
        assert len(active_keys) >= 1
        assert len(deprecated_keys) >= 1


class TestKeyPersistence:
    """Tests for key persistence."""
    
    def test_key_saved_to_storage(self, temp_storage, key_manager):
        """Test that keys are saved to storage."""
        key = key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        
        key_file = Path(temp_storage) / f"{key.key_id}.json"
        assert key_file.exists()
    
    def test_keys_loaded_on_init(self, temp_storage):
        """Test that keys are loaded on initialization."""
        # Create first manager and generate keys
        config = {
            'quantum_algorithm': 'kyber768',
            'kms_enabled': False,
            'key_storage_path': temp_storage
        }
        
        manager1 = create_quantum_key_manager(config)
        key1 = manager1.generate_key(KeyType.SESSION)
        key2 = manager1.generate_key(KeyType.API)
        
        # Create second manager - should load existing keys
        manager2 = create_quantum_key_manager(config)
        
        loaded_keys = manager2.list_keys()
        assert len(loaded_keys) >= 2


class TestStatusReporting:
    """Tests for status reporting."""
    
    def test_get_status_report(self, key_manager):
        """Test getting status report."""
        # Generate some keys
        key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        key_manager.generate_key(KeyType.SESSION)
        
        status = key_manager.get_status_report()
        
        assert 'total_keys' in status
        assert 'active_keys' in status
        assert 'quantum_protected' in status
        assert status['total_keys'] >= 2
        assert status['active_keys'] >= 2
        assert status['quantum_protected'] is True
    
    def test_expiring_keys_in_report(self, key_manager):
        """Test expiring keys appear in report."""
        # Generate key that expires soon
        key = key_manager.generate_key(
            KeyType.API,
            expires_in_days=10
        )
        
        status = key_manager.get_status_report()
        
        assert 'expiring_soon' in status
        # Key should appear as expiring soon (within 30 days)
        expiring_ids = [k['key_id'] for k in status['expiring_soon']]
        assert key.key_id in expiring_ids


class TestAuditLogging:
    """Tests for audit logging."""
    
    def test_audit_log_created(self, key_manager):
        """Test that audit log is created."""
        key_manager.generate_key(KeyType.SESSION)
        
        assert len(key_manager.audit_log) > 0
    
    def test_audit_log_contains_events(self, key_manager):
        """Test audit log contains expected events."""
        key = key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        key_manager.rotate_key(key.key_id, force=True)
        
        events = [entry['event'] for entry in key_manager.audit_log]
        
        assert 'key_generated' in events
        assert 'key_rotated' in events


class TestQuantumProtection:
    """Tests for quantum-safe protection."""
    
    def test_quantum_crypto_available(self, key_manager):
        """Test that quantum crypto is available."""
        assert key_manager.quantum_crypto is not None
    
    def test_hybrid_encryption_used(self, key_manager):
        """Test that hybrid encryption is used for keys."""
        key = key_manager.generate_key(KeyType.DATA_ENCRYPTION)
        
        # Key should have quantum public key
        assert key.quantum_public_key is not None
        assert key.encrypted_key_material is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
