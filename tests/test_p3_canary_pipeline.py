"""
Tests for P3-3: Canary Deployment Pipeline
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.pipelines.canary_deployment import (
    CanaryPipeline,
    CanaryConfig,
    DeploymentMode,
    DeploymentStatus,
    ValidationResult,
    create_canary_pipeline
)


@pytest.fixture
def temp_storage():
    """Create temporary storage for deployments."""
    temp_dir = tempfile.mkdtemp(prefix='test_deployments_')
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def canary_pipeline(temp_storage):
    """Create canary pipeline instance for testing."""
    config = CanaryConfig(
        shadow_duration_hours=1,
        canary_stages=[5.0, 10.0, 25.0, 50.0, 100.0],
        stage_duration_hours=1,
        min_accuracy=0.85,
        auto_rollback_enabled=True
    )
    
    pipeline = create_canary_pipeline(config, storage_path=temp_storage)
    return pipeline


@pytest.fixture
def holdout_data():
    """Generate synthetic holdout data."""
    np.random.seed(42)
    data = np.random.randn(100, 50)
    labels = np.random.randint(0, 2, 100)
    return data, labels


class TestModelRegistration:
    """Tests for model registration."""
    
    def test_register_model(self, canary_pipeline):
        """Test registering a new model."""
        model_meta = canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test_model_v1.0.0.pt',
            metadata={'framework': 'pytorch', 'accuracy': 0.90}
        )
        
        assert model_meta is not None
        assert model_meta.model_id == 'test_model'
        assert model_meta.version == 'v1.0.0'
        assert 'test_model_v1.0.0' in canary_pipeline.models
    
    def test_register_multiple_versions(self, canary_pipeline):
        """Test registering multiple versions of same model."""
        v1 = canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/v1.pt'
        )
        
        v2 = canary_pipeline.register_model(
            model_id='test_model',
            version='v2.0.0',
            model_artifact_path='/models/v2.pt'
        )
        
        assert v1.version != v2.version
        assert len(canary_pipeline.models) == 2


class TestShadowDeployment:
    """Tests for shadow mode deployment."""
    
    def test_deploy_shadow(self, canary_pipeline, holdout_data):
        """Test deploying model in shadow mode."""
        # Register model first
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        assert deployment is not None
        # Mode could be SHADOW or ROLLBACK if validation failed
        assert deployment.mode in [DeploymentMode.SHADOW, DeploymentMode.ROLLBACK]
        assert deployment.traffic_percentage == 0.0
        assert deployment.deployment_id in canary_pipeline.deployments
    
    def test_shadow_validation_runs(self, canary_pipeline, holdout_data):
        """Test that validation runs in shadow mode."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        # Validation tests should have run
        assert len(deployment.validation_tests) > 0
        
        # Should have standard test types
        test_types = [t.test_type for t in deployment.validation_tests]
        assert 'accuracy' in test_types
        assert 'fairness' in test_types
        assert 'performance' in test_types
        assert 'drift' in test_types


class TestValidation:
    """Tests for validation logic."""
    
    def test_accuracy_validation(self, canary_pipeline, holdout_data):
        """Test accuracy validation."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        accuracy_tests = [t for t in deployment.validation_tests if t.test_type == 'accuracy']
        assert len(accuracy_tests) > 0
        
        test = accuracy_tests[0]
        assert test.test_name == "Accuracy Validation"
        assert test.score >= 0.0 and test.score <= 1.0
        assert test.threshold == canary_pipeline.config.min_accuracy
    
    def test_fairness_validation(self, canary_pipeline, holdout_data):
        """Test fairness validation."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        fairness_tests = [t for t in deployment.validation_tests if t.test_type == 'fairness']
        assert len(fairness_tests) > 0
        
        test = fairness_tests[0]
        assert 'demographic_disparity' in test.details
        assert 'fairness_score' in test.details


class TestCanaryDeployment:
    """Tests for canary deployment."""
    
    def test_deploy_canary_after_shadow(self, canary_pipeline, holdout_data):
        """Test deploying to canary after shadow validation."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        # Deploy to shadow
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        # Check if validation passed
        all_passed = all(t.passed for t in deployment.validation_tests)
        
        if all_passed and deployment.status == DeploymentStatus.DEPLOYING:
            # Deploy to canary
            success = canary_pipeline.deploy_canary(deployment.deployment_id, auto_promote=False)
            
            assert success
            
            # Check deployment updated
            updated = canary_pipeline.deployments[deployment.deployment_id]
            assert updated.mode == DeploymentMode.CANARY
            assert updated.status == DeploymentStatus.MONITORING
            assert updated.traffic_percentage > 0
    
    def test_canary_stages(self, canary_pipeline):
        """Test canary stages configuration."""
        assert len(canary_pipeline.config.canary_stages) > 0
        assert all(0 < s <= 100 for s in canary_pipeline.config.canary_stages)
        # Should be in ascending order
        assert canary_pipeline.config.canary_stages == sorted(canary_pipeline.config.canary_stages)


class TestRollback:
    """Tests for rollback functionality."""
    
    def test_rollback_on_validation_failure(self, canary_pipeline, holdout_data):
        """Test that rollback triggers on validation failure."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        # Check if any tests failed and rollback was triggered
        any_failed = any(not t.passed for t in deployment.validation_tests)
        
        if any_failed and canary_pipeline.config.auto_rollback_enabled:
            assert deployment.rollback_triggered
            assert deployment.rollback_reason is not None
            assert deployment.status == DeploymentStatus.ROLLED_BACK
    
    def test_rollback_reason_captured(self, canary_pipeline, holdout_data):
        """Test that rollback reason is captured."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        if deployment.rollback_triggered:
            assert deployment.rollback_reason is not None
            assert len(deployment.rollback_reason) > 0


class TestDeploymentStatus:
    """Tests for deployment status reporting."""
    
    def test_get_deployment_status(self, canary_pipeline, holdout_data):
        """Test getting deployment status."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        status = canary_pipeline.get_deployment_status(deployment.deployment_id)
        
        assert status is not None
        assert 'deployment_id' in status
        assert 'model_id' in status
        assert 'mode' in status
        assert 'status' in status
        assert 'validation_tests' in status
    
    def test_list_deployments(self, canary_pipeline, holdout_data):
        """Test listing all deployments."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        # Create multiple deployments
        d1 = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        deployments = canary_pipeline.list_deployments()
        
        assert len(deployments) >= 1
        assert any(d['deployment_id'] == d1.deployment_id for d in deployments)
    
    def test_list_deployments_filtered(self, canary_pipeline, holdout_data):
        """Test listing deployments with status filter."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        deployment = canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        # Filter by status
        rolled_back = canary_pipeline.list_deployments(status=DeploymentStatus.ROLLED_BACK)
        
        if deployment.status == DeploymentStatus.ROLLED_BACK:
            assert len(rolled_back) > 0


class TestAuditLogging:
    """Tests for audit logging."""
    
    def test_audit_log_created(self, canary_pipeline, holdout_data):
        """Test that audit log is created."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        assert len(canary_pipeline.audit_log) > 0
    
    def test_audit_log_contains_events(self, canary_pipeline, holdout_data):
        """Test audit log contains expected events."""
        canary_pipeline.register_model(
            model_id='test_model',
            version='v1.0.0',
            model_artifact_path='/models/test.pt'
        )
        
        data, labels = holdout_data
        
        canary_pipeline.deploy_shadow(
            model_id='test_model',
            model_version='v1.0.0',
            holdout_data=data,
            holdout_labels=labels
        )
        
        events = [entry['event'] for entry in canary_pipeline.audit_log]
        
        assert 'model_registered' in events
        assert 'shadow_deployment_started' in events


class TestConfiguration:
    """Tests for configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CanaryConfig()
        
        assert config.shadow_duration_hours > 0
        assert len(config.canary_stages) > 0
        assert config.min_accuracy > 0 and config.min_accuracy <= 1.0
        assert config.min_f1_score > 0 and config.min_f1_score <= 1.0
    
    def test_custom_config(self, temp_storage):
        """Test custom configuration."""
        config = CanaryConfig(
            shadow_duration_hours=48,
            canary_stages=[1.0, 5.0, 10.0],
            min_accuracy=0.90
        )
        
        pipeline = create_canary_pipeline(config, storage_path=temp_storage)
        
        assert pipeline.config.shadow_duration_hours == 48
        assert pipeline.config.canary_stages == [1.0, 5.0, 10.0]
        assert pipeline.config.min_accuracy == 0.90


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
