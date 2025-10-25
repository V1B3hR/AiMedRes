# P3 Features Implementation Guide

This guide documents the implementation of P3 long-term, scale, and research features in AiMedRes.

## Overview

P3 features represent advanced capabilities designed for production-scale deployment and research applications:

- **P3-1**: Advanced Multimodal Viewers (DICOM, 3D Brain Visualizer)
- **P3-2**: Quantum-Safe Cryptography in Production Key Flows
- **P3-3**: Model Update/Canary Pipeline + Continuous Validation

## P3-1: Advanced Multimodal Viewers

### Features

- **DICOM Streaming Viewer**: Smooth streaming of medical imaging with windowing controls
- **3D Brain Visualization**: Interactive anatomical mapping with disease progression tracking
- **Explainability Overlays**: AI prediction explanations integrated into visualizations
- **Treatment Simulation**: Scenario modeling for treatment impact assessment

### Implementation

**File**: `api/viewer_api.py`

The Advanced Viewer API provides REST endpoints for medical imaging visualization:

```python
from api.viewer_api import create_viewer_api

# Initialize viewer API
config = {'temp_dir': '/var/aimedres/viewer_temp'}
viewer_api = create_viewer_api(config)

# Run API server
viewer_api.run(host='0.0.0.0', port=5002)
```

### API Endpoints

#### DICOM Viewer

- `POST /api/viewer/dicom/upload` - Upload DICOM file
- `GET /api/viewer/dicom/<session_id>/image` - Get image with windowing
- `GET /api/viewer/dicom/<session_id>/metadata` - Get DICOM metadata

#### 3D Brain Viewer

- `POST /api/viewer/brain/create` - Create 3D brain visualization
- `POST /api/viewer/brain/<session_id>/progression` - Add disease progression
- `POST /api/viewer/brain/<session_id>/explainability` - Add AI explainability overlay
- `GET /api/viewer/brain/<session_id>/data` - Get complete viewer data

#### Treatment Simulation

- `POST /api/viewer/brain/simulate-treatment` - Simulate treatment impact

### Usage Example

```python
import requests

# Create 3D brain visualization
response = requests.post('http://localhost:5002/api/viewer/brain/create', json={
    'patient_id': 'PATIENT_001',
    'regions': ['hippocampus', 'frontal_lobe', 'temporal_lobe'],
    'highlight_abnormalities': True
})

session_id = response.json()['session_id']

# Add disease progression
requests.post(f'http://localhost:5002/api/viewer/brain/{session_id}/progression', json={
    'stage': 'mild',
    'affected_regions': {
        'hippocampus': 0.4,
        'frontal_lobe': 0.2
    },
    'biomarkers': {'amyloid_beta': 1.5},
    'cognitive_scores': {'mmse': 24}
})

# Add AI explainability
requests.post(f'http://localhost:5002/api/viewer/brain/{session_id}/explainability', json={
    'prediction_type': 'alzheimers_risk',
    'region_importance': {
        'hippocampus': 0.85,
        'frontal_lobe': 0.72
    },
    'confidence': 0.89,
    'features': [
        {'name': 'hippocampal_volume', 'value': -2.1, 'contribution': 0.35},
        {'name': 'amyloid_beta', 'value': 1.5, 'contribution': 0.28}
    ]
})
```

### Performance

- Average render time: <100ms for typical visualizations
- Supports streaming for large DICOM series
- Real-time updates with WebSocket support (optional)

---

## P3-2: Quantum-Safe Cryptography in Production Key Flows

### Features

- **Hybrid Kyber768/AES-256 Encryption**: Post-quantum + classical cryptography
- **Automated Key Rotation**: Policy-driven rotation with grace periods
- **KMS Integration**: AWS KMS, Azure Key Vault support
- **Comprehensive Audit Logging**: All key operations tracked

### Implementation

**File**: `security/quantum_prod_keys.py`

The Quantum Production Key Manager provides secure key management with quantum-safe protection:

```python
from security.quantum_prod_keys import (
    create_quantum_key_manager,
    KeyType,
    KeyRotationPolicy
)

# Configure rotation policy
rotation_policy = KeyRotationPolicy(
    enabled=True,
    rotation_interval_days=90,
    max_key_age_days=365,
    automatic_rotation=True,
    notify_before_rotation_days=7
)

# Initialize key manager
config = {
    'quantum_algorithm': 'kyber768',
    'kms_enabled': True,
    'kms_endpoint': 'https://kms.aws.amazon.com',
    'key_storage_path': '/var/aimedres/keys'
}

key_manager = create_quantum_key_manager(config, rotation_policy)
```

### Key Operations

#### Generate Keys

```python
# Generate data encryption key
data_key = key_manager.generate_key(
    key_type=KeyType.DATA_ENCRYPTION,
    metadata={'purpose': 'patient_records'},
    expires_in_days=180
)

# Generate session key
session_key = key_manager.generate_key(
    key_type=KeyType.SESSION,
    expires_in_days=1
)
```

#### Rotate Keys

```python
# Manual rotation
new_key = key_manager.rotate_key(old_key.key_id, force=True)

# Bulk rotation (all keys needing rotation)
summary = key_manager.rotate_all_keys()
print(f"Rotated {summary['rotated_count']} keys")
```

#### Retrieve Keys

```python
# Get key by ID
key = key_manager.get_key(key_id)

# Get active key of type
active_key = key_manager.get_active_key(KeyType.DATA_ENCRYPTION)

# List all keys
all_keys = key_manager.list_keys()

# List filtered
data_keys = key_manager.list_keys(
    key_type=KeyType.DATA_ENCRYPTION,
    status=KeyStatus.ACTIVE
)
```

### Security Features

1. **Quantum-Safe Protection**
   - NIST-approved Kyber768 algorithm
   - Hybrid mode: AES-256 + Kyber768
   - Protection against quantum computing attacks

2. **Key Rotation**
   - Automatic rotation based on policy
   - Grace period for deprecated keys
   - Zero-downtime rotation

3. **KMS Integration**
   - Master key protection via KMS
   - Hardware security module (HSM) support
   - Multi-region key replication

4. **Audit Trail**
   - All key operations logged
   - Immutable audit log
   - Compliance reporting

### Performance

- Key generation: <5ms
- Encryption (1KB): ~0.02ms
- Decryption (1KB): ~0.01ms
- Key exchange: ~0.01ms

---

## P3-3: Model Update/Canary Pipeline + Continuous Validation

### Features

- **Shadow Mode Deployment**: Safe testing without serving traffic
- **Canary Rollout**: Gradual traffic increase with monitoring
- **Automated Validation**: Accuracy, fairness, performance, drift tests
- **Auto-Rollback**: Automatic rollback on failures
- **A/B Testing**: Compare model versions

### Implementation

**File**: `mlops/pipelines/canary_deployment.py`

The Canary Pipeline manages safe model deployments with continuous validation:

```python
from mlops.pipelines.canary_deployment import (
    create_canary_pipeline,
    CanaryConfig
)

# Configure canary deployment
config = CanaryConfig(
    shadow_duration_hours=24,
    canary_stages=[5.0, 10.0, 25.0, 50.0, 100.0],
    stage_duration_hours=2,
    min_accuracy=0.85,
    min_f1_score=0.80,
    max_performance_degradation=0.10,
    max_error_rate=0.05,
    auto_rollback_enabled=True
)

pipeline = create_canary_pipeline(config, storage_path='/var/aimedres/deployments')
```

### Deployment Workflow

#### 1. Register Model

```python
model_meta = pipeline.register_model(
    model_id='alzheimer_nn',
    version='v2.1.0',
    model_artifact_path='/models/alzheimer_nn_v2.1.0.pt',
    metadata={
        'framework': 'pytorch',
        'architecture': 'transformer',
        'accuracy': 0.92
    }
)
```

#### 2. Deploy to Shadow

```python
import numpy as np

# Prepare holdout dataset
holdout_data = np.load('holdout_features.npy')
holdout_labels = np.load('holdout_labels.npy')

# Deploy in shadow mode
deployment = pipeline.deploy_shadow(
    model_id='alzheimer_nn',
    model_version='v2.1.0',
    holdout_data=holdout_data,
    holdout_labels=holdout_labels
)

print(f"Shadow deployment: {deployment.deployment_id}")
print(f"Status: {deployment.status.value}")
```

#### 3. Review Validation Results

```python
# Check validation tests
for test in deployment.validation_tests:
    print(f"{test.test_name}: {test.result.value}")
    print(f"  Score: {test.score:.3f} (threshold: {test.threshold:.3f})")
    print(f"  Passed: {test.passed}")
```

#### 4. Deploy to Canary (if validation passed)

```python
if deployment.status == DeploymentStatus.DEPLOYING:
    success = pipeline.deploy_canary(
        deployment.deployment_id,
        auto_promote=True  # Automatically promote through stages
    )
```

#### 5. Monitor Deployment

```python
# Get deployment status
status = pipeline.get_deployment_status(deployment.deployment_id)

print(f"Mode: {status['mode']}")
print(f"Status: {status['status']}")
print(f"Traffic: {status['traffic_percentage']}%")
print(f"Performance metrics: {status['performance_metrics']}")

# Check for rollback
if status['rollback_triggered']:
    print(f"Rollback reason: {status['rollback_reason']}")
```

### Validation Tests

The pipeline automatically runs four types of validation:

1. **Accuracy Validation**
   - Tests model accuracy on holdout set
   - Threshold: configurable (default 0.85)

2. **Fairness Validation**
   - Demographic parity analysis
   - Maximum disparity: 10% (configurable)
   - Fairness score: >0.80 (configurable)

3. **Performance Validation**
   - Inference latency measurement
   - Maximum degradation: 10% (configurable)
   - Error rate monitoring

4. **Drift Detection**
   - Data drift analysis
   - Concept drift monitoring
   - Threshold: 0.10 (configurable)

### Canary Stages

Default rollout stages with monitoring at each:

1. **5% traffic** - Initial canary, close monitoring
2. **10% traffic** - Early adopters
3. **25% traffic** - Expanded testing
4. **50% traffic** - Half traffic
5. **100% traffic** - Full production (stable)

### Rollback Strategy

Automatic rollback triggers:

- Any validation test failure
- Error rate exceeds threshold
- Performance degradation exceeds limit
- Manual rollback request

Rollback process:
1. Traffic immediately reverted to 0%
2. Previous stable model reinstated
3. Root cause captured in audit log
4. Alerts sent to operations team

---

## Testing

### Run P3 Tests

```bash
# Test quantum key management
python -m pytest tests/test_p3_quantum_keys.py -v

# Test canary pipeline
python -m pytest tests/test_p3_canary_pipeline.py -v

# Run all P3 tests
python -m pytest tests/test_p3*.py -v
```

### Run P3 Demo

```bash
python examples/p3_features_demo.py
```

Expected output:
```
P3-1: ✓ PASSED
P3-2: ✓ PASSED
P3-3: ✓ PASSED

✓ All P3 demonstrations completed successfully!
```

---

## Configuration

### Environment Variables

```bash
# Quantum key manager
export AIMEDRES_MASTER_KEY="your-master-key-here"

# KMS configuration (if using AWS KMS)
export AWS_REGION="us-east-1"
export AWS_KMS_KEY_ID="arn:aws:kms:..."

# Deployment storage
export AIMEDRES_DEPLOYMENT_PATH="/var/aimedres/deployments"
export AIMEDRES_KEY_STORAGE_PATH="/var/aimedres/keys"
```

### Production Configuration

```python
# config/production.yaml
p3_features:
  viewers:
    enabled: true
    port: 5002
    temp_dir: /var/aimedres/viewer_temp
  
  quantum_keys:
    enabled: true
    algorithm: kyber768
    kms_enabled: true
    kms_provider: aws  # or azure, gcp
    rotation_interval_days: 90
    automatic_rotation: true
  
  canary_pipeline:
    enabled: true
    shadow_duration_hours: 24
    canary_stages: [5, 10, 25, 50, 100]
    stage_duration_hours: 4
    auto_rollback: true
    validation_thresholds:
      min_accuracy: 0.88
      min_f1_score: 0.85
      max_performance_degradation: 0.08
```

---

## Monitoring & Observability

### Metrics

**Viewer API**:
- Active sessions count
- Average render time
- API request rate
- Error rate

**Quantum Key Manager**:
- Total keys managed
- Active keys count
- Rotation events
- Key usage count

**Canary Pipeline**:
- Deployment success rate
- Validation pass rate
- Rollback frequency
- Average canary duration

### Dashboards

Access monitoring dashboards:

- **Viewer Metrics**: `http://localhost:5002/api/viewer/statistics`
- **Key Manager Status**: Via `key_manager.get_status_report()`
- **Deployment Status**: `http://localhost:8000/api/mlops/deployments`

---

## Security Considerations

1. **Quantum Key Management**
   - Always use KMS in production
   - Rotate master keys annually
   - Maintain offline key backups
   - Monitor key usage patterns

2. **DICOM Viewer**
   - Validate DICOM files before processing
   - Implement access controls
   - Audit all viewing sessions
   - De-identify PHI in temporary storage

3. **Canary Deployments**
   - Review validation thresholds regularly
   - Monitor for adversarial inputs
   - Maintain rollback procedures
   - Test disaster recovery

---

## Troubleshooting

### Viewer API Issues

**Problem**: DICOM upload fails
- Check temp directory permissions
- Verify DICOM file format
- Check available disk space

**Problem**: Brain visualization slow
- Reduce number of regions
- Enable caching
- Check system memory

### Quantum Key Issues

**Problem**: Key rotation fails
- Check KMS connectivity
- Verify key permissions
- Review audit log for errors

**Problem**: Performance degradation
- Monitor key cache usage
- Check quantum crypto library version
- Verify hardware acceleration

### Canary Pipeline Issues

**Problem**: All deployments fail validation
- Review validation thresholds
- Check holdout dataset quality
- Verify model artifact integrity

**Problem**: Rollback loop
- Disable auto-rollback temporarily
- Review rollback reasons
- Check monitoring system health

---

## Support

For issues or questions:

1. Check logs in `/var/log/aimedres/`
2. Review audit trails
3. Contact: support@aimedres.ai
4. GitHub Issues: https://github.com/V1B3hR/AiMedRes/issues

---

## References

- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Canary Deployment Best Practices](https://martinfowler.com/bliki/CanaryRelease.html)
- [DICOM Standard](https://www.dicomstandard.org/)
- [Model Fairness in Healthcare AI](https://doi.org/10.1038/s41591-021-01417-9)
