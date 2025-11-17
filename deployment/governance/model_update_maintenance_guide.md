# Model Update & Maintenance Guide

## Overview

This guide establishes procedures for ongoing AI model performance tracking, drift monitoring, safe updates, version control, and rollback procedures for the AiMedRes healthcare AI platform.

## Table of Contents

1. [Model Performance Tracking](#model-performance-tracking)
2. [Drift Monitoring](#drift-monitoring)
3. [Model Update Procedures](#model-update-procedures)
4. [Version Control & Rollback](#version-control--rollback)
5. [Real-World Validation](#real-world-validation)
6. [Model Governance](#model-governance)

---

## Model Performance Tracking

### Continuous Performance Monitoring

AiMedRes implements continuous model performance monitoring using the existing MLOps infrastructure.

#### Key Performance Indicators (KPIs)

**For Classification Models (Alzheimer's, ALS):**

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| Accuracy | ≥ 0.85 | < 0.83 | < 0.80 |
| Sensitivity (Recall) | ≥ 0.88 | < 0.85 | < 0.82 |
| Specificity | ≥ 0.84 | < 0.81 | < 0.78 |
| AUC-ROC | ≥ 0.90 | < 0.88 | < 0.85 |
| Precision | ≥ 0.86 | < 0.83 | < 0.80 |
| F1 Score | ≥ 0.87 | < 0.84 | < 0.81 |

**For Regression Models (Parkinson's):**

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| R² Score | ≥ 0.80 | < 0.77 | < 0.73 |
| MAE | ≤ 0.13 | > 0.16 | > 0.20 |
| MSE | ≤ 0.17 | > 0.21 | > 0.26 |
| RMSE | ≤ 0.41 | > 0.46 | > 0.51 |

**Operational Metrics:**

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Inference Latency (p95) | < 300ms | > 400ms | > 500ms |
| Inference Latency (p99) | < 500ms | > 650ms | > 800ms |
| Error Rate | < 0.5% | > 1% | > 2% |
| Throughput | > 100 req/s | < 80 req/s | < 50 req/s |
| Model Load Time | < 5s | > 8s | > 12s |

#### Implementation

**Using Existing Production Monitor:**

```python
from mlops.monitoring.production_monitor import ProductionMonitor

# Initialize monitor for each deployed model
alzheimer_monitor = ProductionMonitor(
    model_name='alzheimer_v1',
    model_type='classification',
    performance_thresholds={
        'accuracy': {'target': 0.85, 'warning': 0.83, 'critical': 0.80},
        'sensitivity': {'target': 0.88, 'warning': 0.85, 'critical': 0.82},
        'auc_roc': {'target': 0.90, 'warning': 0.88, 'critical': 0.85}
    },
    alert_channels=['email', 'slack', 'pagerduty']
)

# Monitor runs automatically on each prediction
prediction = alzheimer_monitor.predict(patient_data)

# Manual performance check
performance_report = alzheimer_monitor.get_performance_report(
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

**Automated Performance Tracking:**

```bash
#!/bin/bash
# Script: track_model_performance.sh
# Schedule: Daily at 3 AM via cron

cd /opt/aimedres

# Generate performance reports for all models
for model in alzheimer_v1 parkinsons_v1 als_v1; do
    python3 -c "
from mlops.monitoring.production_monitor import ProductionMonitor
from datetime import datetime, timedelta

monitor = ProductionMonitor(model_name='$model')

# Get yesterday's performance
yesterday = datetime.now() - timedelta(days=1)
report = monitor.get_performance_report(
    start_date=yesterday.strftime('%Y-%m-%d'),
    end_date=yesterday.strftime('%Y-%m-%d')
)

print(f'Model: $model')
print(f'Predictions: {report[\"prediction_count\"]}')
print(f'Avg Latency: {report[\"avg_latency_ms\"]}ms')
print(f'Error Rate: {report[\"error_rate\"]*100:.2f}%')
print('---')
" >> /var/log/aimedres/performance_tracking.log
done

# Check for threshold violations
python3 /opt/aimedres/scripts/check_performance_thresholds.py
```

### Performance Dashboards

**Grafana Dashboard Configuration:**

Pre-configured dashboards available in `deployment/production_deployment/grafana_dashboard.json`:

1. **Model Performance Overview**
   - Prediction volume over time
   - Accuracy trends (7-day, 30-day moving average)
   - Latency percentiles (p50, p95, p99)
   - Error rates by model

2. **Model Comparison**
   - Side-by-side performance metrics
   - Version comparison (current vs previous)
   - A/B test results (if canary deployment active)

3. **Operational Metrics**
   - Request rate and throughput
   - Resource utilization (CPU, memory, GPU)
   - Queue depth and backlog
   - Cache hit rates

**Access Dashboards:**
- URL: https://grafana.hospital.org/d/aimedres-models
- Update Frequency: Real-time (15-second refresh)
- Historical Data: 90 days retained

### Periodic Re-Benchmarking

**Schedule:** Quarterly (every 3 months)

**Benchmark Process:**

```bash
#!/bin/bash
# Script: quarterly_model_benchmark.sh

BENCHMARK_DATE=$(date +%Y%m%d)
BENCHMARK_DIR="/var/aimedres/benchmarks/$BENCHMARK_DATE"
mkdir -p "$BENCHMARK_DIR"

# 1. Prepare holdout validation dataset
python3 /opt/aimedres/scripts/prepare_validation_dataset.py \
    --output "$BENCHMARK_DIR/validation_data.csv" \
    --samples 1000 \
    --stratified

# 2. Run benchmark for each model
for model in alzheimer_v1 parkinsons_v1 als_v1; do
    echo "Benchmarking $model..."
    
    python3 /opt/aimedres/scripts/benchmark_model.py \
        --model "$model" \
        --validation-data "$BENCHMARK_DIR/validation_data.csv" \
        --output "$BENCHMARK_DIR/${model}_benchmark.json"
done

# 3. Generate comparison report
python3 /opt/aimedres/scripts/generate_benchmark_report.py \
    --benchmark-dir "$BENCHMARK_DIR" \
    --baseline-dir "/var/aimedres/benchmarks/baseline" \
    --output "$BENCHMARK_DIR/benchmark_report.pdf"

# 4. Send report to stakeholders
mail -s "AiMedRes Quarterly Benchmark Report - $BENCHMARK_DATE" \
    -a "$BENCHMARK_DIR/benchmark_report.pdf" \
    clinical-leads@hospital.org,ml-team@hospital.org < /dev/null
```

**Benchmark Validation Dataset:**
- Size: 1,000+ samples per model
- Source: Recent production data (last 90 days)
- De-identified and PHI-scrubbed
- Stratified sampling to ensure class balance
- Ground truth from clinical validation

**Benchmark Acceptance Criteria:**
- Performance within 5% of baseline
- No degradation in critical metrics (sensitivity, specificity)
- Latency within acceptable range
- No increase in error rate

---

## Drift Monitoring

### Types of Drift

AiMedRes monitors three types of drift:

1. **Data Drift:** Changes in input feature distributions
2. **Concept Drift:** Changes in relationship between features and target
3. **Prediction Drift:** Changes in model output distributions

### Drift Detection Implementation

**Using Existing Enhanced Drift Monitoring:**

```python
from src.aimedres.training.enhanced_drift_monitoring import EnhancedDriftMonitor

# Initialize drift monitor
drift_monitor = EnhancedDriftMonitor(
    model_name='alzheimer_v1',
    reference_data_path='/var/aimedres/models/alzheimer_v1/training_data.csv',
    drift_threshold=0.10,  # 10% KL divergence threshold
    alert_channels=['email', 'slack']
)

# Monitor runs automatically on predictions
# Manual drift check
drift_report = drift_monitor.check_drift(
    current_data_path='/var/aimedres/data/recent_predictions.csv'
)

if drift_report['drift_detected']:
    print(f"DRIFT ALERT: {drift_report['drifted_features']}")
    print(f"Drift Score: {drift_report['drift_score']:.3f}")
```

### Drift Monitoring Schedule

**Real-time Monitoring:**
- Continuous sampling of production predictions (10% sample rate)
- Statistical tests run hourly
- Alerts triggered immediately on threshold breach

**Daily Drift Analysis:**
```bash
#!/bin/bash
# Script: daily_drift_check.sh
# Schedule: Daily at 4 AM via cron

cd /opt/aimedres

for model in alzheimer_v1 parkinsons_v1 als_v1; do
    echo "Checking drift for $model..."
    
    python3 -c "
from src.aimedres.training.enhanced_drift_monitoring import EnhancedDriftMonitor
from datetime import datetime, timedelta

monitor = EnhancedDriftMonitor(model_name='$model')

# Get yesterday's predictions
yesterday = datetime.now() - timedelta(days=1)
drift_report = monitor.check_drift_for_date(yesterday.strftime('%Y-%m-%d'))

if drift_report['drift_detected']:
    print(f'DRIFT DETECTED in $model!')
    print(f'Drifted features: {drift_report[\"drifted_features\"]}')
    print(f'Drift score: {drift_report[\"drift_score\"]:.3f}')
    
    # Send alert
    monitor.send_drift_alert(drift_report)
"
done
```

**Weekly Drift Summary:**
- Aggregated drift metrics for the week
- Trend analysis and visualization
- Recommendation for model retraining if needed

### Drift Thresholds and Actions

| Drift Level | Threshold | Action | Timeline |
|-------------|-----------|--------|----------|
| **Minimal** | < 5% | Continue monitoring | None |
| **Low** | 5-10% | Increase monitoring frequency | 1 week |
| **Moderate** | 10-20% | Investigate root cause, plan retraining | 2 weeks |
| **High** | 20-30% | Urgent retraining required | 1 week |
| **Critical** | > 30% | Consider model deactivation, emergency retraining | 3 days |

**Drift Response Procedure:**

```python
def handle_drift_alert(model_name, drift_score, drifted_features):
    """Handle drift alert based on severity."""
    
    if drift_score < 0.10:
        # Low drift - continue monitoring
        action = "continue_monitoring"
        priority = "low"
    elif drift_score < 0.20:
        # Moderate drift - investigate
        action = "investigate_and_plan_retraining"
        priority = "medium"
    elif drift_score < 0.30:
        # High drift - urgent retraining
        action = "urgent_retraining"
        priority = "high"
    else:
        # Critical drift - consider deactivation
        action = "consider_deactivation"
        priority = "critical"
    
    # Create incident ticket
    create_incident_ticket(
        title=f"Model Drift Detected: {model_name}",
        description=f"Drift score: {drift_score:.3f}\nDrifted features: {drifted_features}",
        action=action,
        priority=priority
    )
    
    # Send notifications
    send_alert(
        channels=['email', 'slack'],
        severity=priority,
        message=f"Drift detected in {model_name}: {drift_score:.1%}"
    )
    
    return action
```

### Root Cause Analysis for Drift

When drift is detected, perform root cause analysis:

1. **Data Quality Issues:**
   - Check for missing values, outliers, data entry errors
   - Verify data pipeline integrity
   - Review recent EMR/EHR integration changes

2. **Population Changes:**
   - Analyze patient demographics
   - Check for seasonal variations
   - Review hospital admission patterns

3. **Clinical Practice Changes:**
   - New treatment protocols
   - Updated diagnostic criteria
   - Changes in referral patterns

4. **Technical Issues:**
   - Feature engineering pipeline changes
   - Data preprocessing modifications
   - Infrastructure updates

**Root Cause Analysis Template:**

```markdown
## Drift Root Cause Analysis

**Model:** [model_name]
**Drift Score:** [drift_score]
**Detection Date:** [date]

### Drifted Features
- Feature 1: [description, drift magnitude]
- Feature 2: [description, drift magnitude]

### Investigation Findings
1. Data Quality: [findings]
2. Population Changes: [findings]
3. Clinical Changes: [findings]
4. Technical Issues: [findings]

### Root Cause
[Identified root cause]

### Recommended Action
[ ] Continue monitoring
[ ] Retrain with recent data
[ ] Update feature engineering
[ ] Modify data pipeline
[ ] Clinical workflow change

### Timeline
Start Date: [date]
Target Completion: [date]
Assigned To: [name]
```

---

## Model Update Procedures

### Update Triggers

Models should be updated when:

1. **Performance Degradation:** Metrics below warning thresholds for 7+ days
2. **Significant Drift:** Drift score > 20% sustained for 2+ weeks
3. **New Data Available:** Substantial new validated data (1000+ samples)
4. **Clinical Guideline Updates:** Changes in diagnostic/treatment standards
5. **Security/Bug Fixes:** Critical issues requiring immediate update
6. **Scheduled Updates:** Quarterly planned retraining (best practice)

### Safe Update Process

**Phase 1: Development and Training (2-4 weeks)**

1. **Prepare Training Data:**
   ```bash
   # Collect recent production data
   python3 collect_training_data.py \
       --start-date 2024-01-01 \
       --end-date 2024-03-31 \
       --output training_data_v2.csv \
       --phi-scrubbed \
       --validated-labels
   
   # Augment with historical data
   python3 merge_datasets.py \
       --historical-data training_data_v1.csv \
       --recent-data training_data_v2.csv \
       --output training_data_v2_combined.csv
   ```

2. **Train New Model Version:**
   ```bash
   # Train with existing pipeline
   aimedres train \
       --model alzheimer \
       --data training_data_v2_combined.csv \
       --output-dir models/alzheimer_v2 \
       --config configs/alzheimer_training.yaml
   ```

3. **Validation and Benchmarking:**
   ```bash
   # Benchmark against holdout dataset
   python3 benchmark_model.py \
       --model models/alzheimer_v2 \
       --validation-data validation_holdout.csv \
       --baseline models/alzheimer_v1 \
       --output benchmark_v2_vs_v1.json
   ```

4. **Clinical Validation:**
   - Review sample predictions with clinical team
   - Validate against known cases
   - Assess clinical utility and safety
   - Document validation findings

**Phase 2: Pre-Production Testing (1 week)**

1. **Deploy to Staging Environment:**
   ```bash
   # Deploy to staging
   kubectl apply -f k8s/staging/alzheimer_v2_deployment.yaml
   
   # Run integration tests
   pytest tests/integration/test_alzheimer_v2.py
   ```

2. **Smoke Testing:**
   ```bash
   # Automated smoke tests
   python3 smoke_test_model.py \
       --model alzheimer_v2 \
       --env staging \
       --test-cases test_data/smoke_test_cases.json
   ```

3. **Load Testing:**
   ```bash
   # Stress test with realistic load
   locust -f locustfile.py \
       --host https://staging.aimedres.hospital.org \
       --users 100 \
       --spawn-rate 10 \
       --run-time 1h
   ```

4. **Security Scanning:**
   ```bash
   # Scan model artifact for vulnerabilities
   python3 scan_model_security.py \
       --model-path models/alzheimer_v2
   ```

**Phase 3: Canary Deployment (1-2 weeks)**

```python
from mlops.pipelines.canary_deployment import CanaryDeployment

# Initialize canary deployment
canary = CanaryDeployment(
    new_model='alzheimer_v2',
    baseline_model='alzheimer_v1',
    traffic_split_strategy='gradual',  # 5% → 10% → 25% → 50% → 100%
    validation_metrics=['accuracy', 'latency', 'error_rate'],
    validation_thresholds={
        'accuracy_drop': 0.02,  # Max 2% accuracy drop
        'latency_increase': 1.5,  # Max 50% latency increase
        'error_rate': 0.01  # Max 1% error rate
    },
    rollback_on_failure=True
)

# Execute canary deployment
canary.deploy()

# Monitor automatically handles traffic progression and rollback
```

**Phase 4: Full Production Rollout (1-2 days)**

```bash
# After successful canary validation
kubectl set image deployment/aimedres-alzheimer \
    aimedres-alzheimer=aimedres:alzheimer_v2

# Verify rollout
kubectl rollout status deployment/aimedres-alzheimer

# Monitor closely for 24-48 hours
python3 monitor_deployment.py \
    --model alzheimer_v2 \
    --duration 48h \
    --alert-on-issues
```

**Phase 5: Post-Deployment Validation (1 week)**

1. **Performance Monitoring:**
   - Compare v2 vs v1 performance metrics
   - Validate latency and throughput
   - Monitor error rates

2. **Clinical Feedback:**
   - Collect clinician feedback on predictions
   - Review any reported issues
   - Assess clinical utility

3. **Documentation:**
   - Update model cards and documentation
   - Document changes and improvements
   - Archive previous version

### Update Approval Process

**Required Approvals:**

1. **Technical Review (ML Team Lead):**
   - [ ] Training metrics meet targets
   - [ ] Validation performance acceptable
   - [ ] No regressions in key metrics
   - [ ] Code review completed
   - [ ] Security scan passed

2. **Clinical Review (Clinical Champion):**
   - [ ] Clinical validation completed
   - [ ] Sample predictions reviewed
   - [ ] No safety concerns identified
   - [ ] Clinical utility maintained or improved

3. **Compliance Review (Compliance Officer):**
   - [ ] Audit trail documented
   - [ ] Regulatory requirements met (if applicable)
   - [ ] Risk assessment completed
   - [ ] Privacy controls verified

4. **Final Approval (IT Director / CISO):**
   - [ ] All reviews completed
   - [ ] Rollback plan documented
   - [ ] Deployment schedule approved
   - [ ] Communication plan ready

**Approval Documentation Template:**

```markdown
## Model Update Approval: [model_name] v[version]

### Update Summary
- Current Version: [current_version]
- New Version: [new_version]
- Update Reason: [reason]
- Training Date: [date]
- Approval Date: [date]

### Performance Metrics
| Metric | v[old] | v[new] | Change |
|--------|--------|--------|--------|
| Accuracy | X.XX | X.XX | ±X.X% |
| Sensitivity | X.XX | X.XX | ±X.X% |
| Specificity | X.XX | X.XX | ±X.X% |

### Approvals
- [x] Technical Review: [Name] - [Date]
- [x] Clinical Review: [Name] - [Date]
- [x] Compliance Review: [Name] - [Date]
- [x] Final Approval: [Name] - [Date]

### Deployment Plan
- Staging Date: [date]
- Canary Start: [date]
- Full Rollout: [date]
- Rollback Deadline: [date]

### Signatures
Technical Lead: ___________________ Date: ___________
Clinical Champion: ________________ Date: ___________
Compliance Officer: _______________ Date: ___________
IT Director: ______________________ Date: ___________
```

---

## Version Control & Rollback

### Model Versioning Strategy

**Version Numbering:** `[major].[minor].[patch]`

- **Major (X.0.0):** Significant architecture changes, new features
- **Minor (0.X.0):** Retraining with new data, hyperparameter tuning
- **Patch (0.0.X):** Bug fixes, minor adjustments

**Version Metadata:**

```json
{
  "model_name": "alzheimer",
  "version": "1.2.3",
  "created_date": "2024-01-15T10:30:00Z",
  "training_data": {
    "source": "training_data_v1.2.csv",
    "samples": 5000,
    "date_range": "2020-01-01 to 2023-12-31"
  },
  "metrics": {
    "accuracy": 0.89,
    "sensitivity": 0.92,
    "specificity": 0.87,
    "auc_roc": 0.93
  },
  "framework": "scikit-learn==1.3.0",
  "dependencies": ["numpy==1.24.0", "pandas==2.0.0"],
  "training_config": "configs/alzheimer_v1.2.yaml",
  "approvals": {
    "technical": "john.doe@hospital.org",
    "clinical": "dr.smith@hospital.org",
    "compliance": "jane.compliance@hospital.org"
  },
  "deployment_date": "2024-01-20T09:00:00Z",
  "status": "production"
}
```

### Model Registry

AiMedRes uses a centralized model registry to track all model versions:

```python
from mlops.model_registry import ModelRegistry

registry = ModelRegistry(storage_path='/var/aimedres/models')

# Register new model version
registry.register_model(
    name='alzheimer',
    version='1.2.3',
    model_path='models/alzheimer_v1.2.3',
    metadata={
        'accuracy': 0.89,
        'training_samples': 5000,
        'approved_by': ['john.doe', 'dr.smith']
    }
)

# List all versions
versions = registry.list_versions('alzheimer')

# Get specific version
model_v1 = registry.get_model('alzheimer', version='1.0.0')
```

**Model Storage Structure:**

```
/var/aimedres/models/
├── alzheimer/
│   ├── v1.0.0/
│   │   ├── model.pkl
│   │   ├── metadata.json
│   │   ├── training_config.yaml
│   │   ├── feature_schema.json
│   │   └── model_card.md
│   ├── v1.1.0/
│   │   ├── model.pkl
│   │   ├── metadata.json
│   │   └── ...
│   └── v1.2.0/ (current production)
│       ├── model.pkl
│       ├── metadata.json
│       └── ...
├── parkinsons/
│   └── v1.0.0/
└── als/
    └── v1.0.0/
```

### Rollback Procedures

**Rollback Triggers:**

- Performance degradation > 5% in critical metrics
- Error rate > 2%
- Clinical safety concerns reported
- System instability or crashes
- Data integrity issues

**Automated Rollback (Canary Deployment):**

If using canary deployment, rollback is automatic on validation failures:

```python
# Canary deployment with automatic rollback
canary = CanaryDeployment(
    new_model='alzheimer_v2',
    baseline_model='alzheimer_v1',
    rollback_on_failure=True,  # Automatic rollback enabled
    validation_thresholds={
        'accuracy_drop': 0.02,
        'latency_increase': 1.5,
        'error_rate': 0.01
    }
)

# If thresholds violated, automatically rolls back to v1
canary.deploy()
```

**Manual Rollback:**

**Method 1: Quick Rollback (Docker/Kubernetes)**

```bash
#!/bin/bash
# Script: rollback_model.sh

MODEL_NAME=$1
PREVIOUS_VERSION=$2

echo "Rolling back $MODEL_NAME to $PREVIOUS_VERSION..."

# Kubernetes rollback
kubectl rollout undo deployment/aimedres-$MODEL_NAME

# Or rollback to specific version
kubectl rollout undo deployment/aimedres-$MODEL_NAME --to-revision=$PREVIOUS_VERSION

# Verify rollback
kubectl rollout status deployment/aimedres-$MODEL_NAME

# Update model registry
python3 -c "
from mlops.model_registry import ModelRegistry
registry = ModelRegistry()
registry.set_production_model('$MODEL_NAME', '$PREVIOUS_VERSION')
registry.add_note('$MODEL_NAME', '$PREVIOUS_VERSION', 'Rolled back due to issues with newer version')
"

echo "Rollback complete. Monitoring for 1 hour..."
python3 monitor_deployment.py --model $MODEL_NAME --duration 1h
```

**Method 2: Database/Config Rollback**

```python
# Update model configuration to point to previous version
from src.aimedres.core.config import update_model_config

update_model_config(
    model_name='alzheimer',
    config_updates={
        'version': '1.1.0',  # Rollback to previous version
        'model_path': '/var/aimedres/models/alzheimer/v1.1.0/model.pkl',
        'reason': 'Performance degradation in v1.2.0',
        'rolled_back_by': 'ops-team@hospital.org',
        'rollback_date': datetime.now().isoformat()
    }
)

# Restart application to load previous version
restart_aimedres_service()
```

**Rollback Validation:**

```bash
# After rollback, verify system health
python3 validate_rollback.py --model alzheimer --expected-version 1.1.0

# Checks:
# 1. Model version loaded correctly
# 2. Prediction endpoint responding
# 3. Sample predictions match expected outputs
# 4. Performance metrics stabilized
# 5. No errors in logs
```

**Post-Rollback Actions:**

1. **Incident Report:**
   - Document reason for rollback
   - Analyze root cause of issues
   - Identify preventive measures

2. **Stakeholder Communication:**
   - Notify clinical team of rollback
   - Explain reason and timeline
   - Provide updates on fix

3. **Fix and Re-deploy:**
   - Address issues in development
   - Re-validate and test thoroughly
   - Follow full update procedure for re-deployment

### Rollback Testing

**Quarterly Rollback Drill:**

```bash
#!/bin/bash
# Script: rollback_drill.sh
# Purpose: Test rollback procedures in non-production environment

echo "=== Rollback Drill ==="
echo "Date: $(date)"

# 1. Deploy test version
echo "Deploying test version..."
kubectl apply -f k8s/test/rollback-drill-deployment.yaml

# 2. Wait for deployment
kubectl wait --for=condition=available deployment/rollback-drill --timeout=5m

# 3. Execute rollback
echo "Executing rollback..."
kubectl rollout undo deployment/rollback-drill

# 4. Verify rollback
echo "Verifying rollback..."
kubectl rollout status deployment/rollback-drill

# 5. Check functionality
python3 test_rollback_functionality.py --deployment rollback-drill

# 6. Measure rollback time
echo "Rollback drill complete."
echo "Total time: ${SECONDS}s"
echo "Target: < 5 minutes"

# Clean up
kubectl delete -f k8s/test/rollback-drill-deployment.yaml
```

---

## Real-World Validation

### Continuous Real-World Performance Monitoring

**Prospective Validation:**

Monitor model performance on real clinical use in production:

```python
from src.aimedres.training.model_validation import RealWorldValidator

validator = RealWorldValidator(
    model_name='alzheimer_v1',
    validation_frequency='daily',
    sample_rate=0.10  # Validate 10% of predictions
)

# Collect ground truth from clinical follow-up
validator.collect_ground_truth(
    prediction_id='pred_12345',
    ground_truth_label='positive',
    validated_by='dr_smith@hospital.org',
    validation_date='2024-01-20',
    notes='Clinical diagnosis confirmed via additional testing'
)

# Generate validation report
report = validator.generate_validation_report(
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

**Validation Data Collection:**

1. **Clinical Follow-up:**
   - Request clinicians validate subset of predictions
   - Compare AI prediction to clinical diagnosis
   - Document agreement/disagreement

2. **Outcome Tracking:**
   - Track patient outcomes over time
   - Assess prediction accuracy retrospectively
   - Correlate predictions with actual disease progression

3. **Incident Reporting:**
   - Report any incorrect predictions
   - Analyze near-misses and false results
   - Document lessons learned

### Re-validation Triggers

Models should be re-validated when:

1. **Performance Concerns:** Anecdotal reports of incorrect predictions
2. **Population Changes:** Significant demographic or clinical shifts
3. **Regular Schedule:** Annual comprehensive re-validation
4. **Post-Update:** After every model update
5. **Regulatory Requirement:** As required by FDA or other regulators

### Re-validation Process

**Annual Re-validation (Comprehensive):**

```bash
#!/bin/bash
# Script: annual_model_revalidation.sh

VALIDATION_YEAR=$(date +%Y)
VALIDATION_DIR="/var/aimedres/validations/$VALIDATION_YEAR"
mkdir -p "$VALIDATION_DIR"

echo "Starting annual model re-validation for $VALIDATION_YEAR"

# 1. Collect production data from past year
python3 collect_production_data.py \
    --start-date "$VALIDATION_YEAR-01-01" \
    --end-date "$VALIDATION_YEAR-12-31" \
    --output "$VALIDATION_DIR/production_data.csv"

# 2. Collect ground truth labels
python3 collect_ground_truth.py \
    --production-data "$VALIDATION_DIR/production_data.csv" \
    --output "$VALIDATION_DIR/ground_truth.csv" \
    --validated-only

# 3. Re-calculate performance metrics
python3 revalidate_model.py \
    --model alzheimer_v1 \
    --predictions "$VALIDATION_DIR/production_data.csv" \
    --ground-truth "$VALIDATION_DIR/ground_truth.csv" \
    --output "$VALIDATION_DIR/revalidation_report.json"

# 4. Compare with original validation
python3 compare_validation_results.py \
    --original deployment/validation/original_validation.json \
    --current "$VALIDATION_DIR/revalidation_report.json" \
    --output "$VALIDATION_DIR/validation_comparison.pdf"

# 5. Generate executive summary
python3 generate_validation_summary.py \
    --validation-dir "$VALIDATION_DIR" \
    --output "$VALIDATION_DIR/annual_validation_summary.pdf"

# 6. Send to stakeholders
mail -s "Annual Model Re-validation Report - $VALIDATION_YEAR" \
    -a "$VALIDATION_DIR/annual_validation_summary.pdf" \
    clinical-leads@hospital.org,compliance@hospital.org < /dev/null

echo "Annual re-validation complete. Report: $VALIDATION_DIR/annual_validation_summary.pdf"
```

**Re-validation Acceptance Criteria:**

- Performance within 5% of original validation
- No significant degradation in critical metrics (sensitivity, specificity)
- No systematic bias identified (by demographics, disease stage, etc.)
- Clinical utility maintained based on user feedback

**Action on Failed Re-validation:**

If re-validation fails acceptance criteria:

1. **Immediate Investigation:** Determine root cause
2. **Risk Assessment:** Assess patient safety risk
3. **Mitigation Plan:** Develop and implement fixes
4. **Consider Rollback:** If safety concerns exist
5. **Re-train Model:** If drift or data issues identified
6. **Re-validate:** After fixes implemented

---

## Model Governance

### Model Governance Framework

**Governance Committee:**

- **Chair:** Chief Medical Informatics Officer
- **Members:**
  - ML/AI Team Lead
  - Clinical Champion(s)
  - Compliance Officer
  - Privacy Officer
  - IT Security Lead
  - Biostatistician

**Meeting Schedule:** Monthly

**Responsibilities:**

1. **Model Oversight:**
   - Review model performance reports
   - Approve model updates and changes
   - Monitor compliance with policies
   - Assess AI safety and fairness

2. **Policy Development:**
   - Establish model update policies
   - Define performance thresholds
   - Create approval workflows
   - Set validation requirements

3. **Risk Management:**
   - Identify and assess AI risks
   - Monitor adverse events
   - Ensure patient safety
   - Maintain regulatory compliance

4. **Strategic Planning:**
   - Prioritize model development
   - Allocate resources
   - Align with institutional goals
   - Plan for future capabilities

### Model Documentation

**Model Cards:**

Each model must have a comprehensive model card documenting:

- Intended use and limitations
- Training data and methodology
- Performance metrics and validation
- Fairness and bias assessment
- Approved use cases
- Known limitations and edge cases

**Location:** `/var/aimedres/models/[model_name]/[version]/model_card.md`

**Example Model Card Excerpt:**

```markdown
# Model Card: Alzheimer's Risk Assessment Model v1.2

## Model Details
- **Model Name:** Alzheimer's Risk Assessment
- **Version:** 1.2.0
- **Model Type:** Binary Classification (Random Forest)
- **Developed By:** AiMedRes ML Team
- **Development Date:** 2024-01-15
- **Last Updated:** 2024-01-15

## Intended Use
**Primary Use:** Support clinical decision-making for early Alzheimer's disease detection

**Intended Users:** Neurologists, geriatricians, primary care physicians

**Out-of-Scope Uses:**
- Not for standalone diagnosis
- Not for pediatric populations
- Not for acute care settings

## Performance Metrics
| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.89 | [0.87, 0.91] |
| Sensitivity | 0.92 | [0.89, 0.95] |
| Specificity | 0.87 | [0.84, 0.90] |

## Fairness and Bias
- Performance comparable across demographic groups
- No significant bias identified by age, gender, or ethnicity
- Regular fairness monitoring ongoing

## Limitations
- Requires complete patient history
- Lower accuracy for early-stage disease
- Performance not validated for age < 50
```

### Audit Trail

All model-related activities logged in audit system:

- Model training and validation
- Model deployment and updates
- Version changes and rollbacks
- Performance monitoring and alerts
- Governance committee decisions
- Clinical validation findings

**Audit Query Examples:**

```python
from security.hipaa_audit import HIPAAAuditLogger

audit_logger = HIPAAAuditLogger()

# Query model update history
updates = audit_logger.query_model_updates(
    model_name='alzheimer',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Query model inference events
inferences = audit_logger.query_model_inferences(
    model_name='alzheimer',
    user_id='dr_smith',
    start_date='2024-01-15',
    end_date='2024-01-16'
)

# Query performance alerts
alerts = audit_logger.query_performance_alerts(
    model_name='alzheimer',
    severity='warning',
    start_date='2024-01-01'
)
```

---

## Summary

This Model Update & Maintenance Guide provides:

1. **Continuous performance tracking** with automated monitoring and alerting
2. **Drift detection** with statistical analysis and root cause investigation
3. **Safe update procedures** with phased rollout and validation
4. **Version control** with comprehensive metadata and rollback capabilities
5. **Real-world validation** with prospective monitoring and annual re-validation
6. **Model governance** with oversight committee and audit trail

**Key Processes Implemented:**
- ✅ Daily performance monitoring
- ✅ Hourly drift detection
- ✅ Quarterly benchmarking
- ✅ Phased model updates (staging → canary → production)
- ✅ Automatic rollback on failures
- ✅ Annual comprehensive re-validation
- ✅ Model governance committee

**Supporting Tools:**
- `mlops/monitoring/production_monitor.py` - Performance tracking
- `src/aimedres/training/enhanced_drift_monitoring.py` - Drift detection
- `mlops/pipelines/canary_deployment.py` - Safe deployment
- `mlops/model_registry.py` - Version control

**Next Steps:**
1. Establish model governance committee
2. Configure performance monitoring dashboards
3. Set up automated drift detection
4. Document current model versions in registry
5. Schedule first quarterly benchmark
6. Conduct rollback drill
