# Production-Ready MLOps Pipeline Guide

This guide demonstrates the advanced production-ready MLOps features implemented for the DuetMind Adaptive system, including automated retraining, A/B testing, and comprehensive monitoring.

## 🏗️ Architecture Overview

The production MLOps pipeline consists of four main components:

1. **Production Monitoring** - Real-time model performance tracking
2. **Data-Driven Retraining** - Automated model updates based on triggers
3. **A/B Testing Infrastructure** - Controlled model deployment and comparison
4. **Production Serving** - High-performance model inference with monitoring

## 🔍 Production Monitoring

### Features
- Real-time performance metrics (accuracy, precision, recall, F1-score)
- Infrastructure metrics (latency, throughput, error rates)
- Data drift detection using Evidently AI with statistical fallbacks
- Configurable alerting system with multiple severity levels
- Data quality monitoring and anomaly detection

### Quick Start

```python
from mlops.monitoring.production_monitor import create_production_monitor
import pandas as pd
import numpy as np

# Create sample baseline data
baseline_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(0, 1, 1000)
})
baseline_labels = pd.Series(np.random.choice([0, 1], 1000))

# Initialize production monitor
monitor = create_production_monitor(
    model_name="alzheimer_classifier",
    baseline_data=baseline_data,
    baseline_labels=baseline_labels
)

# Log predictions for monitoring
current_data = pd.DataFrame({
    'feature1': np.random.normal(0.1, 1, 100),  # Slight drift
    'feature2': np.random.normal(0, 1, 100),
    'feature3': np.random.normal(0, 1, 100)
})
predictions = np.random.choice([0, 1], 100)
true_labels = np.random.choice([0, 1], 100)

result = monitor.log_prediction_batch(
    features=current_data,
    predictions=predictions,
    true_labels=true_labels,
    model_version="v1.2.0",
    latency_ms=120.0
)

print(f"Logged {result['metrics']['prediction_count']} predictions")
print(f"Accuracy: {result['metrics']['accuracy']:.3f}")
print(f"Drift Score: {result['metrics']['drift_score']:.3f}")

# Get monitoring summary
summary = monitor.get_monitoring_summary(hours=24)
print(f"System Status: {summary['status']}")
print(f"Total Predictions: {summary['total_predictions']}")
print(f"Average Accuracy: {summary['avg_accuracy']:.3f}")
```

### Alert Configuration

```python
from mlops.monitoring.production_monitor import AlertConfig

# Custom alert configuration
alert_config = AlertConfig(
    critical_accuracy_drop=0.05,    # 5% accuracy drop triggers critical alert
    warning_accuracy_drop=0.02,     # 2% accuracy drop triggers warning
    critical_error_rate=0.02,       # 2% error rate triggers critical alert
    critical_latency_ms=500.0,      # 500ms latency triggers critical alert
    drift_threshold=0.1             # 10% drift threshold
)

monitor = create_production_monitor(
    model_name="alzheimer_classifier",
    alert_config=alert_config
)
```

## 🔄 Data-Driven Automated Retraining

### Features
- File system monitoring for new data arrival
- Performance-based triggers (accuracy degradation, drift, error rates)
- Time-based triggers (maximum days without retraining)
- Configurable constraints and approval workflows
- Comprehensive audit trail

### Quick Start

```python
from mlops.monitoring.data_trigger import create_data_trigger, RetrainingTriggerConfig

# Configure retraining triggers
config = RetrainingTriggerConfig(
    min_new_samples=1000,           # Minimum new samples to trigger
    max_days_without_retrain=7,     # Maximum days without retraining
    accuracy_degradation_threshold=0.05,  # 5% accuracy drop triggers retrain
    drift_score_threshold=0.1,      # 10% drift triggers retrain
    min_hours_between_retrains=6,   # Minimum 6 hours between retrains
    max_retrains_per_day=4,         # Maximum 4 retrains per day
    data_directories=["data/raw", "data/processed"],
    watch_file_patterns=[".csv", ".parquet"]
)

# Create retraining trigger
trigger = create_data_trigger(
    model_name="alzheimer_classifier",
    config=config,
    production_monitor=monitor  # Link to monitoring system
)

# Start automated monitoring
trigger.start_monitoring()

# Force manual retrain if needed
trigger.force_retrain("Manual trigger for testing")

# Check trigger history
history = trigger.get_trigger_history(days=7)
print(f"Retraining events in last 7 days: {len(history)}")
```

### Custom Retraining Integration

The retraining trigger can automatically execute custom retraining scripts:

```python
# In your retraining configuration
def custom_retrain_handler(trigger_type, reason):
    """Custom handler for retraining events."""
    import subprocess
    
    # Execute your retraining script
    # Example: Custom script execution
    result = subprocess.run([
        "python", "scripts/trigger_retraining.py",
        "--trigger-type", trigger_type,
        "--reason", reason
    ])
    
    return result.returncode == 0

# Configure trigger with custom handler
trigger.retraining_handler = custom_retrain_handler
```

## 🧪 A/B Testing Infrastructure

### Features
- Statistical experiment design with configurable traffic splits
- User segmentation and deterministic assignment
- Statistical significance testing (chi-square, t-tests)
- Confidence intervals and effect size analysis
- Automated experiment lifecycle management

### Quick Start

```python
from mlops.monitoring.ab_testing import ABTestingManager, ABTestConfig

# Initialize A/B testing manager
ab_manager = ABTestingManager()

# Create experiment configuration
config = ABTestConfig(
    experiment_name="alzheimer_model_v2_test",
    model_a_version="v1.0.0",      # Control model
    model_b_version="v2.0.0",      # Treatment model
    traffic_split=0.3,             # 30% traffic to new model
    duration_days=14,              # Run for 2 weeks
    min_samples_per_variant=1000,  # Minimum samples for statistical power
    significance_level=0.05,       # 95% confidence level
    primary_metric="accuracy"
)

# Create and start experiment
experiment_name = ab_manager.create_experiment(config)
ab_manager.start_experiment(experiment_name)

# Mock models for demonstration
class MockModel:
    def __init__(self, accuracy):
        self.accuracy = accuracy
    
    def predict(self, X):
        return [1 if np.random.random() < self.accuracy else 0 for _ in range(len(X))]

models = {
    "v1.0.0": MockModel(0.85),  # 85% accuracy
    "v2.0.0": MockModel(0.88)   # 88% accuracy  
}

# Make predictions through A/B testing
user_outcomes = []
for i in range(500):
    result = ab_manager.make_prediction(
        experiment_name=experiment_name,
        user_id=f"user_{i}",
        features={'age': 65 + i % 20, 'mmse': 25 - i % 10},
        models=models
    )
    
    # Simulate true outcome
    true_label = 1 if np.random.random() < 0.8 else 0
    user_outcomes.append((f"user_{i}", true_label))

# Update with true outcomes
ab_manager.update_prediction_outcomes(experiment_name, user_outcomes)

# Analyze experiment results
results = ab_manager.analyze_experiment(experiment_name)

print(f"Experiment: {results.experiment_name}")
print(f"Control (A): {results.samples_a} samples, {results.metrics_a['accuracy']:.3f} accuracy")
print(f"Treatment (B): {results.samples_b} samples, {results.metrics_b['accuracy']:.3f} accuracy")
print(f"Statistical Significance: {results.statistical_significance}")
print(f"Winner: {results.winner}")
print(f"Recommendation: {results.recommendation}")

# Stop experiment when ready
final_results = ab_manager.stop_experiment(experiment_name)
```

### User Segmentation

```python
# Configure user segmentation criteria
config = ABTestConfig(
    experiment_name="segmented_experiment",
    model_a_version="v1.0.0",
    model_b_version="v2.0.0",
    traffic_split=0.5,
    segment_criteria={
        'age': {'min': 65, 'max': 85},    # Age range 65-85
        'education': {'min': 8},          # Education >= 8 years
        'region': ['north', 'south']      # Specific regions only
    }
)
```

## 🚀 Production Model Serving

### Features
- High-performance Flask API with health checks
- Multi-model version management and loading
- Integrated monitoring and A/B testing
- Batch prediction support
- Graceful error handling and fallbacks

### Quick Start

```python
from mlops.serving.production_server import create_production_server

# Create production server
server = create_production_server(
    model_name="alzheimer_classifier",
    enable_monitoring=True,
    enable_ab_testing=True
)

# Setup baseline monitoring data
server.setup_baseline_monitoring("data/processed/baseline_features.csv")

# Run server (in production)
# server.run(host='0.0.0.0', port=5000)
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_name": "alzheimer_classifier",
  "current_model_version": "v1.2.0",
  "uptime_seconds": 3600,
  "request_count": 1250,
  "error_count": 3,
  "models_loaded": ["v1.0.0", "v1.2.0"]
}
```

#### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "M/F": 0,
      "Age": 72,
      "EDUC": 16,
      "SES": 3,
      "MMSE": 23,
      "CDR": 0.5,
      "eTIV": 1450,
      "nWBV": 0.72,
      "ASF": 1.15
    },
    "user_id": "patient_12345",
    "experiment_name": "alzheimer_model_v2_test"
  }'
```

Response:
```json
{
  "prediction": 1,
  "model_version": "v2.0.0",
  "variant": "treatment",
  "user_id": "patient_12345",
  "confidence": 0.87,
  "latency_ms": 45.2,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "batch_features": [
      {"M/F": 0, "Age": 72, "EDUC": 16, "MMSE": 23},
      {"M/F": 1, "Age": 68, "EDUC": 12, "MMSE": 28}
    ],
    "user_ids": ["patient_1", "patient_2"]
  }'
```

#### Monitoring Status
```bash
curl http://localhost:5000/monitoring/status?hours=24
```

#### Experiment Status
```bash
curl http://localhost:5000/experiments/alzheimer_model_v2_test/status
```

## 📊 Monitoring Dashboard

### Real-time Metrics

The monitoring system provides real-time insights into model performance:

```python
# Get comprehensive monitoring summary
summary = monitor.get_monitoring_summary(hours=24)

print(f"""
📊 Model Performance Summary (Last 24 Hours)
============================================
Status: {summary['status']}
Total Predictions: {summary['total_predictions']:,}
Average Accuracy: {summary['avg_accuracy']:.3f}
Average Latency: {summary['avg_latency_ms']:.1f}ms
Average Error Rate: {summary['avg_error_rate']:.3f}
Drift Incidents: {summary['drift_incidents']}
Data Quality: {summary['avg_data_quality']:.3f}

Alerts:
- Critical: {summary['alerts'].get('CRITICAL', 0)}
- Warning: {summary['alerts'].get('WARNING', 0)}
- Info: {summary['alerts'].get('INFO', 0)}
""")
```

### Performance Trends

```python
# Analyze performance trends over time
def analyze_performance_trends(monitor, days=7):
    trends = []
    
    for day in range(days):
        start_hour = day * 24
        end_hour = start_hour + 24
        
        summary = monitor.get_monitoring_summary(hours=24)
        trends.append({
            'day': day,
            'accuracy': summary['avg_accuracy'],
            'latency': summary['avg_latency_ms'],
            'predictions': summary['total_predictions']
        })
    
    return trends

trends = analyze_performance_trends(monitor)
for trend in trends:
    print(f"Day {trend['day']}: {trend['accuracy']:.3f} accuracy, "
          f"{trend['latency']:.1f}ms latency, {trend['predictions']} predictions")
```

## 🔧 Configuration Management

### Central Configuration

All components can be configured through YAML files:

```yaml
# production_config.yaml
production:
  monitoring:
    alert_config:
      critical_accuracy_drop: 0.05
      warning_accuracy_drop: 0.02
      critical_latency_ms: 500
      drift_threshold: 0.1
    
  retraining:
    min_new_samples: 1000
    max_days_without_retrain: 7
    accuracy_degradation_threshold: 0.05
    min_hours_between_retrains: 6
    max_retrains_per_day: 4
    data_directories:
      - "data/raw"
      - "data/processed"
    
  ab_testing:
    default_traffic_split: 0.1
    min_samples_per_variant: 500
    significance_level: 0.05
    
  serving:
    host: "0.0.0.0"
    port: 5000
    enable_monitoring: true
    enable_ab_testing: true
```

### Environment-Specific Configuration

```python
import yaml
from mlops.monitoring.production_monitor import create_production_monitor
from mlops.monitoring.data_trigger import create_data_trigger

def load_production_config(env="production"):
    with open(f"config/{env}_config.yaml") as f:
        return yaml.safe_load(f)

def setup_production_pipeline(env="production"):
    config = load_production_config(env)
    
    # Initialize monitoring
    monitor = create_production_monitor(
        model_name="alzheimer_classifier",
        alert_config=config['production']['monitoring']['alert_config']
    )
    
    # Initialize retraining triggers
    trigger = create_data_trigger(
        model_name="alzheimer_classifier",
        config=config['production']['retraining'],
        production_monitor=monitor
    )
    
    return monitor, trigger
```

## 🚨 Alerting and Incident Response

### Alert Types

1. **Critical Alerts**
   - Model accuracy drops > 5%
   - Error rate exceeds 2%
   - Response time exceeds 500ms
   - Concept drift detected

2. **Warning Alerts**
   - Model accuracy drops 2-5%
   - Error rate 1-2%
   - Response time 200-500ms
   - Feature drift detected

3. **Info Alerts**
   - Minor performance fluctuations
   - Data quality issues
   - New data availability

### Incident Response Workflow

```python
def handle_critical_alert(alert_type, metrics):
    """Handle critical alerts with automated response."""
    
    if alert_type == "accuracy_drop":
        # Trigger immediate retraining
        trigger.force_retrain(f"Critical accuracy drop: {metrics['accuracy']}")
        
        # Activate rollback if available
        if hasattr(server, 'rollback_model'):
            server.rollback_model("previous_stable")
            
    elif alert_type == "high_error_rate":
        # Scale down traffic to problematic model
        if ab_manager.active_experiments:
            for exp_name in ab_manager.active_experiments:
                ab_manager.pause_experiment(exp_name)
                
    elif alert_type == "concept_drift":
        # Schedule immediate retraining
        trigger.force_retrain("Concept drift detected")
        
        # Notify ML team
        send_notification(
            channel="ml-ops-alerts",
            message=f"Concept drift detected - retraining initiated"
        )

# Register alert handler
monitor.register_alert_handler(handle_critical_alert)
```

## 📈 Performance Optimization

### Batch Processing

```python
# Optimize for batch predictions
def batch_predict_with_monitoring(server, batch_data, batch_size=100):
    """Process predictions in batches with monitoring."""
    results = []
    
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i+batch_size]
        
        # Process batch
        batch_results = server.predict_batch(batch)
        results.extend(batch_results)
        
        # Log batch metrics
        if len(results) % 1000 == 0:
            print(f"Processed {len(results)} predictions")
    
    return results
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedPredictor:
    def __init__(self, model):
        self.model = model
        self.cache = {}
    
    def predict_with_cache(self, features):
        """Predict with feature-based caching."""
        # Create cache key from features
        feature_key = hashlib.md5(str(sorted(features.items())).encode()).hexdigest()
        
        if feature_key in self.cache:
            return self.cache[feature_key]
        
        # Make prediction
        prediction = self.model.predict([list(features.values())])[0]
        
        # Cache result
        self.cache[feature_key] = prediction
        
        return prediction
```

## 🔒 Security Considerations

### Input Validation

```python
from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    user_id: str = Field(..., min_length=1, max_length=100)
    experiment_name: str = Field(None, max_length=100)
    
    class Config:
        # Prevent additional fields
        extra = "forbid"

def validate_prediction_request(data: Dict[str, Any]) -> PredictionRequest:
    """Validate incoming prediction request."""
    try:
        return PredictionRequest(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid request: {e}")
```

### Audit Logging

```python
import logging
from datetime import datetime

def audit_log(action: str, user_id: str, details: Dict[str, Any]):
    """Log actions for audit trail."""
    audit_logger = logging.getLogger('audit')
    
    audit_record = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'user_id': user_id,
        'details': details
    }
    
    audit_logger.info(json.dumps(audit_record))

# Usage
audit_log("model_prediction", "user_123", {
    "model_version": "v1.2.0",
    "prediction": 1,
    "confidence": 0.87
})
```

## 🎯 Best Practices

### 1. Monitoring
- Set appropriate alert thresholds based on business requirements
- Monitor both model performance and infrastructure metrics
- Implement gradual rollout for model updates
- Maintain historical performance baselines

### 2. A/B Testing
- Use sufficient sample sizes for statistical power
- Run experiments for adequate duration
- Consider seasonal effects and external factors
- Document experiment results and learnings

### 3. Automated Retraining
- Balance automation with human oversight
- Implement safety checks and approval workflows
- Monitor retraining frequency and success rates
- Maintain rollback capabilities

### 4. Production Serving
- Implement health checks and graceful degradation
- Use load balancing and auto-scaling
- Monitor API performance and error rates
- Implement proper logging and observability

## 📚 Further Reading

- [MLOps Architecture Documentation](./MLOPS_ARCHITECTURE.md)
- [Model Promotion Guidelines](./mlops/registry/model_promotion.md)
- [Configuration Reference](./params.yaml)
- [API Documentation](./api/README.md)

## 🤝 Contributing

To contribute to the production MLOps pipeline:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure backward compatibility
5. Test thoroughly in staging environment

For questions or support, contact the MLOps team or create an issue in the repository.