# Automation & Scalability Implementation Guide

This document provides a comprehensive guide to the newly implemented Automation & Scalability features for DuetMind Adaptive.

## 🎯 Overview

The Automation & Scalability system implements four core components:

1. **AutoML Integration** - Automated hyperparameter optimization and model selection
2. **Pipeline Customization** - Flexible, configurable training pipelines
3. **Scalable Orchestration** - Workflow management and distributed computing
4. **Enhanced Drift Monitoring** - Automated drift detection with response workflows

## 📦 Components

### 1. AutoML Integration (`automl.py`)

**Purpose**: Automated machine learning with hyperparameter optimization using Optuna.

**Key Features**:
- Multi-algorithm support (RandomForest, LogisticRegression, XGBoost, LightGBM)
- Bayesian optimization with Optuna
- Cross-validation based evaluation
- Parameter importance analysis
- Study persistence and loading

**Usage Example**:
```python
from src.duetmind_adaptive.training.automl import create_automl_optimizer

optimizer = create_automl_optimizer(
    objective_metric='roc_auc',
    n_trials=100,
    timeout=3600
)

results = optimizer.optimize(X_train, y_train, X_val, y_val)
print(f"Best score: {results['best_score']}")
print(f"Best algorithm: {results['best_algorithm']}")
```

**Configuration**:
- `objective_metric`: 'roc_auc', 'accuracy', 'f1'
- `n_trials`: Maximum optimization trials
- `timeout`: Time limit in seconds
- `cv_folds`: Cross-validation folds

### 2. Pipeline Customization (`custom_pipeline.py`)

**Purpose**: Dynamic pipeline builder with configurable preprocessing and model components.

**Key Features**:
- Flexible preprocessing configuration
- Custom transformer support
- Pipeline registry for reusable configurations
- Dynamic model selection
- Ensemble method support

**Usage Example**:
```python
from src.duetmind_adaptive.training.custom_pipeline import (
    create_pipeline_builder, PipelineConfig, PreprocessingConfig, ModelConfig
)

config = PipelineConfig(
    name="my_pipeline",
    preprocessing=PreprocessingConfig(
        numerical_scaler='standard',
        categorical_encoding='onehot'
    ),
    model=ModelConfig(
        algorithm='random_forest',
        hyperparameters={'n_estimators': 100}
    )
)

builder = create_pipeline_builder(config)
pipeline = builder.build_complete_pipeline(X)
```

**Configuration Options**:
- **Preprocessing**: Scaling, encoding, feature selection, outlier detection
- **Models**: RandomForest, LogisticRegression, SVM, XGBoost, LightGBM
- **Ensemble**: Voting, stacking, bagging methods

### 3. Scalable Orchestration (`orchestration.py`)

**Purpose**: Workflow orchestration with resource management and distributed execution.

**Key Features**:
- Task dependency management
- Resource allocation and scheduling
- Ray-based distributed computing (optional)
- Fault tolerance with retries
- Progress monitoring

**Usage Example**:
```python
from src.duetmind_adaptive.training.orchestration import (
    create_orchestrator, ResourceRequirement
)

orchestrator = create_orchestrator(use_ray=True)

# Add tasks
task_id = orchestrator.add_task(
    task_id="train_model",
    function=training_function,
    args=(data, config),
    resources=ResourceRequirement(cpu_cores=4, memory_gb=8.0)
)

# Run workflow
results = orchestrator.run_workflow()
```

**Resource Management**:
- CPU cores allocation
- Memory requirements (GB)
- GPU support
- Automatic resource tracking

### 4. Enhanced Drift Monitoring (`enhanced_drift_monitoring.py`)

**Purpose**: Comprehensive drift detection with automated response workflows.

**Key Features**:
- Multi-type drift detection (data, model, concept)
- Configurable alerting (email, webhook, Slack)
- Automated response actions
- Historical drift tracking
- Integration with retraining workflows

**Usage Example**:
```python
from src.duetmind_adaptive.training.enhanced_drift_monitoring import (
    create_enhanced_drift_monitor, AlertConfig, ResponseConfig
)

alert_config = AlertConfig(
    enabled_channels=[AlertChannel.EMAIL, AlertChannel.LOG],
    email_config={'sender_email': 'system@example.com'}
)

response_config = ResponseConfig(
    enabled_actions=[ResponseAction.RETRAIN_MODEL],
    retrain_config={'max_auto_retrains_per_day': 3}
)

monitor = create_enhanced_drift_monitor(
    reference_data=ref_data,
    baseline_metrics=baseline,
    alert_config=alert_config,
    response_config=response_config
)

results = monitor.detect_comprehensive_drift(current_data, current_metrics)
```

**Alert Channels**:
- Email notifications
- Webhook calls
- Slack integration
- Standard logging

**Response Actions**:
- Automated model retraining
- Resource scaling
- Threshold adjustments
- Human review requests
- Model rollback

### 5. Integrated System (`automation_system.py`)

**Purpose**: Unified system integrating all automation components.

**Key Features**:
- Single entry point for all automation features
- Workflow orchestration integration
- State persistence and recovery
- Configuration management
- Complete automation pipelines

**Usage Example**:
```python
from src.duetmind_adaptive.training.automation_system import setup_complete_system

system = setup_complete_system(
    reference_data=reference_data,
    baseline_metrics=baseline_metrics,
    working_dir="./automation"
)

workflow_id = system.create_automated_training_workflow(
    data_path="data.csv",
    target_column="diagnosis",
    enable_automl=True
)

results = system.run_workflow(workflow_id)
```

## 🚀 Getting Started

### Installation

```bash
# Required dependencies
pip install optuna scikit-learn pandas numpy

# Optional dependencies for full functionality
pip install ray xgboost lightgbm evidently requests
```

### Quick Start

```python
# 1. Create sample data
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# 2. Setup automation system
from src.duetmind_adaptive.training.automation_system import setup_complete_system

baseline_metrics = {'accuracy': 0.85, 'roc_auc': 0.88}
system = setup_complete_system(data.drop('target', axis=1), baseline_metrics)

# 3. Create and run workflow
workflow_id = system.create_automated_training_workflow(
    data_path="data.csv",
    target_column="target",
    enable_automl=True
)

results = system.run_workflow(workflow_id)
```

## 📊 Demo Script

Run the comprehensive demonstration:

```bash
python examples/demo_automation_scalability.py
```

This script demonstrates:
- AutoML hyperparameter optimization
- Custom pipeline configurations
- Distributed workflow execution
- Drift detection and alerting
- Complete system integration

## 🔧 Configuration

### System Configuration (`system_config.yaml`)

```yaml
automl:
  objective_metric: "roc_auc"
  n_trials: 100
  timeout: 3600
  cv_folds: 5

orchestration:
  use_ray: false
  max_concurrent_tasks: 4

drift_monitoring:
  monitoring_interval: 3600
  drift_threshold: 0.15
  alert_channels: ["log", "email"]
  auto_retrain_enabled: false

pipeline_defaults:
  preprocessing:
    numerical_scaler: "standard"
    categorical_encoding: "onehot"
  model:
    algorithm: "random_forest"
    hyperparameters:
      n_estimators: 100
      random_state: 42
```

### Pipeline Configuration

```yaml
name: "advanced_pipeline"
description: "Advanced classification pipeline"
preprocessing:
  numerical_scaler: "robust"
  categorical_encoding: "onehot"
  missing_value_strategy: "median"
  feature_selection: "selectkbest"
model:
  algorithm: "xgboost"
  hyperparameters:
    n_estimators: 200
    max_depth: 6
  ensemble_method: "voting"
evaluation:
  metrics: ["accuracy", "roc_auc", "f1"]
```

## 🔄 Integration with Existing System

### Training Pipeline Integration

```python
# Integrate with existing training
from training.train_alzheimers import AlzheimerTrainingPipeline
from src.duetmind_adaptive.training.automation_system import create_automation_system

# Setup automation
system = create_automation_system()
system.initialize_system(reference_data, baseline_metrics)

# Create enhanced workflow
workflow_id = system.create_automated_training_workflow(
    data_path="alzheimer_data.csv",
    target_column="diagnosis",
    pipeline_name="medical_classification",
    enable_automl=True
)

# Execute with monitoring
results = system.run_workflow(workflow_id)
```

### MLOps Integration

```python
# Integrate with existing MLOps components
from mlops.monitoring.production_monitor import create_production_monitor
from src.duetmind_adaptive.training.enhanced_drift_monitoring import create_enhanced_drift_monitor

# Enhanced monitoring setup
enhanced_monitor = create_enhanced_drift_monitor(
    reference_data=production_data,
    baseline_metrics=production_metrics,
    alert_config=alert_config,
    response_config=response_config
)

# Continuous monitoring loop
async def monitoring_loop():
    while True:
        current_data = await get_current_production_data()
        current_metrics = await get_current_metrics()
        
        drift_results = enhanced_monitor.detect_comprehensive_drift(
            current_data, current_metrics
        )
        
        # Execute automated responses
        for action in drift_results.get('recommended_actions', []):
            await enhanced_monitor.execute_response_action(action, alert)
        
        await asyncio.sleep(3600)  # Check every hour
```

## 📈 Performance Considerations

### AutoML Optimization
- Use parallel evaluation with Ray for faster optimization
- Set appropriate trial limits based on dataset size
- Consider early stopping for long-running optimizations

### Orchestration Scaling
- Configure Ray cluster for distributed execution
- Monitor resource utilization and adjust limits
- Use efficient data serialization for large datasets

### Drift Monitoring
- Batch drift detection for efficiency
- Implement intelligent sampling for large data streams
- Use statistical approximations for real-time detection

## 🛠️ Troubleshooting

### Common Issues

1. **Ray Import Error**: Install Ray with `pip install ray`
2. **Optuna Timeout**: Increase timeout or reduce n_trials
3. **Memory Issues**: Reduce batch sizes or enable data streaming
4. **Missing Dependencies**: Install optional packages as needed

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# All components will now provide detailed logging
```

## 🔮 Future Enhancements

### Planned Features
- **Advanced AutoML**: Neural architecture search, multi-objective optimization
- **Enhanced Orchestration**: Kubernetes integration, auto-scaling
- **Intelligent Monitoring**: Anomaly detection, predictive drift analysis
- **UI Dashboard**: Web-based monitoring and control interface

### Extension Points
- Custom drift detection algorithms
- Additional model algorithms
- Custom response actions
- External system integrations

## 📚 API Reference

### AutoML Classes
- `AutoMLOptimizer`: Main optimization class
- `create_automl_optimizer()`: Factory function

### Pipeline Classes  
- `DynamicPipelineBuilder`: Pipeline builder
- `PipelineRegistry`: Configuration registry
- `PipelineConfig`: Configuration dataclass

### Orchestration Classes
- `WorkflowOrchestrator`: Main orchestrator
- `WorkflowBuilder`: Workflow builder utility
- `Task`: Task definition dataclass

### Monitoring Classes
- `EnhancedDriftMonitor`: Main monitoring class
- `DriftAlert`: Alert dataclass
- `AlertConfig`: Alert configuration

### System Classes
- `AutomationScalabilitySystem`: Integrated system
- `setup_complete_system()`: Quick setup function

## 📄 License

This implementation is part of the DuetMind Adaptive project and follows the same licensing terms.