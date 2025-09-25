# DuetMind Adaptive User Experience Guide

Welcome to DuetMind Adaptive - a comprehensive AI platform for medical research and healthcare applications. This guide will help you get the most out of the platform's features.

## 🚀 Getting Started

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/V1B3hR/duetmind_adaptive.git
cd duetmind_adaptive

# Install dependencies
pip install -r requirements-dev.txt

# Run your first example
python examples/simple_integration_demo.py
```

### Quick Start Examples
```python
# Basic medical data analysis
from duetmind import DuetMindAdaptive

# Initialize the system
dm = DuetMindAdaptive()

# Load and analyze medical data
result = dm.analyze_patient_data("patient_data.csv")
print(f"Risk assessment: {result['risk_score']}")
```

## 🏥 Medical Use Cases

### 1. Alzheimer's Disease Prediction
```python
from training.train_alzheimers import AlzheimerTrainer
import pandas as pd

# Load patient data
patient_data = pd.read_csv("alzheimer_dataset.csv")

# Train the model
trainer = AlzheimerTrainer()
trainer.train_model(patient_data)

# Make predictions
test_data = pd.DataFrame({
    'age': [72, 68, 75],
    'mmse_score': [24, 28, 20],
    'apoe_genotype': ['E3/E4', 'E3/E3', 'E4/E4']
})

predictions = trainer.predict(test_data)
print("Alzheimer's risk predictions:", predictions)
```

### 2. Medical Image Analysis
```python
from training.train_brain_mri import BrainMRITrainer

# Initialize MRI analysis
mri_trainer = BrainMRITrainer()

# Analyze brain scans
result = mri_trainer.analyze_scan("brain_mri.nii.gz")
print(f"Analysis: {result['findings']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### 3. Cardiovascular Risk Assessment
```python
from training.train_cardiovascular import CardiovascularTrainer

# Load cardiovascular data
cv_trainer = CardiovascularTrainer()
cv_trainer.load_data("cardiovascular_dataset.csv")

# Train and evaluate
cv_trainer.train_model()
risk_assessment = cv_trainer.assess_risk({
    'age': 55,
    'cholesterol': 240,
    'blood_pressure': [140, 90],
    'smoking': True
})

print(f"Cardiovascular risk: {risk_assessment}")
```

## 🔬 Advanced Features

### Multi-Agent Reasoning
```python
from duetmind import MultiAgentSystem

# Create specialized medical agents
agents = MultiAgentSystem([
    'diagnostic_agent',
    'treatment_agent', 
    'risk_assessment_agent'
])

# Collaborative analysis
case_analysis = agents.analyze_case({
    'patient_id': 'P001',
    'symptoms': ['chest_pain', 'shortness_of_breath'],
    'vitals': {'bp': [140, 90], 'hr': 95},
    'lab_results': {'troponin': 0.05}
})

print(f"Consensus diagnosis: {case_analysis['consensus']}")
```

### Federated Learning
```python
from training import FederatedTrainer

# Set up federated learning across institutions
fed_trainer = FederatedTrainer(
    model_type='alzheimer_classifier',
    num_clients=5,
    privacy_method='differential_privacy'
)

# Train collaboratively while preserving privacy
fed_trainer.federated_train(
    client_data_paths=['hospital_1.csv', 'hospital_2.csv', ...],
    rounds=10
)
```

### AutoML Integration
```python
from src.duetmind_adaptive.training.automl import AutoMLTrainer

# Automated hyperparameter optimization
automl = AutoMLTrainer(
    task='classification',
    target='diagnosis',
    optimization_method='bayesian'
)

# Auto-optimize your medical model
best_model = automl.optimize(
    train_data=medical_data,
    trials=100,
    timeout_hours=2
)

print(f"Best accuracy: {best_model.accuracy:.3f}")
```

## 📊 Monitoring & Production

### Production Deployment
```python
from duetmind import ProductionDeploymentManager

# Configure production environment
config = {
    'port': 8080,
    'workers': 4,
    'gpu_enabled': True,
    'deployment_path': './production_deployment'
}

# Generate deployment files
manager = ProductionDeploymentManager(config)
manager.deploy_to_files()

# Files generated: Dockerfile, docker-compose.yml, k8s manifests, etc.
```

### Model Monitoring
```python
from mlops.monitoring.production_monitor import create_production_monitor

# Set up production monitoring
monitor = create_production_monitor(
    model_name="alzheimer_classifier",
    baseline_data=baseline_data,
    baseline_labels=baseline_labels
)

# Monitor predictions in real-time
monitor.log_prediction(
    input_data=patient_features,
    prediction='MCI',
    actual_outcome='MCI',  # when available
    confidence=0.87
)

# Check for drift and issues
health_report = monitor.get_health_report()
```

### A/B Testing
```python
from mlops.ab_testing import ABTestManager

# Set up A/B testing for model comparison
ab_manager = ABTestManager()
ab_manager.create_experiment(
    name='alzheimer_model_v2',
    model_a=current_model,
    model_b=new_model,
    traffic_split=0.2  # 20% to new model
)

# Route predictions through A/B test
result = ab_manager.make_prediction(
    experiment_name='alzheimer_model_v2',
    user_id='patient_123',
    features=patient_data
)
```

## 🛡️ Security & Privacy

### Data De-identification
```python
from security.data_deidentification import DeidentificationEngine

# De-identify sensitive medical data
deidentifier = DeidentificationEngine()
clean_data = deidentifier.deidentify_dataset(
    data=medical_records,
    method='k_anonymity',
    k_value=5
)
```

### Secure API Access
```python
from secure_api_server import SecureAPIServer

# Start secure medical API server
server = SecureAPIServer(
    ssl_cert='cert.pem',
    ssl_key='key.pem',
    api_key_required=True
)

server.run(host='0.0.0.0', port=443)
```

## 📈 Clinical Decision Support

### Risk Stratification
```python
from clinical_decision_support import RiskStratificationEngine

# Advanced clinical risk assessment
risk_engine = RiskStratificationEngine()
risk_engine.load_clinical_guidelines('medical_guidelines.yaml')

patient_risk = risk_engine.stratify_risk({
    'patient_id': 'P001',
    'age': 65,
    'conditions': ['hypertension', 'diabetes'],
    'medications': ['metformin', 'lisinopril'],
    'vitals': {'systolic_bp': 145, 'diastolic_bp': 85}
})

print(f"Risk level: {patient_risk['level']}")
print(f"Recommendations: {patient_risk['recommendations']}")
```

### Clinical Scenario Validation
```python
from demo_enhanced_features import ClinicalScenarioBuilder

# Build and validate clinical scenarios
scenario_builder = ClinicalScenarioBuilder()
scenario = scenario_builder.create_scenario({
    'patient_age': 70,
    'conditions': ['atrial_fibrillation'],
    'proposed_medication': 'warfarin',
    'contraindications_check': True
})

validation = scenario_builder.validate_scenario(scenario)
print(f"Scenario valid: {validation['valid']}")
```

## 🎛️ Web Dashboard

### Simulation Dashboard
```python
# Start the web-based simulation dashboard
from examples.demo_simulation_dashboard import main as start_dashboard

# Launch interactive medical simulation interface
start_dashboard()
# Access at: http://localhost:5000
```

Dashboard features:
- 📊 Real-time model performance metrics
- 🧬 Medical data visualization
- 🔬 Interactive scenario testing
- 📈 Training progress monitoring
- 🚨 Alert management system

## 🔧 Customization & Extension

### Custom Medical Models
```python
from training.custom_pipeline import CustomPipelineBuilder

# Build custom medical analysis pipeline
builder = CustomPipelineBuilder()
pipeline = builder.create_pipeline({
    'preprocessors': ['medical_scaler', 'feature_selector'],
    'model': 'random_forest',
    'postprocessors': ['confidence_calibrator'],
    'validation': 'clinical_validation'
})

# Train with your medical data
pipeline.fit(medical_training_data, labels)
```

### Integration with Medical Systems
```python
from ehr_integration import EHRIntegration

# Connect to Electronic Health Records
ehr = EHRIntegration()
ehr.connect_to_system('epic', credentials=epic_config)

# Fetch and analyze patient data
patient_data = ehr.get_patient_data('patient_123')
analysis = dm.analyze_patient_data(patient_data)

# Send results back to EHR
ehr.update_patient_record('patient_123', analysis)
```

## 🚨 Common Issues & Solutions

### Performance Optimization
```python
# Enable GPU acceleration
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use distributed training
from training.orchestration import RayTrainer
trainer = RayTrainer(num_workers=4)
```

### Memory Management
```python
# For large medical datasets
from utils import MemoryEfficientLoader
loader = MemoryEfficientLoader(batch_size=32)
for batch in loader.load_medical_data('large_dataset.csv'):
    process_batch(batch)
```

### Data Quality Issues
```python
from data_quality_monitor import DataQualityMonitor

# Monitor data quality in real-time
quality_monitor = DataQualityMonitor()
quality_report = quality_monitor.analyze_dataset(medical_data)

if quality_report['issues']:
    print("Data quality issues detected:")
    for issue in quality_report['issues']:
        print(f"- {issue}")
```

## 📚 Additional Resources

### Documentation
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Clinical Guidelines](docs/CLINICAL_DECISION_SUPPORT_README.md) - Medical best practices
- [Security Guide](docs/SECURITY_GUIDELINES.md) - Privacy and security features
- [MLOps Architecture](MLOPS_ARCHITECTURE.md) - Production deployment guide

### Example Notebooks
- `examples/usage_diabetes_example.py` - Diabetes risk prediction
- `examples/demo_enhanced_features.py` - Advanced clinical features
- `examples/demo_production_mlops.py` - Production MLOps pipeline

### Support
- 📖 Check the documentation first
- 🐛 Report bugs via GitHub Issues
- 💬 Join discussions in GitHub Discussions
- 🤝 Contributing? See [CONTRIBUTING.md](CONTRIBUTING.md)

## 🎯 Best Practices

### Medical Data Handling
1. **Always validate** medical data before analysis
2. **Use synthetic data** for testing and examples
3. **Follow privacy regulations** (HIPAA, GDPR)
4. **Include clinical context** in all analyses
5. **Validate results** with medical experts

### Model Development
1. **Start with established benchmarks**
2. **Use cross-validation** for robust evaluation
3. **Test on diverse populations**
4. **Include uncertainty quantification**
5. **Document clinical assumptions**

### Production Deployment
1. **Monitor model performance** continuously
2. **Implement gradual rollouts** (A/B testing)
3. **Maintain audit trails** for regulatory compliance
4. **Plan for model degradation**
5. **Have rollback procedures** ready

---

Welcome to the future of AI-powered healthcare! 🏥✨

For more information, visit our [GitHub repository](https://github.com/V1B3hR/duetmind_adaptive) or check out the complete [documentation](docs/).