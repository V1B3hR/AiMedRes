# Examples and Demonstrations

This directory contains demonstration scripts and usage examples organized by complexity and use case.

## Directory Structure

### 📚 `basic/` - Getting Started
Simple demonstrations and introductory examples:
- `run_all_demo.py` - Basic training orchestration demonstration
- `training_demo.py` - Simple training command examples
- `integration.py` - Basic integration patterns
- `integration_example.py` - Integration usage examples
- `usage_examples.py` - General usage examples

**Start here** if you're new to AiMedRes!

### 🏥 `clinical/` - Disease-Specific Examples
Clinical and medical domain demonstrations:
- `als_demo.py` - ALS (Amyotrophic Lateral Sclerosis) training pipeline
- `alzheimer_demo.py` - Alzheimer's disease detection and progression
- `diabetes.py` - Diabetes prediction and risk assessment

These examples show real-world clinical applications.

### 🚀 `advanced/` - Advanced Features
Complex features and performance optimization:
- `enhanced_features_demo.py` - Advanced ML features demonstration
- `enhancements_demo.py` - System enhancement examples
- `parallel_mode.py` - Parallel training orchestration
- `parallel_6workers_50epochs_5folds.py` - High-performance training configuration
- `parallel_custom_params.py` - Custom parallel training parameters
- `simulation_dashboard.py` - Interactive simulation dashboard
- `simulation_dashboard_full.py` - Comprehensive dashboard implementation
- `simulation.py` - Simulation runner
- `labyrinth_simulation.py` - Adaptive labyrinth simulation
- `remote_training.py` - Remote/distributed training examples

Use these examples to optimize performance and scale your workflows.

### 🏢 `enterprise/` - Production & Compliance
Enterprise-grade features, compliance, and production deployment:
- `production_demo.py` - Production deployment patterns
- `security_demo.py` - Security and compliance framework
- `fda_demo.py` - FDA pre-submission documentation
- `automation.py` - Automation and scalability demonstrations
- `mlops.py` - Production MLOps pipeline
- `enhanced_mlops.py` - Advanced MLOps features (drift detection, audit, promotion)
- `enterprise_demo.py` - Complete enterprise system demonstration

These examples demonstrate production-ready implementations and regulatory compliance.

## Quick Start

1. **First time user?** Start with `basic/run_all_demo.py`
2. **Clinical application?** Check `clinical/` for disease-specific examples
3. **Need performance?** Explore `advanced/` for parallel training
4. **Production deployment?** Review `enterprise/` for best practices

## Running Examples

Most examples can be run directly:
```bash
# Basic example
python examples/basic/run_all_demo.py

# Clinical example
python examples/clinical/alzheimer_demo.py

# Advanced parallel training
python examples/advanced/parallel_mode.py

# Enterprise production demo
python examples/enterprise/production_demo.py
```

## Dependencies

Some examples may require additional dependencies. Install with:
```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt  # For ML features
```

## Contributing

When adding new examples:
- Place them in the appropriate category directory
- Add clear documentation and comments
- Follow the existing code style
- Update this README with your example

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.