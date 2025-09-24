# duetmind_adaptive

## What is duetmind_adaptive?

**duetmind_adaptive** is an advanced AI research project focused on simulating adaptive neural networks and multi-agent systems for medical decision-making. It integrates biological state simulation (energy, sleep, mood), multi-agent dialogue, and real-world data to model and understand complex brain diseases.

## Purpose

The purpose of duetmind_adaptive is to accelerate research and development of AI-driven solutions for neurodegenerative and mental health conditions. By leveraging adaptive neural architectures and agent-based collaboration, this project aims to create tools and frameworks that enhance understanding of disease mechanisms and improve clinical decision support.

## Goal

The primary goal is to deepen our understanding of diseases affecting the brain—such as Alzheimer's, stroke, and other neurological disorders—and to develop better solutions for diagnosis, monitoring, and intervention. duetmind_adaptive strives to answer critical questions about disease progression and treatment by:
- Simulating biological and cognitive states
- Enabling collaborative reasoning among medical AI agents
- Supporting data-driven decision-making in clinical settings
- Facilitating real-time monitoring and explainable AI for clinicians

Ultimately, the project seeks to find new ways to fight brain diseases and contribute to improved patient outcomes.

## Simple Roadmap

- **Core Integration**
  - Adaptive neural network engine
  - Multi-agent dialogue system
- **Biological State Simulation**
  - Energy, sleep, and mood modeling
- **Medical Data Training**
  - Real-world Alzheimer's and neurological disease datasets
- **Clinical Decision Support**
  - Risk stratification and scenario analysis
  - Explainable AI dashboard
- **MLOps Pipeline**
  - Automated retraining, model monitoring, and CI/CD integration
- **EHR Integration**
  - FHIR/HL7 compatibility, real-time data flows
- **Future Goals**
  - Advanced safety monitoring
  - Expanded memory consolidation algorithms
  - Visualization tools for network and agent states
  - API for custom agent behaviors
  - Web-based simulation dashboard
  - Clinical integration and deployment

---

## Migration Notes

### File Structure Updates 

**Important**: The project file structure has been reorganized. If you have existing code that imports from the old paths, please update your imports:

#### Updated Import Paths:
- **Old**: `from training.alzheimer_training_system import ...`
- **New**: `from files.training.alzheimer_training_system import ...`

#### Files Updated:
- Training modules are now located in `files/training/` directory
- All references to `training.*` imports have been updated to `files.training.*`
- Documentation and examples have been updated with correct import paths

#### For Developers:
If you encounter import errors like `ModuleNotFoundError: No module named 'training.alzheimer_training_system'`, update your imports to use the new path structure:

```python
# Old import (will fail)
from training.alzheimer_training_system import load_alzheimer_data

# New import (correct)
from files.training.alzheimer_training_system import load_alzheimer_data
```

## Latest Achievements

### Clinical Decision Support System (CDSS)
- Multi-condition risk assessment with quantitative scores and early warning triggers
- Interactive explainable AI dashboard for clinical feature importance and decision transparency
- EHR system integration (FHIR/HL7), real-time ingestion/export
- Regulatory compliance (HIPAA/FDA), audit trail and adverse event reporting
- Clinical performance: <100ms response, 92%+ sensitivity, 87%+ specificity

### Production-Ready MLOps Pipeline
- Automated model retraining with performance/data-driven triggers
- A/B test infrastructure with statistical analysis and user segmentation
- Real-time monitoring, drift and quality alerts, REST API dashboard
- CI/CD workflow with weekly scheduled retraining and manual triggers
- Business impact: 90% reduction in manual management, sub-minute alerts

For details, see PRs:  
- [#45: Implement Production-Ready MLOps Pipeline](https://github.com/V1B3hR/duetmind_adaptive/pull/45)
- [#46: Implement Clinical Decision Support System](https://github.com/V1B3hR/duetmind_adaptive/pull/46)

---

## 🚀 Installation & Setup

### Requirements
- **Python**: 3.10 or higher  
- **System**: Linux, macOS, or Windows with WSL2

### Quick Start

```bash
# Clone the repository
git clone https://github.com/V1B3hR/duetmind_adaptive.git
cd duetmind_adaptive

# Basic installation
pip install -e .

# Development installation (includes testing, linting, formatting tools)
pip install -e .[dev]

# Full installation with all features
pip install -e .[all]
```

### Installation Options

| Option | Command | Includes |
|--------|---------|----------|
| **Basic** | `pip install -e .` | Core functionality only |
| **Development** | `pip install -e .[dev]` | Core + testing & linting tools |  
| **Visualization** | `pip install -e .[viz]` | Core + plotting & dashboard tools |
| **Medical Imaging** | `pip install -e .[imaging]` | Core + DICOM, NIfTI, BIDS support |
| **MLOps** | `pip install -e .[mlops]` | Core + MLflow, DVC, monitoring |
| **Everything** | `pip install -e .[all]` | All optional features included |

### Using Make Commands

```bash
# Setup development environment  
make setup-env

# Install development dependencies only
make install-dev  

# Setup medical imaging pipeline
make imaging-setup

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Run full CI pipeline (lint + test + validate)
make ci-pipeline
```

### Verifying Installation

```bash
# Run basic tests
python -m pytest tests/test_safety_basic.py -v

# Check CLI tools are installed
duetmind --help
duetmind-train --help  
duetmind-api --help
```

### Configuration

All Python dependencies and versions are centrally managed in `pyproject.toml`. 

For backward compatibility, a minimal `requirements.txt` is maintained but `pyproject.toml` is the authoritative source.

---

## 🔧 Development

### Code Quality Tools

All configured in `pyproject.toml`:
- **Testing**: pytest with coverage
- **Formatting**: black (line length: 100)
- **Import sorting**: isort  
- **Linting**: flake8
- **Type checking**: mypy

### Running Development Tasks

```bash
# Format code
black .
isort .

# Run linting
flake8 --max-line-length=100 --ignore=E203,W503 .

# Run tests with coverage
pytest --cov=. --cov-report=term-missing

# Type checking  
mypy --ignore-missing-imports .
```

### Continuous Integration

The project uses GitHub Actions for CI/CD. See `.github/workflows/ci.yml` for the complete pipeline that runs on Python 3.10, 3.11, and 3.12.
