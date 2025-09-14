# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed broken import references after file structure reorganization
- Updated all imports from `training.*` to `files.training.*` to reflect moved files
- Updated documentation to use correct import paths

### Changed
- **BREAKING CHANGE**: Training modules import paths updated
  - Old: `from training.alzheimer_training_system import ...`
  - New: `from files.training.alzheimer_training_system import ...`

### Migration Guide
If you have existing code that imports from the old paths, update your imports:

```python
# Before
from training.alzheimer_training_system import load_alzheimer_data, train_model

# After  
from files.training.alzheimer_training_system import load_alzheimer_data, train_model
```

### Files Updated
- `files/training/comprehensive_training_simulation.py`
- `data_quality_monitor.py`
- `usage_examples.py`
- `COMPREHENSIVE_TRAINING_DOCUMENTATION.md`

## [Previous Releases]

### Major Features Implemented
- Clinical Decision Support System (CDSS)
- Production-Ready MLOps Pipeline  
- Multi-agent medical reasoning
- Real-time monitoring and alerting
- EHR integration (FHIR/HL7)
- Regulatory compliance features