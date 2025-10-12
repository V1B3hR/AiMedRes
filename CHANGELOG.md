# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Pre-Release Cleanup (2025)

### Added
- Documentation index at `docs/INDEX.md` for easy navigation
- Consolidated all documentation under `docs/` directory
- Created `docs/archive/` for historical implementation summaries

### Changed
- Moved 40+ documentation files from root to `docs/` directory
- Updated README.md to point to new documentation locations
- Reorganized documentation structure for better discoverability
- **BREAKING CHANGE**: Training modules import paths updated
  - Old: `from training.alzheimer_training_system import ...`
  - New: `from aimedres.training.alzheimer_training_system import ...`

### Removed
- Removed 18 legacy `.shim` compatibility wrapper files
- Cleaned up unused imports in core modules (agent.py, cognitive_engine.py, config.py, labyrinth.py)
- Archived implementation summaries and phase completion documents

### Fixed
- Fixed import paths in examples and tests to use new `aimedres.*` module structure
- Corrected enterprise demo imports to use proper constants from core modules
- Fixed broken import references after file structure reorganization
- Updated all imports from `training.*` to `aimedres.training.*` to reflect moved files
- Updated documentation to use correct import paths

## [Previous Releases]

### Major Features Implemented
- Clinical Decision Support System (CDSS)
- Production-Ready MLOps Pipeline  
- Multi-agent medical reasoning
- Real-time monitoring and alerting
- EHR integration (FHIR/HL7)
- Regulatory compliance features