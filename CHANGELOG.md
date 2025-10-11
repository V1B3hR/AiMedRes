# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-10-11

### Pre-Release Cleanup (v0.2.0 preparation)

#### Removed
- **18 deprecated .shim marker files** - Removed backward compatibility markers
- **Legacy shim directories** - Removed `training/`, `agent_memory/`, `files/training/` compatibility shims
- **50+ implementation summary documents** - Moved to `docs/archive/` for historical reference
- Cleaned up duplicate training modules across multiple locations

#### Changed
- **Documentation reorganization**
  - Moved architecture docs to `docs/`
  - Created `docs/guides/` for training and usage guides
  - Created `docs/archive/` for historical implementation notes
  - Added comprehensive `docs/README.md` index
  - Root now contains only: README.md, CHANGELOG.md, CONTRIBUTING.md, LICENSE, security.md

#### Fixed
- **Critical code errors**
  - Fixed undefined `results` variable in `production_agent.py`
  - Added conditional `mlflow` import in `multimodal.py` to handle missing dependency
  - Fixed missing `start_time` variable in `orchestration.py`
- **Import paths**
  - Updated 4 test files to use canonical `aimedres.training` imports
  - All code now uses `aimedres.*` import paths

#### Notes
- All training modules now exclusively in `src/aimedres/training/`
- All agent memory modules in `src/aimedres/agent_memory/`
- Clean, professional structure for external partner release

---

## [0.1.0] - Previous Releases

### Major Features Implemented
- Clinical Decision Support System (CDSS)
- Production-Ready MLOps Pipeline  
- Multi-agent medical reasoning
- Real-time monitoring and alerting
- EHR integration (FHIR/HL7)
- Regulatory compliance features