# AiMedRes Structure Consolidation

## Overview

The AiMedRes codebase has been consolidated into a clean, organized structure under `src/aimedres/`. This migration improves maintainability and follows Python packaging best practices.

## What Changed

### Directory Structure

**Before:**
- Root-level Python modules: `specialized_medical_agents.py`, `neuralnet.py`, `duetmind.py`, etc.
- `agent_memory/` at root level
- Duplicate `training/` directories in multiple locations:
  - `training/` (root)
  - `files/training/`
  - `src/aimedres/training/` (partial)

**After:**
- All modules consolidated under `src/aimedres/`:
  - `src/aimedres/agents/` - Specialized medical agents
  - `src/aimedres/agent_memory/` - Memory consolidation system
  - `src/aimedres/training/` - All disease-specific training pipelines
  - `src/aimedres/core/` - Core components
  - `src/aimedres/security/` - Security modules
  - `src/aimedres/api/` - REST API
  - `src/aimedres/utils/` - Utilities

### Import Changes

**Old imports (deprecated but still work via compatibility shims):**
```python
from agent_memory.memory_consolidation import MemoryConsolidator
from training.train_als import ALSTrainingPipeline
from specialized_medical_agents import MedicalKnowledgeAgent
```

**New imports (recommended):**
```python
from aimedres.agent_memory.memory_consolidation import MemoryConsolidator
from aimedres.training.train_als import ALSTrainingPipeline
from aimedres.agents.specialized_medical_agents import MedicalKnowledgeAgent
```

### Training Pipelines Consolidated

All disease-specific training pipelines are now in `src/aimedres/training/`:
- `train_alzheimers.py` - Alzheimer's disease classification
- `train_als.py` - ALS classification
- `train_parkinsons.py` - Parkinson's disease classification
- `train_brain_mri.py` - Brain MRI image classification
- `train_cardiovascular.py` - Cardiovascular risk prediction
- `train_diabetes.py` - Diabetes classification

Supporting modules:
- `alzheimer_training_system.py` - Alzheimer's training utilities
- `data_processing.py` - Data processing utilities
- `model_validation.py` - Model validation
- `cross_validation.py` - Cross-validation utilities

Infrastructure modules:
- `automation_system.py` - Training automation
- `custom_pipeline.py` - Dynamic pipeline builder
- `orchestration.py` - Workflow orchestration
- `automl.py` - AutoML capabilities

## Backward Compatibility

Compatibility shims have been created in the old locations to maintain backward compatibility:
- `agent_memory/__init__.py` - Redirects to `aimedres.agent_memory`
- `training/__init__.py` - Redirects to `aimedres.training`
- `files/training/__init__.py` - Redirects to `aimedres.training`
- `specialized_medical_agents.py.shim` - Redirects to `aimedres.agents.specialized_medical_agents`

These shims emit deprecation warnings but allow existing code to continue working.

## Migration Guide for Developers

1. **Update imports** in your code to use the new `aimedres.*` paths
2. **Update PYTHONPATH** to include the `src/` directory if running scripts directly
3. **Use new import patterns** in new code

## Benefits

1. **Cleaner repository structure** - All code is organized under `src/aimedres/`
2. **Better discoverability** - Related modules are grouped together
3. **Improved maintainability** - No more duplicate files in multiple locations
4. **Follows best practices** - Standard Python package layout
5. **Easier testing** - Clear module boundaries
6. **Simpler imports** - Consistent import paths

## Files Consolidated

### agent_memory/
- Moved from root to `src/aimedres/agent_memory/`
- All 5 modules consolidated
- New `__init__.py` exports key classes

### agents/
- Created new `src/aimedres/agents/` directory
- Moved `specialized_medical_agents.py`
- New `__init__.py` exports key classes

### training/
- Consolidated from 3 locations into `src/aimedres/training/`
- Includes all disease-specific trainers
- Includes all supporting modules
- Updated `__init__.py` exports all trainers

## Testing

All imports have been updated in:
- Test files (`tests/`)
- Example scripts (`examples/`)
- Demo scripts (root level)
- API modules (`api/`)

The consolidation maintains backward compatibility while providing a clear migration path to the new structure.
