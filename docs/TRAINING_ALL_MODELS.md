# Running Training for ALL Medical AI Models

This document demonstrates how to use the orchestrator to run training for ALL 7 medical AI models.

## All 7 Core Medical AI Models

The AiMedRes training orchestrator now includes all 7 core medical AI models:

1. **ALS (Amyotrophic Lateral Sclerosis)** - `train_als.py`
2. **Alzheimer's Disease** - `train_alzheimers.py`
3. **Parkinson's Disease** - `train_parkinsons.py`
4. **Brain MRI Classification** - `train_brain_mri.py`
5. **Cardiovascular Disease Prediction** - `train_cardiovascular.py`
6. **Diabetes Prediction** - `train_diabetes.py`
7. **Specialized Medical Agents** - `train_specialized_agents.py` ✨ **NEW**

## Usage Examples

### List All Available Models

```bash
python src/aimedres/cli/train.py --list --no-auto-discover
```

This will show all 7 core models with their configuration.

### Run Training for ALL Models (Sequential)

```bash
python src/aimedres/cli/train.py --epochs 50 --folds 5
```

This will train all 7 models sequentially with 50 epochs and 5-fold cross-validation.

### Run Training for ALL Models (Parallel)

```bash
python src/aimedres/cli/train.py --parallel --max-workers 4 --epochs 50 --folds 5
```

This will train all 7 models in parallel using up to 4 workers.

### Run Training for Specific Models Only

```bash
python src/aimedres/cli/train.py --only als alzheimers specialized_agents --epochs 50 --folds 5
```

This will train only the selected models (ALS, Alzheimer's, and Specialized Agents).

### Dry Run (Preview Commands Without Executing)

```bash
python src/aimedres/cli/train.py --dry-run --epochs 10 --folds 3
```

This will show what commands would be executed for all 7 models without actually running them.

## Model-Specific Features

### Specialized Medical Agents (NEW)

The Specialized Medical Agents model trains multiple specialized agent roles for enhanced medical diagnosis and consensus-based decision making. It includes:

- Training models for multiple specialized agent roles (Radiologists, Neurologists, etc.)
- Cross-validation with configurable folds
- Neural network training with configurable epochs
- Comprehensive metrics for multi-agent consensus
- Model persistence for deployment

**Parameters supported:**
- `--epochs`: Number of training epochs for neural networks
- `--folds`: Number of cross-validation folds
- `--output-dir`: Directory to save training outputs
- `--data-path`: Optional path to custom dataset

**Example:**
```bash
python src/aimedres/cli/train.py --only specialized_agents --epochs 50 --folds 5
```

## Test Verification

To verify that all 7 models are properly configured and ready for training:

```bash
python tests/integration/test_all_seven_models.py
```

This test suite verifies:
- All 7 core models are included in the orchestrator
- Each model is properly configured with the correct script path
- Commands are generated correctly for all models
- The specialized_agents model specifically is properly integrated

## Model Compatibility

| Model | Epochs | Folds | Sample | Batch | Output Dir |
|-------|--------|-------|--------|-------|------------|
| ALS | ✅ | ✅ | ❌ | ✅ | ✅ |
| Alzheimer's | ✅ | ✅ | ❌ | ❌ | ✅ |
| Parkinson's | ✅ | ✅ | ❌ | ❌ | ✅ |
| Brain MRI | ✅ | ❌ | ❌ | ❌ | ✅ |
| Cardiovascular | ✅ | ✅ | ❌ | ❌ | ✅ |
| Diabetes | ✅ | ✅ | ❌ | ❌ | ✅ |
| Specialized Agents | ✅ | ✅ | ❌ | ❌ | ✅ |

## Output

Training results for each model are saved to:
```
results/
├── als_comprehensive_results/
├── alzheimer_comprehensive_results/
├── parkinsons_comprehensive_results/
├── brain_mri_comprehensive_results/
├── cardiovascular_comprehensive_results/
├── diabetes_comprehensive_results/
└── specialized_agents_comprehensive_results/  ✨ NEW
```

## Summary

The AiMedRes training orchestrator now provides complete coverage for **ALL 7 core medical AI models**, enabling comprehensive medical AI research and development with a unified interface for training, validation, and deployment.
