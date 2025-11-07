# Training Scripts (Legacy)

**Version**: 1.0.0 | **Last Updated**: November 2025

**⚠️ DEPRECATED**: This directory contains legacy training scripts.

**Please use the canonical versions in `src/aimedres/training/` instead.**

The scripts in this directory are outdated and may not support all command-line flags.
The training orchestrator (`run_all_training.py`) automatically uses the canonical versions.

## Canonical Location

All training scripts have been moved to: **`src/aimedres/training/`**

The canonical scripts include:
- `train_alzheimers.py` - Alzheimer's disease classification (comprehensive, 806 lines)
- `train_brain_mri.py` - Brain MRI image classification
- `train_cardiovascular.py` - Cardiovascular disease prediction
- `train_diabetes.py` - Diabetes prediction
- `train_als.py` - ALS (Amyotrophic Lateral Sclerosis) prediction
- `train_parkinsons.py` - Parkinson's disease prediction

All canonical scripts support:
- `--output-dir` - Output directory for results
- `--epochs` - Number of training epochs
- `--folds` - Number of cross-validation folds (where applicable)

## Usage

To run training for all models:
```bash
python run_all_training.py
```

To run specific models:
```bash
python run_all_training.py --only als alzheimers
```

To run with custom parameters:
```bash
python run_all_training.py --epochs 20 --folds 5
```

See `run_all_training.py --help` for more options.