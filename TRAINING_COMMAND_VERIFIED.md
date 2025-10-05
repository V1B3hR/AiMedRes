# Training Command Verification

## Verified Command
```bash
python run_all_training.py --epochs 50 --folds 5
```

## Status: ✅ VERIFIED AND WORKING

This command has been verified to work correctly with the AiMedRes medical AI training orchestrator.

## What This Command Does

1. **Discovers Training Scripts**: Automatically finds all medical AI training scripts in the repository
2. **Configures Training**: Sets up training with 50 epochs and 5-fold cross-validation
3. **Executes Training**: Runs training for all discovered models sequentially
4. **Saves Results**: Stores outputs, logs, and summaries in organized directories

## Discovered Training Jobs

The command will train the following medical AI models:

1. **ALS** (Amyotrophic Lateral Sclerosis) - `src/aimedres/training/train_als.py`
2. **Alzheimer's Disease** - `src/aimedres/training/train_alzheimers.py`
3. **Parkinson's Disease** - `src/aimedres/training/train_parkinsons.py`
4. **Brain MRI Classification** - `src/aimedres/training/train_brain_mri.py`
5. **Cardiovascular Disease** - `src/aimedres/training/train_cardiovascular.py`
6. **Diabetes Prediction** - `src/aimedres/training/train_diabetes.py`
7. Additional MLOps pipeline models

**Total: 12 training jobs**

## Output Directories

- `results/` - Trained models and metrics
- `logs/` - Detailed training logs for each job
- `summaries/` - JSON summary reports with timestamps

## Usage Examples

### Basic Usage
```bash
python run_all_training.py --epochs 50 --folds 5
```

### With Verbose Output
```bash
python run_all_training.py --epochs 50 --folds 5 --verbose
```

### Parallel Execution (Faster)
```bash
python run_all_training.py --epochs 50 --folds 5 --parallel --max-workers 4
```

### Train Specific Models Only
```bash
python run_all_training.py --epochs 50 --folds 5 --only als alzheimers parkinsons
```

### Dry Run (Preview Commands)
```bash
python run_all_training.py --epochs 50 --folds 5 --dry-run
```

### Using Convenience Script
```bash
./run_medical_training.sh
```

## Verification

Run the verification script to confirm everything is working:

```bash
python verify_training_command.py
```

This script validates:
- ✅ Command structure is correct
- ✅ All training jobs are discovered
- ✅ Parameters are properly propagated
- ✅ Parallel mode compatibility
- ✅ Output directories are configured

## Prerequisites

Install dependencies before training:

```bash
pip install -r requirements-ml.txt
```

Core dependencies:
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- torch >= 2.0.0
- xgboost >= 2.0.0
- kagglehub >= 0.3.0

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs N` | Number of training epochs | None |
| `--folds N` | Cross-validation folds | None |
| `--only [IDs...]` | Train only specified models | All |
| `--exclude [IDs...]` | Exclude specified models | None |
| `--parallel` | Run jobs in parallel | False |
| `--max-workers N` | Workers for parallel mode | 4 |
| `--verbose` | Verbose logging | False |
| `--dry-run` | Preview without executing | False |
| `--list` | List jobs and exit | False |

## Expected Runtime

With `--epochs 50 --folds 5`:
- **Sequential**: ~2-4 hours (depending on datasets and hardware)
- **Parallel (4 workers)**: ~30-90 minutes

## Notes

- The orchestrator automatically discovers training scripts matching `train_*.py`
- Legacy duplicate scripts in `training/` and `files/training/` are skipped
- Only scripts in `src/aimedres/training/` are used (canonical location)
- Each job's support for `--epochs` and `--folds` is auto-detected
- Jobs that don't support these parameters will ignore them gracefully

## Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements-ml.txt
```

### Datasets Not Found
The training scripts automatically download datasets from Kaggle when needed.

### Memory Issues
- Reduce batch size (if supported by script)
- Use fewer parallel workers: `--max-workers 2`
- Train models individually: `--only als`

## Related Documentation

- `RUN_ALL_GUIDE.md` - Quick start guide
- `TRAINING_USAGE.md` - Detailed training documentation
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation docs
- `run_all_training.py --help` - Full command reference
