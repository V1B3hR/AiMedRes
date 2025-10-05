# AiMedRes Training Scripts

This directory contains the **canonical** training scripts for all disease prediction models.

## Available Training Scripts

| Script | Disease | Features | Lines |
|--------|---------|----------|-------|
| `train_alzheimers.py` | Alzheimer's Disease | Full pipeline with Kaggle dataset download | 806 |
| `train_als.py` | ALS (Amyotrophic Lateral Sclerosis) | Multiple datasets, hyperparameter search | 1053 |
| `train_parkinsons.py` | Parkinson's Disease | Multiple datasets, SHAP analysis | ~800 |
| `train_brain_mri.py` | Brain MRI Classification | Image classification pipeline | ~700 |
| `train_cardiovascular.py` | Cardiovascular Disease | Clinical data prediction | ~800 |
| `train_diabetes.py` | Diabetes Prediction | Clinical risk assessment | ~800 |

## Command-Line Interface

All scripts support a consistent command-line interface:

```bash
python src/aimedres/training/train_<disease>.py [OPTIONS]
```

### Common Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--data-path` | str | Path to custom dataset CSV | None (auto-download) |
| `--output-dir` | str | Output directory for models/metrics | `<disease>_outputs` |
| `--epochs` | int | Number of training epochs | 30-100 (varies) |
| `--folds` | int | Cross-validation folds | 5 |
| `--target-column` | str | Target column name | Auto-detected |
| `--batch-size` | int | Training batch size | 32 |
| `--seed` | int | Random seed for reproducibility | 42 |

### Advanced Options

Some scripts support additional options:
- `--dataset-choice` - Select from multiple built-in datasets (ALS, Parkinson's)
- `--use-hyperparam-search` - Enable hyperparameter optimization
- `--compute-shap` - Generate SHAP explanations
- `--disable-nn` / `--disable-classic` - Skip specific model types

## Usage Examples

### Quick Start - Run Single Model

```bash
# Train Alzheimer's model with default settings
python src/aimedres/training/train_alzheimers.py

# Train with custom parameters
python src/aimedres/training/train_als.py --epochs 50 --folds 3 --output-dir my_results
```

### Run All Training Models

Use the orchestrator to run all models:

```bash
# Run all models with default settings
python run_all_training.py

# Run all models with custom epochs/folds
python run_all_training.py --epochs 20 --folds 5

# Run specific models only
python run_all_training.py --only als alzheimers parkinsons

# Run in parallel (faster)
python run_all_training.py --parallel --max-workers 4
```

### Dry Run (Preview Commands)

```bash
# See what would be executed without running
python run_all_training.py --dry-run --epochs 10
```

### List Available Jobs

```bash
# List all discovered training jobs
python run_all_training.py --list
```

## Output Structure

Each training script creates the following output structure:

```
<output-dir>/
├── models/                      # Trained model files
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── neural_network.pth
│   └── neural_network_info.json
├── preprocessors/               # Data preprocessing components
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   └── feature_names.json
└── metrics/                     # Training reports and metrics
    ├── training_report.json
    ├── training_summary.txt
    └── feature_importance.png (if generated)
```

## Key Features

### ✅ Automatic Dataset Handling
- Downloads datasets from Kaggle automatically (when available)
- Auto-detects target columns
- Handles missing values and categorical features
- Removes ID columns automatically

### ✅ Multiple ML Algorithms
- Logistic Regression (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting) - if installed
- LightGBM (light gradient boosting) - if installed
- Neural Network (PyTorch MLP)

### ✅ Robust Evaluation
- Stratified k-fold cross-validation
- Multiple metrics (Accuracy, F1, ROC-AUC, Balanced Accuracy)
- Statistical significance testing
- Feature importance analysis

### ✅ Production Ready
- Model persistence for inference
- Preprocessing pipeline saving
- Comprehensive logging and reporting
- GPU support for neural networks (when available)

## Dependencies

### Required
```bash
pip install numpy pandas scikit-learn torch kagglehub
```

### Optional (for enhanced models)
```bash
pip install xgboost lightgbm shap matplotlib seaborn
```

Or install all at once:
```bash
pip install -r requirements-ml.txt
```

## Integration with Training Orchestrator

The `run_all_training.py` script automatically discovers and runs these scripts with:

1. **Auto-discovery**: Finds all `train_*.py` scripts in this directory
2. **Flag detection**: Detects which flags each script supports
3. **Unified execution**: Runs all scripts with consistent parameters
4. **Result aggregation**: Collects and summarizes results
5. **Parallel execution**: Optional parallel training for speed

## GitHub Actions Integration

You can trigger training runs via GitHub Actions:

1. Go to **Actions** tab in your GitHub repository
2. Select **Training Orchestrator** workflow
3. Click **Run workflow**
4. Configure parameters (epochs, folds, which models to run)
5. Monitor execution and download results

## Migration Notes

**Important**: If you have scripts in the old `training/` or `files/training/` directories:

- Those are legacy locations and are skipped by auto-discovery
- Use the scripts in this directory (`src/aimedres/training/`) instead
- The legacy scripts don't support all command-line flags
- Update any custom workflows to reference `src/aimedres/training/`

## Contributing

When adding new training scripts:

1. Name files as `train_<disease>.py`
2. Support standard flags: `--output-dir`, `--epochs`, `--folds`
3. Use argparse for command-line arguments
4. Include docstrings and help text
5. Save models and metrics to output directory
6. Test with `run_all_training.py --list` to verify discovery

## Troubleshooting

### Script not discovered
- Ensure filename matches `train_*.py` pattern
- Check script is in `src/aimedres/training/` directory
- Run `python run_all_training.py --list --verbose` to debug

### Flags not detected
- Ensure argparse definitions are in first 40KB of file
- Use standard flag names: `--output-dir`, `--epochs`, `--folds`
- Check with dry run: `python run_all_training.py --dry-run`

### Import errors
- Install dependencies: `pip install -r requirements-ml.txt`
- Check Python version: requires Python 3.8+

### Out of memory
- Reduce batch size: `--batch-size 16`
- Reduce epochs: `--epochs 10`
- Use fewer folds: `--folds 3`
