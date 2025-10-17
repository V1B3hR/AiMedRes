# train_als.py Usage Guide

## Quick Start

Run the training pipeline with default settings:
```bash
python training/train_als.py
```

Or from the src directory:
```bash
python src/aimedres/training/train_als.py
```

## Common Usage Examples

### Basic Training
```bash
# Default training (all models, synthetic data)
python training/train_als.py

# With custom output directory
python training/train_als.py --output-dir my_als_results

# Verbose output
python training/train_als.py --verbose
```

### Using Your Own Data
```bash
# Train with custom CSV data
python training/train_als.py --data-path /path/to/your_als_data.csv

# Specify target column and task type
python training/train_als.py --data-path data.csv --target-column disease_progression --task-type regression
```

### Model Selection
```bash
# Train only classical models (no neural network)
python training/train_als.py --disable-nn

# Train only neural network
python training/train_als.py --disable-classic

# Exclude specific models
python training/train_als.py --no-xgboost --no-svm --no-gb
```

### Hyperparameter Tuning
```bash
# Enable hyperparameter search
python training/train_als.py --use-hyperparam-search

# With custom number of iterations
python training/train_als.py --use-hyperparam-search --hyperparam-iter 50
```

### Neural Network Configuration
```bash
# Custom epochs and batch size
python training/train_als.py --epochs 100 --batch-size 128

# Quick test run
python training/train_als.py --epochs 10 --batch-size 64
```

### Advanced Analysis
```bash
# Compute SHAP values for explainability
python training/train_als.py --compute-shap

# Save feature matrix and disable importance plots
python training/train_als.py --save-feature-matrix --no-importance-plot
```

### Dataset Options
```bash
# Choose specific Kaggle dataset
python training/train_als.py --dataset-choice als-progression
python training/train_als.py --dataset-choice bram-als
```

## All Available Options

```
--data-path DATA_PATH         Path to CSV file (optional, uses Kaggle or synthetic data if not provided)
--output-dir OUTPUT_DIR        Output directory for models and results (default: als_outputs)
--dataset-choice {als-progression,bram-als}  Kaggle dataset to use
--task-type {regression,classification,auto}  Type of task (default: auto)
--target-column TARGET_COLUMN  Name of target column in dataset

--epochs EPOCHS                Number of training epochs for neural network (default: 80)
--batch-size BATCH_SIZE        Batch size for neural network (default: 32)
--folds FOLDS                  Number of cross-validation folds (default: 5)
--seed SEED                    Random seed for reproducibility (default: 42)

--disable-nn                   Disable neural network training
--disable-classic              Disable classical ML models

--use-hyperparam-search        Enable hyperparameter search
--hyperparam-iter ITER         Number of hyperparameter search iterations (default: 20)

--no-xgboost                   Exclude XGBoost models
--no-svm                       Exclude SVM models
--no-gb                        Exclude Gradient Boosting models

--compute-shap                 Compute SHAP values for feature importance
--no-importance-plot           Disable feature importance plots
--save-feature-matrix          Save processed feature matrix

--verbose                      Enable verbose logging
```

## Output Structure

After running the script, you'll find:

```
als_outputs/
├── models/                    # Trained model files (.pkl, .pth)
│   ├── regression_*.pkl
│   └── neural_best.pth
├── metrics/                   # Performance metrics
│   ├── summary.txt
│   ├── model_metrics.csv
│   ├── training_report.json
│   └── neural_results.json
├── visualizations/            # Feature importance plots and SHAP
│   ├── feature_importance_*.csv
│   └── feature_importance_*.png
├── preprocessors/             # Preprocessing pipelines
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
└── artifacts/                 # Additional artifacts
    ├── X_processed.npy
    └── y.npy
```

## Requirements

### Minimum Dependencies
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0

### Optional Dependencies
- torch >= 2.0.0 (for neural network training)
- xgboost (for XGBoost models)
- shap (for SHAP analysis)
- kagglehub (for downloading Kaggle datasets)
- matplotlib >= 3.5.0 (for visualizations)
- seaborn >= 0.12.0 (for visualizations)

## Tips

1. **Start Small**: Test with fewer epochs first to ensure everything works
   ```bash
   python training/train_als.py --epochs 5 --verbose
   ```

2. **Custom Data Format**: Your CSV should have:
   - Feature columns (numerical and/or categorical)
   - A target column (for regression or classification)
   - Optional: subject ID column (will be automatically excluded)

3. **Reproducibility**: Use the same seed for reproducible results
   ```bash
   python training/train_als.py --seed 42
   ```

4. **Memory Issues**: If running out of memory, try:
   - Smaller batch size: `--batch-size 16`
   - Disable some models: `--no-svm --no-gb`
   - Disable neural network: `--disable-nn`

5. **Speed Up Training**: 
   - Reduce epochs: `--epochs 20`
   - Fewer folds: `--folds 3`
   - Disable hyperparameter search
