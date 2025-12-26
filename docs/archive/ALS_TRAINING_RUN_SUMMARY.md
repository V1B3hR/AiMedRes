# ALS Training Run Summary

**Version**: 1.0.0 | **Last Updated**: November 2025

## Overview

This document summarizes ALS (Amyotrophic Lateral Sclerosis) training implementation and results.

## Overview
Successfully executed the `train_als.py` training pipeline for ALS (Amyotrophic Lateral Sclerosis) progression prediction.

## Script Location
- Primary: `/training/train_als.py`
- Alternative: `/src/aimedres/training/train_als.py`

## Training Configuration
- **Dataset**: Synthetic ALS dataset (800 samples, 26 features)
- **Task Type**: Regression (auto-detected)
- **Target**: Progression_Rate
- **Cross-validation**: 5-fold KFold
- **Random Seed**: 42
- **Models Trained**: 5 classical ML models

## Results

### Best Performing Model: Ridge Regression
- **RMSE**: 0.0761
- **R²**: 0.7880
- **MAE**: 0.0608

### All Models Performance

| Model | RMSE | R² Score | MAE |
|-------|------|----------|-----|
| Ridge | 0.0761 | 0.7880 | 0.0608 |
| LinearRegression | 0.0761 | 0.7879 | 0.0608 |
| RandomForest | 0.0810 | 0.7596 | 0.0647 |
| GradientBoosting | 0.0820 | 0.7539 | 0.0650 |
| SVR | 0.0890 | 0.7102 | 0.0714 |

## Generated Outputs

### Models
All trained models saved as pickle files in `als_outputs/models/`:
- regression_LinearRegression.pkl
- regression_Ridge.pkl
- regression_RandomForest.pkl
- regression_GradientBoosting.pkl
- regression_SVR.pkl

### Metrics and Reports
- `als_outputs/metrics/summary.txt` - Human-readable summary
- `als_outputs/metrics/model_metrics.csv` - Detailed metrics in CSV format
- `als_outputs/metrics/training_report.json` - Complete training report with configuration

### Feature Importance
- `als_outputs/visualizations/feature_importance_RandomForest.csv`
- `als_outputs/visualizations/feature_importance_GradientBoosting.csv`

### Preprocessing Artifacts
- `als_outputs/preprocessors/preprocessor.pkl` - ColumnTransformer for feature preprocessing

## Execution Time
Approximately 0.08 minutes (~5 seconds)

## Notes
- Neural network training was skipped due to PyTorch not being installed
- XGBoost models were not trained due to missing dependency
- Training used synthetic data for development/testing purposes
- For production use, real ALS progression datasets should be provided via the `--data-path` parameter

## Command Used
```bash
python /home/runner/work/AiMedRes/AiMedRes/training/train_als.py
```

## Next Steps
To enhance the training:
1. Install PyTorch for neural network training: `pip install torch`
2. Install XGBoost for additional models: `pip install xgboost`
3. Use real ALS dataset: `python train_als.py --data-path /path/to/als_data.csv`
4. Enable hyperparameter search: `python train_als.py --use-hyperparam-search`
5. Compute SHAP values: `python train_als.py --compute-shap`
