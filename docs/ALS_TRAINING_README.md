# ALS Progression Prediction Training Pipeline

This document outlines the structure and functionality of the training pipeline for ALS (Amyotrophic Lateral Sclerosis) progression prediction.

## Overview

The `train_als.py` script provides a comprehensive framework for training machine learning models to predict the progression of ALS based on patient data. The script supports both regression and classification tasks, with automatic dataset downloading from Kaggle, comprehensive preprocessing, and training of multiple ML models.

## Key Features

- **Data Loading**: Automatic download of ALS datasets from Kaggle or loading from local files
  - Primary dataset: daniilkrasnoproshin/amyotrophic-lateral-sclerosis-als
  - Alternative dataset: mpwolke/cusersmarildownloadsbramcsv
  - Fallback: Synthetic dataset generation for demonstration
- **Dual Task Support**: Supports both regression and classification tasks
  - Regression: Predicting progression rates, ALSFRS-R scores, survival time
  - Classification: Binary classification for fast vs. slow progression
- **Preprocessing**: Advanced preprocessing pipeline handling missing values, categorical encoding, and feature scaling
- **Model Training**: Training of both classical machine learning models and neural networks
  - **For Regression**: Linear Regression, Ridge Regression, Random Forest Regressor, SVR, XGBoost Regressor
  - **For Classification**: Logistic Regression, Random Forest Classifier, SVM, XGBoost Classifier
  - **Neural Networks**: Separate MLP architectures for regression and classification
- **Evaluation**: Task-specific evaluation metrics
  - **Regression**: RMSE, MAE, R² score with cross-validation
  - **Classification**: Accuracy, balanced accuracy, F1-score, ROC-AUC with cross-validation
- **Model Persistence**: Automatic saving of trained models and preprocessing pipelines
- **Reporting**: Detailed training reports in both JSON and human-readable formats

## Usage

### Basic Usage (Auto-detect task type)
```bash
python training/train_als.py
```

### Regression Task
```bash
python training/train_als.py \
    --task-type regression \
    --target-column Progression_Rate \
    --epochs 100
```

### Classification Task
```bash
python training/train_als.py \
    --task-type classification \
    --target-column Fast_Progression \
    --epochs 100
```

### Advanced Usage
```bash
python training/train_als.py \
    --data-path /path/to/your/dataset.csv \
    --epochs 100 \
    --folds 5 \
    --dataset-choice als-progression \
    --task-type auto \
    --target-column Survival_Months \
    --output-dir als_results
```

### Command Line Arguments

- `--data-path`: Path to local dataset CSV file (optional)
- `--epochs`: Number of epochs for neural network training (default: 100)
- `--folds`: Number of folds for cross-validation (default: 5)
- `--dataset-choice`: Kaggle dataset to use (choices: als-progression, bram-als)
- `--task-type`: Type of ML task (choices: regression, classification, auto)
- `--target-column`: Name of target column (auto-detects if not provided)
- `--output-dir`: Output directory for results (default: als_outputs)

## Output Structure

The training pipeline creates the following output structure:
```
als_outputs/
├── models/                          # Trained model files
│   ├── regression_linear_regression_model.pkl
│   ├── regression_ridge_regression_model.pkl
│   ├── regression_random_forest_model.pkl
│   ├── regression_svr_model.pkl
│   ├── regression_xgboost_model.pkl
│   ├── classification_logistic_regression_model.pkl
│   ├── classification_random_forest_model.pkl
│   ├── classification_svm_model.pkl
│   ├── classification_xgboost_model.pkl
│   └── neural_network_best.pth
├── preprocessors/                   # Data preprocessing components
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
├── metrics/                         # Evaluation metrics
├── visualizations/                  # Generated plots (if available)
├── training_report.json            # Detailed JSON report
└── summary_report.txt              # Human-readable summary
```

## Features Specific to ALS

The pipeline is optimized for ALS progression prediction with:

- **ALSFRS-R Integration**: Support for ALS Functional Rating Scale - Revised scores
- **Clinical Features**: Handles respiratory function (FVC), demographics, and disease characteristics
- **Progression Modeling**: Specialized features for modeling disease progression over time
- **Survival Analysis**: Support for survival time prediction tasks
- **Multi-task Architecture**: Separate neural network architectures for regression and classification
- **Clinical Metrics**: Evaluation metrics relevant for medical prediction tasks

## Example Output

### For Classification Task
```
Training completed successfully!
Task Type: classification
Results saved to: als_outputs

Best Classification Model Performance:
  Random Forest: 0.850 accuracy
Neural Network: 0.820 best validation accuracy
```

### For Regression Task
```
Training completed successfully!
Task Type: regression
Results saved to: als_outputs

Best Regression Model Performance:
  Random Forest: 0.439 RMSE, 0.223 R²
Neural Network: 0.526 best validation RMSE
```

## Target Variables

The pipeline can automatically detect and work with various target variables:

### Regression Targets
- `Progression_Rate`: Rate of disease progression (0-1 scale)
- `ALSFRS_R_Total`: Total ALSFRS-R score
- `FVC_percent`: Forced Vital Capacity percentage
- `Survival_Months`: Survival time in months

### Classification Targets
- `Fast_Progression`: Binary classification for fast vs. slow progression
- `status`: Disease status
- `class`: General classification label

## Dependencies

- pandas
- scikit-learn
- numpy
- torch (for neural networks)
- xgboost (optional, for XGBoost models)
- kagglehub (for automatic dataset downloading)
- matplotlib, seaborn (for visualizations)
