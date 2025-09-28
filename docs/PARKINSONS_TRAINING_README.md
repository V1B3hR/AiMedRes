# Parkinson's Disease Training Pipeline

This document outlines the structure and functionality of the training pipeline for Parkinson's disease classification.

## Overview

The `train_parkinsons.py` script provides a comprehensive framework for training machine learning models to classify Parkinson's disease based on patient data. The script supports automatic dataset downloading from Kaggle, comprehensive preprocessing, and training of multiple ML models.

## Key Features

- **Data Loading**: Automatic download of Parkinson's datasets from Kaggle or loading from local files
  - Primary dataset: vikasukani/parkinsons-disease-data-set
  - Alternative dataset: anonymous6623/uci-parkinsons-datasets
  - Fallback: Synthetic dataset generation for demonstration
- **Preprocessing**: Advanced preprocessing pipeline handling missing values, categorical encoding, and feature scaling
- **Model Training**: Training of both classical machine learning models and neural networks
  - Logistic Regression with cross-validation
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - XGBoost Classifier (if available)
  - Multi-Layer Perceptron (MLP) Neural Network
- **Evaluation**: Comprehensive evaluation metrics including accuracy, balanced accuracy, F1-score, and ROC-AUC
- **Model Persistence**: Automatic saving of trained models and preprocessing pipelines
- **Reporting**: Detailed training reports in both JSON and human-readable formats

## Usage

### Basic Usage
```bash
python training/train_parkinsons.py
```

### Advanced Usage
```bash
python training/train_parkinsons.py \
    --data-path /path/to/your/dataset.csv \
    --epochs 100 \
    --folds 5 \
    --dataset-choice vikasukani \
    --output-dir parkinsons_results
```

### Command Line Arguments

- `--data-path`: Path to local dataset CSV file (optional)
- `--epochs`: Number of epochs for neural network training (default: 100)
- `--folds`: Number of folds for cross-validation (default: 5)
- `--dataset-choice`: Kaggle dataset to use (choices: vikasukani, uci-parkinsons)
- `--output-dir`: Output directory for results (default: parkinsons_outputs)

## Output Structure

The training pipeline creates the following output structure:
```
parkinsons_outputs/
├── models/                          # Trained model files
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── xgboost_model.pkl
│   └── neural_network_best.pth
├── preprocessors/                   # Data preprocessing components
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
├── metrics/                         # Evaluation metrics
├── visualizations/                  # Generated plots (if available)
├── training_report.json            # Detailed JSON report
└── summary_report.txt              # Human-readable summary
```

## Features Specific to Parkinson's Disease

The pipeline is optimized for Parkinson's disease classification with:

- **Voice Analysis Features**: Handles typical voice measurement features like jitter, shimmer, and harmonic-to-noise ratio
- **UPDRS Integration**: Support for Unified Parkinson's Disease Rating Scale scores
- **Specialized Neural Architecture**: MLP design optimized for Parkinson's classification patterns
- **Clinical Metrics**: Evaluation metrics relevant for medical classification tasks

## Example Output

```
Training completed successfully!
Results saved to: parkinsons_outputs

Best Classical Model Performance:
  XGBoost: 0.980 accuracy
Neural Network: 0.810 best validation accuracy
```

## Dependencies

- pandas
- scikit-learn
- numpy
- torch (for neural networks)
- xgboost (optional, for XGBoost classifier)
- kagglehub (for automatic dataset downloading)
- matplotlib, seaborn (for visualizations)
