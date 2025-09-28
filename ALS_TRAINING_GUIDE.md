# ALS Training Pipeline Guide

This guide explains how to run the ALS (Amyotrophic Lateral Sclerosis) progression prediction training pipeline as implemented in this repository.

## Quick Start

### Prerequisites

Install the required ML dependencies:
```bash
pip install -r requirements-ml.txt
```

### Basic Usage

Run the ALS training pipeline with default settings:
```bash
python training/train_als.py
```

### Advanced Usage

#### Using specific Kaggle datasets
```bash
# Primary ALS dataset (recommended)
python training/train_als.py --dataset-choice als-progression --task-type auto

# Alternative dataset (falls back to sample data if malformed)
python training/train_als.py --dataset-choice bram-als --task-type auto
```

#### Specifying task types
```bash
# Classification task
python training/train_als.py --task-type classification --target-column "Diagnosis (ALS)"

# Regression task  
python training/train_als.py --task-type regression --target-column "Progression_Rate"

# Auto-detect task type
python training/train_als.py --task-type auto
```

#### Configuring training parameters
```bash
python training/train_als.py \
    --epochs 50 \
    --folds 10 \
    --output-dir my_als_results \
    --dataset-choice als-progression
```

#### Using local data
```bash
python training/train_als.py --data-path /path/to/your/als_data.csv
```

## Datasets Supported

### 1. Primary ALS Dataset
- **URL**: https://www.kaggle.com/datasets/daniilkrasnoproshin/amyotrophic-lateral-sclerosis-als
- **Features**: Speech and audio analysis features for ALS detection
- **Size**: 64 samples, 135 features
- **Tasks**: Both classification and regression supported

### 2. Alternative Dataset
- **URL**: https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsbramcsv
- **Note**: This dataset has formatting issues; the pipeline gracefully falls back to synthetic sample data

### 3. Synthetic Sample Data
- **Generated automatically** when Kaggle datasets are unavailable
- **Features**: Clinically-relevant ALS progression indicators:
  - ALSFRS-R scores (ALS Functional Rating Scale)
  - Vital capacity measures (FVC)
  - Demographics and disease characteristics
  - Laboratory values
  - Treatment indicators
- **Size**: 1000 samples, 26 features
- **Targets**: 
  - `Progression_Rate` (regression)
  - `Fast_Progression` (classification)
  - `Survival_Months` (regression)

## Models Trained

### Classical Machine Learning
- Linear Regression / Logistic Regression
- Ridge Regression
- Random Forest
- Support Vector Machines (SVR/SVC)
- XGBoost (if available)

### Neural Networks
- Multi-layer Perceptron (MLP) with:
  - Batch normalization
  - Dropout regularization
  - Early stopping
  - Adaptive learning rate

## Output Structure

After training, results are saved in the specified output directory:

```
output_directory/
├── models/                     # Trained model files (.pkl, .pth)
├── preprocessors/              # Data preprocessing pipelines
├── metrics/                    # Detailed metrics
├── summary_report.txt          # Human-readable summary
├── training_report.json        # Detailed JSON report
└── visualizations/             # Plots and charts
```

## Example Results

### Classification Performance (Kaggle Dataset)
- **XGBoost**: 82.9% accuracy (±11.2%)
- **Neural Network**: 84.6% best validation accuracy
- **Random Forest**: 78.1% accuracy (±11.4%)

### Regression Performance (Sample Data)  
- **Linear Regression**: RMSE 0.096, R² 0.708
- **Random Forest**: RMSE 0.101, R² 0.675
- **Neural Network**: RMSE 0.436, R² 0.235

## Demo Script

Run the comprehensive demonstration:
```bash
python demo_als_training.py
```

This script demonstrates:
1. Kaggle dataset integration
2. Both classification and regression tasks
3. Error handling and fallback mechanisms
4. Sample data generation

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-path` | Path to local CSV file | None (downloads from Kaggle) |
| `--dataset-choice` | Kaggle dataset to use | `als-progression` |
| `--task-type` | ML task type | `auto` |
| `--target-column` | Target column name | Auto-detected |
| `--epochs` | Neural network epochs | 100 |
| `--folds` | Cross-validation folds | 5 |
| `--output-dir` | Output directory | `als_outputs` |

## Dependencies

Core requirements (install with `pip install -r requirements-ml.txt`):
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- torch >= 2.0.0
- xgboost >= 2.0.0
- kagglehub >= 0.3.0

## Troubleshooting

### Kaggle API Issues
If Kaggle download fails, the pipeline automatically uses synthetic sample data.

### Memory Issues
For large datasets, reduce batch size:
```bash
python training/train_als.py --epochs 50 # Reduces default batch size usage
```

### CUDA/GPU Support
PyTorch automatically detects and uses GPU if available. For CPU-only:
```bash
CUDA_VISIBLE_DEVICES="" python training/train_als.py
```