# Alzheimer's Disease Classification Training Pipeline

This document describes the complete machine learning training pipeline for Alzheimer's disease classification implemented in `train_alzheimers.py`.

## Overview

The training pipeline implements a comprehensive machine learning solution that:

1. **Downloads** the Kaggle Alzheimer's Disease dataset using `kagglehub`
2. **Preprocesses** the data with professional ML practices
3. **Trains** four classical ML models with cross-validation
4. **Trains** a tabular neural network (MLP)
5. **Reports** detailed metrics and performance statistics
6. **Saves** all trained models and preprocessing pipelines for inference

## Dataset

- **Source**: [rabieelkharoua/alzheimers-disease-dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) from Kaggle
- **Size**: 2,149 patients with 35 features
- **Target**: Binary classification (Diagnosis: 0 = No Alzheimer's, 1 = Alzheimer's)
- **Features**: Comprehensive medical, demographic, and lifestyle variables

## Features

### Data Preprocessing
- **ID Column Removal**: Automatically detects and removes ID columns (PatientID, etc.)
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Missing Value Imputation**: Median for numeric, most frequent for categorical
- **Feature Standardization**: StandardScaler normalization for numeric features
- **Target Auto-detection**: Automatically finds target column (Class, Diagnosis, etc.)

### Classical Machine Learning Models
All models trained with **5-fold stratified cross-validation**:

1. **Logistic Regression** - Linear baseline classifier
2. **Random Forest** - Ensemble tree-based method
3. **XGBoost** - Gradient boosting (if available)
4. **LightGBM** - Fast gradient boosting (if available)

### Neural Network
- **Architecture**: Multi-Layer Perceptron (MLP) with 4 hidden layers
- **Layers**: [256, 128, 64, 32] neurons with BatchNorm and Dropout
- **Training**: 50 epochs with Adam optimizer and learning rate scheduling
- **Regularization**: Batch normalization, dropout (30%), weight decay

### Metrics Reported
For each model, the following metrics are calculated and reported:
- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **Macro F1-Score**: F1 score averaged across classes
- **ROC AUC**: Area Under the Receiver Operating Characteristic curve

## Usage

### Basic Usage
```bash
# Train with default settings (downloads dataset automatically)
python3 train_alzheimers.py

# Train with custom parameters
python3 train_alzheimers.py --epochs 100 --folds 10 --output-dir my_results
```

### Advanced Usage
```bash
# Use your own dataset file
python3 train_alzheimers.py --data-path /path/to/alzheimer_data.csv

# Specify target column explicitly
python3 train_alzheimers.py --target-column "MyTargetColumn"

# Quick training for testing
python3 train_alzheimers.py --epochs 10 --folds 3
```

### Command Line Options
```
--data-path DATA_PATH          Path to dataset CSV (downloads from Kaggle if None)
--target-column TARGET_COLUMN  Target column name (auto-detects if None)
--output-dir OUTPUT_DIR        Output directory (default: outputs)
--epochs EPOCHS               Neural network epochs (default: 50)
--folds FOLDS                 Cross-validation folds (default: 5)
```

## Results

### Performance Summary (Latest Run)

**Classical Models (5-fold Cross-Validation):**
| Model | Accuracy | Balanced Accuracy | Macro F1 | ROC AUC |
|-------|----------|------------------|----------|---------|
| Logistic Regression | 83.29% ± 1.68% | 80.94% ± 1.51% | 81.43% ± 1.72% | 89.76% ± 1.18% |
| Random Forest | 93.86% ± 1.04% | 92.39% ± 1.53% | 93.14% ± 1.23% | 94.99% ± 0.60% |
| XGBoost | 95.11% ± 0.49% | 94.22% ± 0.76% | 94.61% ± 0.56% | 95.31% ± 0.66% |
| **LightGBM** | **95.44% ± 0.48%** | **94.68% ± 0.80%** | **94.98% ± 0.56%** | **95.33% ± 0.79%** |

**Neural Network (50 epochs):**
| Metric | Score |
|--------|-------|
| Accuracy | **97.53%** |
| Balanced Accuracy | **97.23%** |
| Macro F1 | **97.30%** |
| ROC AUC | **99.79%** |

## Output Structure

After training, the following files are generated in the `outputs/` directory:

```
outputs/
├── models/                     # Trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── neural_network.pth
│   └── neural_network_info.json
├── preprocessors/              # Data preprocessing pipeline
│   ├── preprocessor.pkl        # ColumnTransformer for features
│   ├── label_encoder.pkl       # Target encoder
│   └── feature_names.json      # Feature names after preprocessing
└── metrics/                    # Training results and reports
    ├── training_report.json    # Detailed JSON results
    └── training_summary.txt     # Human-readable summary
```

## Dependencies

### Required Dependencies
```bash
pip install numpy pandas scikit-learn kagglehub
```

### Optional Dependencies (automatically detected)
```bash
pip install xgboost lightgbm torch torchvision
```

If optional dependencies are missing, the script will skip those models and continue with available ones.

## Model Inference

All trained models and preprocessing pipelines are saved for inference. Example loading:

```python
import pickle
import torch
import json

# Load preprocessing pipeline
with open('outputs/preprocessors/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load label encoder
with open('outputs/preprocessors/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load a trained model (e.g., Random Forest)
with open('outputs/models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# For neural network
nn_state = torch.load('outputs/models/neural_network.pth')
with open('outputs/models/neural_network_info.json', 'r') as f:
    nn_info = json.load(f)

# Use for prediction on new data
# X_new = ... # New patient data
# X_processed = preprocessor.transform(X_new)
# predictions = model.predict(X_processed)
# predicted_classes = label_encoder.inverse_transform(predictions)
```

## Technical Implementation Details

### Data Preprocessing Pipeline
1. **ID Detection**: Automatically identifies columns containing 'id' or 'ID'
2. **Type Detection**: Separates numeric and categorical features
3. **Pipeline Construction**: Creates sklearn ColumnTransformer with appropriate transformers
4. **Feature Engineering**: Handles missing values and scaling consistently

### Cross-Validation Strategy
- **Method**: Stratified K-Fold to maintain class distribution
- **Metrics**: Multiple metrics computed for comprehensive evaluation
- **Final Training**: Models retrained on full dataset after CV evaluation

### Neural Network Architecture
```
Input Layer (32 features)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓  
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(32) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(2) → Softmax
```

### Error Handling
- **Graceful Degradation**: Missing optional dependencies are handled gracefully
- **Data Validation**: Input data is validated before processing
- **Target Auto-Detection**: Multiple target column names are attempted automatically
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Troubleshooting

### Common Issues

1. **"Target column 'Class' not found"**
   - Solution: Use `--target-column` to specify the correct column name
   - The script auto-detects common names: Class, Diagnosis, Target, Label

2. **"kagglehub not available"**
   - Solution: `pip install kagglehub`
   - Alternative: Use `--data-path` to specify a local CSV file

3. **"XGBoost not available"**
   - This is normal if XGBoost isn't installed
   - The script will continue with available models
   - Install with: `pip install xgboost lightgbm torch`

4. **Memory issues with large datasets**
   - Reduce batch size: The script uses reasonable defaults
   - Reduce epochs for testing: `--epochs 10`

### Performance Tips

1. **Quick Testing**: Use `--epochs 10 --folds 3` for faster iteration
2. **Full Training**: Use default settings for publication-quality results
3. **Large Datasets**: Monitor memory usage and adjust batch sizes if needed

## License and Citation

This implementation follows standard machine learning practices and uses established libraries. When using this code, please cite the appropriate papers for the algorithms used (XGBoost, LightGBM, etc.) and acknowledge the Kaggle dataset source.