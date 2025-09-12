# Structured Alzheimer's Disease Training Pipeline

## Overview

The Structured Alzheimer's Disease Training Pipeline provides a comprehensive, production-ready framework for training machine learning models to predict Alzheimer's disease using structured tabular data. The pipeline supports multiple algorithms, ensemble methods, early stopping, multi-seed reproducible training, and extensive evaluation metrics.

## Key Features

- **Multi-Algorithm Support**: Logistic Regression, Random Forest, and Multi-Layer Perceptron (MLP)
- **Ensemble Methods**: Soft/hard voting classifiers with configurable weights
- **Early Stopping**: Intelligent early stopping for neural networks with validation monitoring
- **Multi-Seed Training**: Reproducible training across multiple random seeds with statistical aggregation
- **Comprehensive Evaluation**: Accuracy, F1-scores (macro/micro/weighted), ROC AUC, confusion matrices
- **Automated Preprocessing**: Intelligent feature detection, missing value imputation, scaling, and encoding
- **Configuration-Driven**: YAML-based configuration for easy experimentation
- **Artifact Management**: Model persistence, preprocessing pipelines, and detailed metrics logging

## Architecture

### Core Components

1. **StructuredAlzTrainer**: Main training orchestrator
2. **EarlyStoppingMLP**: Advanced early stopping wrapper for neural networks  
3. **Configuration System**: YAML-based parameter management
4. **CLI Interface**: Command-line scripts for training and evaluation
5. **Evaluation Pipeline**: Comprehensive model assessment tools

### Directory Structure

```
src/duetmind_adaptive/training/
├── structured_alz_trainer.py          # Core trainer implementation
├── configs/
│   ├── structured_alz_baseline.yaml   # Baseline configuration
│   └── structured_alz_ensemble.yaml   # Ensemble configuration
└── __init__.py                        # Module exports

scripts/
├── train_alzheimers_structured.py     # Training CLI
└── eval_alzheimers_structured.py      # Evaluation CLI

docs/
└── STRUCTURED_ALZ_TRAINING.md         # This documentation

metrics/structured_alz/                 # Output directory
├── runs/
│   ├── seed_42/
│   │   ├── best_model.pkl
│   │   ├── final_model.pkl
│   │   ├── preprocessing.pkl
│   │   ├── epoch_metrics.csv
│   │   └── run_metrics.json
│   ├── seed_1337/
│   └── seed_2025/
├── aggregate_metrics.json
└── aggregate_summary.md
```

## Configuration

### Baseline Configuration

The baseline configuration (`structured_alz_baseline.yaml`) provides sensible defaults:

```yaml
profile: baseline
epochs: 80
patience: 8
batch_size: 64
validation_split: 0.2
seeds: [42, 1337, 2025]
models: [logreg, random_forest, mlp]
ensemble: false
metric_primary: macro_f1
class_weight: balanced
```

### Ensemble Configuration

The ensemble configuration (`structured_alz_ensemble.yaml`) enables voting classifiers:

```yaml
profile: ensemble
ensemble: true
ensemble_params:
  voting: soft
  estimators: [logreg, random_forest, mlp]
  weights: null  # Equal weights
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Maximum training epochs | 80 |
| `patience` | Early stopping patience | 8 |
| `batch_size` | Training batch size | 64 |
| `validation_split` | Validation set proportion | 0.2 |
| `seeds` | Random seeds for reproducibility | [42, 1337, 2025] |
| `models` | Algorithms to train | [logreg, random_forest, mlp] |
| `ensemble` | Enable ensemble methods | false |
| `metric_primary` | Primary optimization metric | macro_f1 |
| `class_weight` | Handle class imbalance | balanced |

## Usage Examples

### Basic Training

Train models using the baseline configuration:

```bash
python scripts/train_alzheimers_structured.py \
  --data-path data/alzheimer_dataset.csv
```

### Ensemble Training

Enable ensemble methods with soft voting:

```bash
python scripts/train_alzheimers_structured.py \
  --data-path data/alzheimer_dataset.csv \
  --config src/duetmind_adaptive/training/configs/structured_alz_ensemble.yaml \
  --ensemble
```

### Custom Configuration

Override specific parameters:

```bash
python scripts/train_alzheimers_structured.py \
  --data-path data/alzheimer_dataset.csv \
  --epochs 100 \
  --batch-size 32 \
  --override-seeds 42 1337 2025 999 \
  --target-column diagnosis
```

### Evaluation

Evaluate a single trained model:

```bash
python scripts/eval_alzheimers_structured.py \
  --model-dir metrics/structured_alz/runs/seed_42 \
  --test-data data/test_set.csv
```

Evaluate all models from multi-seed training:

```bash
python scripts/eval_alzheimers_structured.py \
  --aggregate-dir metrics/structured_alz \
  --test-data data/test_set.csv
```

## Data Requirements

### Input Format

The pipeline accepts CSV or Parquet files with the following expectations:

- **Target Column**: Automatically detected from common names (diagnosis, label, class, Group) or specified via `--target-column`
- **Feature Columns**: Mix of numerical and categorical features supported
- **Missing Values**: Handled automatically with configurable imputation strategies

### Example Dataset Structure

```csv
age,gender,education_level,mmse_score,cdr_score,apoe_genotype,diagnosis
72,F,12,24,0.5,E3/E4,MCI
65,M,16,28,0.0,E3/E3,Normal
78,F,8,18,1.0,E4/E4,Dementia
```

### Preprocessing Pipeline

1. **Missing Value Detection**: Warns if >10% missing values in any column
2. **Feature Type Detection**: Automatic classification of numerical vs categorical features
3. **Imputation**: Configurable strategies (median/mean for numerical, most_frequent for categorical)
4. **Scaling**: StandardScaler for numerical features  
5. **Encoding**: OneHotEncoder for categorical features with unknown value handling
6. **Pipeline Persistence**: Complete preprocessing pipeline saved for inference

## Model Algorithms

### Logistic Regression

- **Use Case**: Linear baseline model with high interpretability
- **Parameters**: L-BFGS solver, balanced class weights, automatic multi-class handling
- **Strengths**: Fast training, probabilistic outputs, feature importance via coefficients

### Random Forest

- **Use Case**: Robust ensemble method with feature importance
- **Parameters**: 300 trees, balanced class weights, no maximum depth limit
- **Strengths**: Handles non-linear relationships, built-in feature selection, robust to outliers

### Multi-Layer Perceptron (MLP)

- **Use Case**: Neural network for complex pattern learning
- **Architecture**: [64, 32] hidden layers with ReLU activation
- **Training**: Adam optimizer with early stopping and batch processing
- **Strengths**: Universal approximation, handles complex interactions

### Ensemble Methods (Optional)

- **Voting Classifier**: Combines predictions from multiple algorithms
- **Soft Voting**: Uses predicted probabilities for decision making
- **Hard Voting**: Uses majority vote from predicted classes
- **Configurable Weights**: Optional custom weighting of ensemble members

## Early Stopping Implementation

The EarlyStoppingMLP wrapper provides sophisticated training control:

### Features

- **Validation Monitoring**: Tracks macro F1 or accuracy on validation set
- **Patience Mechanism**: Stops training after N epochs without improvement
- **Best Weight Restoration**: Automatically restores weights from best epoch
- **Batch Processing**: Efficient mini-batch training with partial_fit

### Configuration

```yaml
early_stopping:
  enabled: true
  monitor: macro_f1      # or 'accuracy'
  patience: 8            # epochs to wait
  min_delta: 0.000001    # minimum improvement threshold
  restore_best_weights: true
```

## Multi-Seed Training

### Reproducibility Strategy

1. **Deterministic Seeding**: Sets random seeds for Python, NumPy, and scikit-learn
2. **Independent Runs**: Each seed produces completely independent training run
3. **Statistical Aggregation**: Mean and standard deviation across all seeds
4. **Artifact Isolation**: Separate directories for each seed's outputs

### Aggregated Metrics

- **Performance Statistics**: Mean ± standard deviation for key metrics
- **Model Selection Frequency**: Which algorithm performed best per seed
- **Confidence Intervals**: Statistical confidence in reported performance
- **Reproducibility Assessment**: Variance analysis across seeds

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Overall correct prediction rate
- **Macro F1**: Unweighted mean F1 across all classes
- **Weighted F1**: Class-frequency weighted F1 score  
- **Micro F1**: Global F1 considering all predictions
- **Per-Class F1**: Individual F1 score for each diagnostic category

### Probabilistic Metrics

- **ROC AUC**: Area under ROC curve (binary classification)
- **ROC AUC OvR**: One-vs-Rest ROC AUC (multi-class classification)
- **Precision-Recall Curves**: Available in evaluation pipeline

### Diagnostic Tools

- **Confusion Matrix**: Class-by-class prediction accuracy
- **Classification Report**: Per-class precision, recall, and F1
- **Feature Importance**: Algorithm-specific importance scores (where available)

## Output Artifacts

### Model Files

- **best_model.pkl**: Model with highest validation performance
- **final_model.pkl**: Model state at training completion
- **preprocessing.pkl**: Complete scikit-learn preprocessing pipeline
- **feature_names.json**: Mapping of feature names and types

### Metrics and Reports

- **run_metrics.json**: Complete training metrics and metadata
- **epoch_metrics.csv**: Per-epoch training progression (MLP only)
- **classification_report**: Detailed per-class performance analysis
- **confusion_matrix**: Prediction accuracy matrix

### Aggregated Results

- **aggregate_metrics.json**: Statistical summary across all seeds
- **aggregate_summary.md**: Human-readable performance report
- **detailed_results.json**: Individual results from each seed

## Advanced Configuration

### Custom Model Parameters

Override default hyperparameters in configuration:

```yaml
model_params:
  logreg:
    max_iter: 500
    solver: lbfgs
    C: 1.0
  random_forest:
    n_estimators: 500
    max_depth: 20
    min_samples_split: 5
  mlp:
    hidden_layer_sizes: [128, 64, 32]
    activation: relu
    learning_rate_init: 0.001
    alpha: 0.0001
```

### Preprocessing Customization

Fine-tune preprocessing behavior:

```yaml
preprocessing:
  numeric_strategy: median     # or 'mean', 'constant'
  categorical_strategy: most_frequent  # or 'constant'
  missing_threshold: 0.1       # warn if >10% missing
  scale_features: true         # apply StandardScaler
  handle_unknown: ignore       # or 'error'
```

### Ensemble Configuration

Customize ensemble behavior:

```yaml
ensemble_params:
  voting: soft                 # or 'hard'
  estimators: [logreg, random_forest, mlp]
  weights: [0.3, 0.4, 0.3]    # custom weights
  n_jobs: 1                   # parallel processing
```

## Performance Optimization

### Memory Management

- **Batch Processing**: Configurable batch sizes for large datasets
- **Model Compression**: Optional model compression for deployment
- **Feature Selection**: Automatic removal of low-variance features

### Computational Efficiency

- **Parallel Processing**: Multi-core support where available
- **Early Termination**: Skip unnecessary epochs with early stopping
- **Caching**: Preprocessing pipeline reuse across seeds

### Scalability

- **Large Datasets**: Efficient handling of datasets with 100K+ samples
- **High-Dimensional Features**: Support for thousands of features
- **Class Imbalance**: Built-in handling via balanced class weights

## Troubleshooting

### Common Issues

1. **Missing Target Column**
   - **Error**: "Could not auto-detect target column"
   - **Solution**: Use `--target-column` to specify explicitly

2. **Insufficient Memory**
   - **Error**: Memory allocation failures during training
   - **Solution**: Reduce `--batch-size` or use fewer models

3. **Poor Convergence**
   - **Error**: MLP fails to converge or early stopping triggers immediately
   - **Solution**: Increase `--epochs` or adjust learning rate in config

4. **Class Imbalance**
   - **Error**: Poor performance on minority classes
   - **Solution**: Ensure `class_weight: balanced` in configuration

### Debugging Options

Enable verbose logging for detailed diagnostics:

```bash
python scripts/train_alzheimers_structured.py \
  --data-path data/alzheimer_dataset.csv \
  --verbose
```

Use dry-run mode to validate configuration:

```bash
python scripts/train_alzheimers_structured.py \
  --data-path data/alzheimer_dataset.csv \
  --dry-run
```

## Extending the Pipeline

### Adding New Algorithms

1. **Update Configuration**: Add algorithm to `models` list and `model_params`
2. **Extend `build_models()`**: Add algorithm instantiation logic
3. **Handle Training**: Ensure algorithm works with standard scikit-learn interface

### Custom Metrics

1. **Extend `evaluate()` Method**: Add metric computation logic
2. **Update Output**: Include new metrics in saved artifacts
3. **Modify Aggregation**: Update statistical aggregation for new metrics

### Integration with MLOps

- **MLflow Integration**: Log metrics and artifacts to MLflow tracking server
- **Model Registry**: Register best models in centralized model registry  
- **Automated Retraining**: Trigger retraining based on data drift detection
- **A/B Testing**: Compare model versions in production

## Best Practices

### Reproducibility

1. **Always specify seeds**: Use `--override-seeds` for consistent results
2. **Version configurations**: Track configuration changes with version control
3. **Document data versions**: Record dataset versions used for training

### Model Selection

1. **Use validation metrics**: Rely on `macro_f1` for balanced class evaluation
2. **Consider ensemble methods**: Often provide more robust performance
3. **Analyze per-class performance**: Ensure balanced performance across diagnostic categories

### Production Deployment

1. **Save complete pipelines**: Include preprocessing in saved artifacts
2. **Monitor data drift**: Track feature distributions over time
3. **Validate input data**: Ensure inference data matches training distribution

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/)
- [Model Evaluation Best Practices](https://scikit-learn.org/stable/modules/model_evaluation.html)

## Support

For issues and questions:

1. **Configuration Problems**: Review this documentation and example configurations
2. **Algorithm Questions**: Consult scikit-learn documentation for specific algorithms
3. **Performance Issues**: Use verbose logging and profiling tools for diagnosis

## Version History

- **v1.0**: Initial implementation with multi-algorithm support
- **v1.1**: Added ensemble methods and early stopping
- **v1.2**: Enhanced evaluation pipeline and visualization
- **v1.3**: Production-ready artifact management and MLOps integration