# Phase 7: Model Training & Evaluation - README

## Overview

Phase 7 implements comprehensive model training and evaluation capabilities for the AiMedRes debugging process. This phase focuses on training multiple models with cross-validation, recording detailed metrics, and comparing results with baseline models.

## Implementation

### Files Created

1. **`debug/phase7_model_training_evaluation.py`** - Main implementation
2. **`tests/test_phase7_model_training.py`** - Comprehensive test suite
3. **`demo_phase7_model_training.py`** - Demo script
4. **`debug/phase7_results.json`** - Results output (generated)
5. **Visualizations** (generated):
   - `debug/visualizations/phase7_model_training_evaluation.png`
   - `debug/visualizations/phase7_baseline_comparison.png`

## Features

### Subphase 7.1: Train Models with Cross-Validation

Trains 7 different models using stratified k-fold cross-validation:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM (RBF kernel)
- Multi-Layer Perceptron (MLP)
- Naive Bayes

**Metrics Tracked:**
- Training and test accuracy
- Precision (macro)
- Recall (macro)
- F1 score (macro)
- Overfitting gap detection
- Training time per model

### Subphase 7.2: Record Training, Validation, and Test Metrics

Records comprehensive metrics for each model:
- **Training metrics**: accuracy, precision (macro/weighted), recall (macro/weighted), F1 (macro/weighted), loss
- **Test metrics**: Same as training metrics
- **Additional outputs**: Confusion matrix, classification report

### Subphase 7.3: Compare Results with Baseline Models

Compares advanced models with simple baseline models:
- Baseline Logistic Regression
- Baseline Decision Tree (shallow)

**Comparison Features:**
- Side-by-side performance comparison
- Identification of best performers by:
  - Test accuracy
  - F1 score
  - Least overfitting
- Visual comparison charts

## Usage

### Basic Usage

```bash
# Run with default synthetic data
python debug/phase7_model_training_evaluation.py

# Run with verbose logging
python debug/phase7_model_training_evaluation.py --verbose

# Run with custom data source
python debug/phase7_model_training_evaluation.py --data-source path/to/data.csv
```

### Demo Script

```bash
python demo_phase7_model_training.py
```

### Python API

```python
from debug.phase7_model_training_evaluation import Phase7ModelTrainingEvaluator

# Create evaluator
evaluator = Phase7ModelTrainingEvaluator(verbose=True, data_source="synthetic")

# Run complete phase
results = evaluator.run_phase_7()

# Access results
print(f"Best model: {results['comparison_results']['comparison_summary']['best_by_accuracy']['model']}")
```

## Output

### Results JSON Structure

```json
{
  "timestamp": "ISO 8601 timestamp",
  "data_source": "synthetic | alzheimer | path/to/csv",
  "data_shape": {
    "n_samples": 1000,
    "n_features": 6
  },
  "cv_results": {
    "model_name": {
      "train_accuracy": {"mean": 0.xx, "std": 0.xx, "scores": [...]},
      "test_accuracy": {"mean": 0.xx, "std": 0.xx, "scores": [...]},
      "test_precision": {...},
      "test_recall": {...},
      "test_f1": {...},
      "overfitting_gap": 0.xx,
      "training_time": 1.23
    }
  },
  "metrics_results": {
    "metrics_record": {
      "model_name": {
        "training_metrics": {...},
        "test_metrics": {...},
        "confusion_matrix": [[...]],
        "classification_report": {...}
      }
    }
  },
  "comparison_results": {
    "baseline_results": {...},
    "comparison_summary": {
      "best_by_accuracy": {...},
      "best_by_f1": {...},
      "least_overfit": {...}
    }
  },
  "evaluation_metrics": [...]
}
```

### Visualizations

1. **phase7_model_training_evaluation.png**: 
   - 4-panel visualization showing:
     - Cross-validation accuracy with error bars
     - Cross-validation F1 scores with error bars
     - Train vs test accuracy comparison
     - Overfitting gap analysis

2. **phase7_baseline_comparison.png**:
   - Horizontal bar chart comparing F1 scores
   - Baseline models vs advanced models
   - Color-coded for easy identification

## Testing

Run tests:

```bash
# Run all Phase 7 tests
python -m unittest tests.test_phase7_model_training -v

# Run specific test
python -m unittest tests.test_phase7_model_training.TestPhase7ModelTrainingEvaluator.test_full_phase_7_run -v
```

## Requirements

- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0

## Key Achievements

✅ **Comprehensive model training** with 7 different algorithms
✅ **Cross-validation** using stratified k-fold
✅ **Detailed metrics** for training, validation, and test sets
✅ **Overfitting detection** with warnings
✅ **Baseline comparison** for performance benchmarking
✅ **Visual analysis** with charts and plots
✅ **JSON output** for easy integration
✅ **Full test coverage** with 14 test cases

## Integration with Other Phases

Phase 7 builds upon:
- **Phase 4**: Uses architecture recommendations
- **Phase 5**: Implements cross-validation strategies
- **Phase 6**: Can use hyperparameter tuning results

Phase 7 leads to:
- **Phase 8**: Model visualization & interpretability
- **Phase 9**: Error analysis & edge cases
- **Phase 10**: Final model validation

## Performance Considerations

- Training time varies by model (0.1s for Naive Bayes to 3-4s for MLP)
- Memory usage is modest (<500MB for 1000 samples)
- Supports parallel cross-validation (n_jobs=-1)
- Efficient handling of large datasets through batching

## Troubleshooting

### Common Issues

1. **MLP Convergence Warning**
   - Solution: Increase `max_iter` parameter or reduce model complexity

2. **Memory issues with large datasets**
   - Solution: Reduce `n_estimators` for tree-based models or use smaller cross-validation folds

3. **Slow training**
   - Solution: Reduce model complexity or use subset of data for initial testing

## Future Enhancements

- [ ] Support for deep learning models (PyTorch/TensorFlow)
- [ ] GPU acceleration for compatible models
- [ ] Distributed training for large-scale datasets
- [ ] Additional ensemble methods (Stacking, Blending)
- [ ] Automated hyperparameter optimization integration
- [ ] Real-time training monitoring dashboard

## Contributors

Implemented as part of the AiMedRes debugging process enhancement.

## License

Part of AiMedRes project. See main LICENSE file.
