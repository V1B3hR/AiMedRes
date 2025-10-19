# Brain MRI Training Guide

## Overview

This guide provides comprehensive information about training brain MRI classification models using the AiMedRes training pipeline.

## Training Script

The main training script is located at: `training/train_brain_mri.py`

### Features

- **Brain MRI Image Classification**: Binary classification of brain MRI images
- **Dataset**: Automatically downloads from Kaggle (14,715 images)
- **Model Architecture**: CNN with ~6.8M parameters
- **MLflow Integration**: Complete experiment tracking
- **Flexible Configuration**: Command-line arguments for customization

## Quick Start

### 1. Validate Installation

Before training, validate that all dependencies and the pipeline are working:

```bash
python validate_brain_mri_training.py
```

This will check:
- ✓ Required dependencies (PyTorch, torchvision, kagglehub, mlflow, etc.)
- ✓ Script syntax and imports
- ✓ Model creation and forward pass
- ✓ Data transforms
- ✓ Dataset loading (downloads if needed)

### 2. Run Demo Training

Test the training pipeline with a small subset of data:

```bash
python demo_brain_mri_training.py --max-samples 100 --epochs 2
```

This runs a quick demo (takes ~1-2 minutes) to verify everything works.

### 3. Full Training

Run the complete training pipeline:

```bash
# Default configuration (50 epochs)
python training/train_brain_mri.py

# Custom configuration
python training/train_brain_mri.py --epochs 20 --batch-size 32 --output-dir my_results

# With 3D CNN (for volumetric data)
python training/train_brain_mri.py --use-3d --epochs 30

# Custom MLflow experiment
python training/train_brain_mri.py --mlflow-experiment brain_mri_production
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `brain_mri_outputs` | Directory for saving results |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `32` | Batch size for training |
| `--use-3d` | `False` | Use 3D CNN instead of 2D CNN |
| `--mlflow-experiment` | `brain_mri_classification` | MLflow experiment name |

## Dataset

- **Source**: [Kaggle - Brain MRI Images](https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images)
- **Size**: 14,715 images
- **Format**: JPG (554×554 pixels, resized to 224×224)
- **Classes**: 2 (binary classification)
- **Download**: Automatic via kagglehub

## Model Architecture

### 2D CNN (Default)

- **Input**: RGB images (3, 224, 224)
- **Architecture**: 4 convolutional blocks + 3 fully connected layers
- **Parameters**: ~6.8 million
- **Output**: 2 classes

### 3D CNN (Optional)

- **Input**: Volumetric data (1, D, H, W)
- **Architecture**: 4 3D convolutional blocks + 3 fully connected layers
- **Use Case**: For volumetric brain MRI scans

## Training Pipeline

1. **Data Loading**: Downloads and caches dataset from Kaggle
2. **Preprocessing**: Resizes to 224×224, applies normalization
3. **Data Augmentation**: Random flips, rotations, color jitter
4. **Training**: Adam optimizer with ReduceLROnPlateau scheduler
5. **Early Stopping**: Stops if validation loss doesn't improve
6. **Model Saving**: Saves best and final models
7. **MLflow Tracking**: Logs all metrics, parameters, and artifacts

## Output Structure

```
brain_mri_outputs/
├── models/
│   ├── best_brain_mri_model.pth
│   └── final_brain_mri_model.pth
├── metrics/
│   └── training_metrics.json
└── logs/
    └── (training logs)
```

## Performance Notes

- **Hardware**: Runs on CPU (GPU recommended for faster training)
- **Training Time**: 
  - CPU: Several hours for 50 epochs with full dataset
  - GPU: Significantly faster
- **Memory**: ~4-8GB RAM recommended
- **Storage**: ~500MB for dataset

## Programmatic Usage

```python
from train_brain_mri import BrainMRITrainingPipeline

# Create pipeline
pipeline = BrainMRITrainingPipeline(output_dir='my_results')

# Train model
metrics = pipeline.train_model(
    epochs=50,
    batch_size=32,
    use_3d=False,
    mlflow_experiment='brain_mri_production'
)

# Check results
print(f"Best validation accuracy: {metrics['best_validation_accuracy']:.2f}%")
```

## Troubleshooting

### Issue: Dependencies Missing

**Solution**: Install required packages:
```bash
pip install torch torchvision numpy pandas scikit-learn Pillow kagglehub mlflow
```

### Issue: Dataset Download Fails

**Solution**: 
1. Ensure you have internet connection
2. Set up Kaggle credentials if needed
3. Dataset will be cached at `~/.cache/kagglehub/`

### Issue: Out of Memory

**Solution**: 
1. Reduce batch size: `--batch-size 16` or `--batch-size 8`
2. Close other applications
3. Use GPU if available

### Issue: Training Too Slow

**Solution**:
1. Use GPU: Install CUDA version of PyTorch
2. Reduce number of epochs for testing
3. Use demo script for quick validation

## Testing & Validation

### Validation Script

```bash
python validate_brain_mri_training.py
```

Validates:
- Dependencies installed
- Script compiles
- Models can be created
- Data transforms work
- Dataset can be loaded

### Demo Script

```bash
python demo_brain_mri_training.py --max-samples 50 --epochs 2
```

Runs quick training demo with subset of data.

## Bug Fixes Applied

### Fix: ReduceLROnPlateau verbose parameter

**Issue**: PyTorch 2.9+ removed the `verbose` parameter from `ReduceLROnPlateau`.

**Fix**: Removed `verbose=True` parameter from scheduler initialization.

**Location**: `training/train_brain_mri.py` line 306

## Best Practices

1. **Start with Demo**: Always run demo before full training
2. **Monitor MLflow**: Check experiment tracking during training
3. **Use Early Stopping**: Prevents overfitting (patience=10)
4. **Save Checkpoints**: Best model saved automatically
5. **Validation Split**: 20% of data for validation (stratified)
6. **Data Augmentation**: Enabled by default for better generalization

## Integration with AiMedRes

The brain MRI training pipeline integrates with:

- **MLflow**: Experiment tracking and model registry
- **CLI Tools**: Can be called via `aimedres train`
- **API Server**: Models can be deployed via API
- **Clinical Dashboard**: Results visible in dashboards

## References

- [Dataset](https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)

## Support

For issues or questions:
1. Check this guide and troubleshooting section
2. Run validation script to diagnose problems
3. Review MLflow logs for training issues
4. Check GitHub issues for known problems

---

**Status**: ✅ Fully functional and validated
**Last Updated**: 2025-10-19
