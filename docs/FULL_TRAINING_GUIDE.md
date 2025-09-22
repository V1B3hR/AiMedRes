# Full Training Guide for DuetMind Adaptive

This guide explains how to run full training for the DuetMind Adaptive AI framework.

## Overview

The DuetMind Adaptive system now supports comprehensive training with multiple modes and data sources:

- **Basic Training**: Using synthetic test data
- **Kaggle Training**: Using real Alzheimer's MRI data from Kaggle
- **Simulation Training**: Training AI agents with adaptive behaviors
- **Comprehensive Training**: All components combined
- **Medical Training**: Specialized medical AI training

## Quick Start

### Run Full Comprehensive Training

```bash
python full_training.py --mode comprehensive
```

This will run all training components:
1. Basic training with test data
2. Extended training with hyperparameter optimization and cross-validation
3. Advanced training with multiple models and ensemble methods
4. Kaggle dataset training with real MRI data
5. Agent simulation training
6. Complete evaluation and summary

### Training Modes

```bash
# Basic training only (test data)
python full_training.py --mode basic

# Extended training with enhanced ML (hyperparameter tuning, cross-validation)
python full_training.py --mode extended

# Advanced training with multiple models and ensemble methods
python full_training.py --mode advanced

# Kaggle dataset training only
python full_training.py --mode kaggle

# Agent simulation only
python full_training.py --mode simulation

# Comprehensive training (all components)
python full_training.py --mode comprehensive

# Medical AI training
python full_training.py --mode medical

# Verbose output
python full_training.py --mode comprehensive --verbose
```

### Enhanced Training Features

#### Extended Training Mode
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Enhanced Metrics**: Comprehensive performance analysis
- **Model Selection**: Automatic selection of best parameters

#### Advanced Training Mode
- **Multiple Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM, Neural Networks
- **Ensemble Methods**: Voting classifier combining best models
- **Model Comparison**: Side-by-side performance comparison
- **Best Model Selection**: Automatic selection of top-performing model

### Original Training Interface

The original training interface is still available:

```bash
# Run both training and simulation
python run_training.py --mode both

# Training only
python run_training.py --mode train

# Simulation only
python run_training.py --mode simulate

# Custom data path
python run_training.py --data-path /path/to/data.csv

# Custom model output
python run_training.py --model-output my_model.pkl
```

### Alternative Training Entry Points

```bash
# Comprehensive training modes in files/training/
cd files/training
python train.py comprehensive

# Complete training with problem statement data
python run_training_complete.py
```

## Training Components

### 1. Basic Training
- Uses synthetic Alzheimer's test data
- Trains RandomForest classifier
- Saves model as `basic_alzheimer_model.pkl`
- Perfect for testing and development

### 2. Kaggle Training
- Downloads real MRI and Alzheimer's data from Kaggle
- Preprocesses and trains on clinical data
- Saves model as `alzheimer_mri_model.pkl`
- Provides clinical accuracy metrics

### 3. Agent Simulation Training
- Trains AI agents with ML prediction capabilities
- Integrates machine learning with agent reasoning
- Tests agent interaction and knowledge sharing
- Demonstrates adaptive behavior

### 4. Comprehensive Training
- Runs all training components
- Provides success metrics for each component
- Generates multiple trained models
- Complete evaluation and summary

## Output Files

After training, you'll have:

```
alzheimer_model.pkl              # Standard training model
basic_alzheimer_model.pkl        # Basic training model
alzheimer_mri_model.pkl          # Kaggle MRI data model
files/training/alzheimer_mri_model.pkl  # Problem statement model
```

## Requirements

Dependencies are automatically managed, but you can install manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
```

## Kaggle Setup

For Kaggle dataset training, you need:

1. Kaggle account and API credentials
2. Place `kaggle.json` in `~/.kaggle/` directory
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

If Kaggle is not available, the system falls back to test data.

## Training Results

### Success Indicators

- ✅ **PASSED**: Component completed successfully
- ❌ **FAILED**: Component encountered errors
- 📊 **Metrics**: Accuracy scores and feature importance
- 👥 **Agents**: Number of trained agents and their states

### Example Output

```
🎉 FULL TRAINING COMPLETED SUCCESSFULLY!
📊 Success Rate: 100% (3/3 components)

   BASIC: ✅ PASSED
   KAGGLE: ✅ PASSED  
   SIMULATION: ✅ PASSED
```

## Architecture

The training system integrates:

- **AlzheimerTrainer**: Core ML training class
- **TrainingIntegratedAgent**: AI agents with ML capabilities
- **UnifiedAdaptiveAgent**: Base agent framework
- **ResourceRoom**: Knowledge sharing system
- **AliveLoopNode**: Dynamic agent positioning

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Kaggle Access**: Check API credentials and internet connection
3. **Import Errors**: Ensure Python path includes repository root
4. **Memory Issues**: Use basic mode for limited resources

### Debug Mode

```bash
python full_training.py --mode comprehensive --verbose
```

### Manual Component Testing

```bash
# Test basic training only
python -c "from training import AlzheimerTrainer; t=AlzheimerTrainer(); df=t.load_data(); print('OK')"

# Test Kaggle access
python -c "import kagglehub; print('Kaggle available')"

# Test agent system
python -c "from training import run_training_simulation; print('Agents available')"
```

## Advanced Usage

### Custom Training Pipeline

```python
from training import AlzheimerTrainer, run_training_simulation

# Custom training
trainer = AlzheimerTrainer(data_path="my_data.csv")
df = trainer.load_data()
X, y = trainer.preprocess_data(df)
results = trainer.train_model(X, y)
trainer.save_model("my_model.pkl")

# Custom simulation
results, agents = run_training_simulation()
```

### Integration with DuetMind Framework

The training system seamlessly integrates with:
- Neural network framework
- Agent communication system
- Resource management
- Security and medical compliance

## Support

For issues or questions:
1. Check existing training documentation in `TRAINING_README.md`
2. Review training test files in `files/tests/`
3. Examine example implementations in `files/training/`