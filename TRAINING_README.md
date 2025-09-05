# Training Module for DuetMind Adaptive

This document describes the training functionality for the duetmind_adaptive AI framework.

## Overview

The training module integrates machine learning capabilities with the existing adaptive neural network agents, specifically focusing on Alzheimer's disease prediction and assessment.

## Features

- **Alzheimer Dataset Loading**: Load datasets from Kaggle or use test data
- **Machine Learning Training**: Train Random Forest classifiers for disease prediction
- **Agent Integration**: Enhanced agents that can use ML predictions in their reasoning
- **Model Persistence**: Save and load trained models
- **Command-Line Interface**: Easy-to-use CLI for training operations

## Quick Start

### Basic Training

Run the complete training simulation:

```bash
python3 run_training.py
```

### Training Only (No Simulation)

Train a model without running the full agent simulation:

```bash
python3 run_training.py --mode train
```

### Using Custom Dataset

Train with your own Alzheimer dataset:

```bash
python3 run_training.py --data-path /path/to/your/dataset.csv
```

### Command-Line Options

- `--data-path`: Path to custom CSV dataset
- `--model-output`: Where to save the trained model (default: alzheimer_model.pkl)
- `--mode`: Choose between `train`, `simulate`, or `both` (default: both)
- `--verbose`: Enable detailed logging

## Dataset Format

The expected CSV format for Alzheimer datasets:

| Column | Description | Example Values |
|--------|-------------|----------------|
| age | Patient age | 65, 72, 58 |
| gender | Patient gender | M, F |
| education_level | Years of education | 12, 16, 18 |
| mmse_score | Mini-Mental State Exam score | 20-30 |
| cdr_score | Clinical Dementia Rating | 0.0, 0.5, 1.0, 2.0 |
| apoe_genotype | APOE genotype | E3/E3, E3/E4, E4/E4 |
| diagnosis | Target diagnosis | Normal, MCI, Dementia |

## Programming Interface

### Basic Usage

```python
from training import AlzheimerTrainer, TrainingIntegratedAgent

# Create and train a model
trainer = AlzheimerTrainer()
df = trainer.load_data()
X, y = trainer.preprocess_data(df)
results = trainer.train_model(X, y)

# Save the model
trainer.save_model("my_model.pkl")

# Make predictions
prediction = trainer.predict({
    'age': 72,
    'gender': 'F',
    'education_level': 12,
    'mmse_score': 24,
    'cdr_score': 0.5,
    'apoe_genotype': 'E3/E4'
})
```

### Enhanced Agents

```python
from neuralnet import AliveLoopNode, ResourceRoom

# Create an agent with ML capabilities
resource_room = ResourceRoom()
alive_node = AliveLoopNode((0,0), (0.5,0), 15.0, node_id=1)
agent = TrainingIntegratedAgent("MLAgent", {"logic": 0.8}, alive_node, resource_room, trainer)

# Use enhanced reasoning with ML
result = agent.enhanced_reason_with_ml(
    "Assess patient risk", 
    patient_features
)
```

## Model Performance

The training system provides several metrics:

- **Training Accuracy**: Performance on training data
- **Test Accuracy**: Performance on held-out test data
- **Feature Importance**: Which features are most predictive
- **Classification Report**: Detailed per-class metrics

## Integration with DuetMind Framework

The training module seamlessly integrates with the existing duetmind_adaptive components:

- **UnifiedAdaptiveAgent**: Enhanced with ML prediction capabilities
- **ResourceRoom**: Stores training data and model predictions
- **NetworkMetrics**: Includes ML confidence in health scoring
- **MazeMaster**: Can use ML insights for intervention decisions

## Testing

Run the training tests:

```bash
python3 -m pytest tests/test_training.py -v
```

## Files

- `training.py`: Core training functionality
- `run_training.py`: Command-line interface
- `tests/test_training.py`: Comprehensive tests
- `files/dataset/`: Dataset loading utilities

## Dependencies

- numpy
- pandas
- scikit-learn
- kagglehub (for dataset loading)
- pickle (for model persistence)

All dependencies are automatically installed when setting up the duetmind_adaptive environment.