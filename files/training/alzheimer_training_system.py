#!/usr/bin/env python3
"""
Alzheimer Training System
Core training pipeline for Alzheimer's disease prediction
"""

# Re-export key functionality from the existing training modules
from .training import AlzheimerTrainer, TrainingConfig, TrainingIntegratedAgent, run_training_simulation
from .train_alzheimers import AlzheimerTrainingPipeline

# Common functions that are expected by the examples and documentation
def load_alzheimer_data(*args, **kwargs):
    """Load Alzheimer dataset - delegates to AlzheimerTrainer"""
    trainer = AlzheimerTrainer()
    return trainer.load_data()

def train_model(*args, **kwargs):
    """Train model - delegates to AlzheimerTrainer"""
    trainer = AlzheimerTrainer()
    df = trainer.load_data()
    X, y = trainer.preprocess_data(df)
    return trainer.train_model(X, y)

def load_model(*args, **kwargs):
    """Load trained model - delegates to AlzheimerTrainer"""
    trainer = AlzheimerTrainer()
    return trainer.model

# Export the main classes and functions
__all__ = [
    'AlzheimerTrainer', 
    'TrainingConfig', 
    'TrainingIntegratedAgent',
    'AlzheimerTrainingPipeline',
    'run_training_simulation',
    'load_alzheimer_data',
    'train_model', 
    'load_model'
]