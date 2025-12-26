"""
DuetMind Adaptive Training Module

Secure training pipelines for neural networks and agents, plus structured
Alzheimer's disease prediction models and disease-specific training pipelines.
"""

from .pipeline import TrainingPipeline
from .structured_alz_trainer import EarlyStoppingMLP, StructuredAlzTrainer
from .trainer import SecureTrainer

# Disease-specific training pipelines
try:
    from .train_als import ALSTrainingPipeline
    from .train_alzheimers import AlzheimerTrainingPipeline
    from .train_brain_mri import BrainMRITrainingPipeline
    from .train_cardiovascular import CardiovascularTrainingPipeline
    from .train_diabetes import DiabetesTrainingPipeline
    from .train_parkinsons import ParkinsonsTrainingPipeline
except ImportError:
    # These may not be available in all configurations
    pass

__all__ = [
    "SecureTrainer",
    "TrainingPipeline",
    "StructuredAlzTrainer",
    "EarlyStoppingMLP",
    "AlzheimerTrainingPipeline",
    "ALSTrainingPipeline",
    "BrainMRITrainingPipeline",
    "CardiovascularTrainingPipeline",
    "DiabetesTrainingPipeline",
    "ParkinsonsTrainingPipeline",
]
