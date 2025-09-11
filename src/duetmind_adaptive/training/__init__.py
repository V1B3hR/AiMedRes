"""
DuetMind Adaptive Training Module

Secure training pipelines for neural networks and agents.
"""

from .trainer import SecureTrainer
from .pipeline import TrainingPipeline

__all__ = ["SecureTrainer", "TrainingPipeline"]