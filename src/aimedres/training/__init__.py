"""
DuetMind Adaptive Training Module

Secure training pipelines for neural networks and agents, plus structured
Alzheimer's disease prediction models and disease-specific training pipelines.
"""

__all__ = []


def _export(name: str, value: object) -> None:
    """Expose an optional pipeline only when its import succeeded."""
    globals()[name] = value
    __all__.append(name)


try:
    from .pipeline import TrainingPipeline
    from .structured_alz_trainer import EarlyStoppingMLP, StructuredAlzTrainer
    from .trainer import SecureTrainer

    __all__.extend(["SecureTrainer", "TrainingPipeline", "StructuredAlzTrainer", "EarlyStoppingMLP"])
except ImportError:
    pass


# Keep optional disease pipelines independent: an unavailable imaging dependency
# must not remove unrelated training pipelines from the package namespace.
try:
    from .train_als import ALSTrainingPipeline

    _export("ALSTrainingPipeline", ALSTrainingPipeline)
except ImportError:
    pass

try:
    from .train_alzheimers import AlzheimerTrainingPipeline

    _export("AlzheimerTrainingPipeline", AlzheimerTrainingPipeline)
except ImportError:
    pass

try:
    from .train_brain_mri import BrainMRITrainingPipeline

    _export("BrainMRITrainingPipeline", BrainMRITrainingPipeline)
except ImportError:
    pass

try:
    from .train_cardiovascular import CardiovascularTrainingPipeline

    _export("CardiovascularTrainingPipeline", CardiovascularTrainingPipeline)
except ImportError:
    pass

try:
    from .train_diabetes import DiabetesTrainingPipeline

    _export("DiabetesTrainingPipeline", DiabetesTrainingPipeline)
except ImportError:
    pass

try:
    from .train_parkinsons import ParkinsonsTrainingPipeline

    _export("ParkinsonsTrainingPipeline", ParkinsonsTrainingPipeline)
except ImportError:
    pass
