"""
Parkinson's Disease Scientific Research Package.
Implements Phase 2.1 decoupled, typed, leakage-free reproducibility pipeline.
"""

from .config import ParkinsonsPipelineConfig
from .manifest import compute_file_sha256, generate_experiment_manifest
from .pipeline import ParkinsonsResearchPipeline

__all__ = [
    "ParkinsonsPipelineConfig",
    "ParkinsonsResearchPipeline",
    "compute_file_sha256",
    "generate_experiment_manifest",
]
