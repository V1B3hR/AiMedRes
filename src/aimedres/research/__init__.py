"""
AiMedRes Scientific Research Domain.
Contains decoupled, reproducible, leakage-free clinical disease pipelines.
"""

from .parkinsons import (
    ParkinsonsPipelineConfig,
    ParkinsonsResearchPipeline,
    compute_file_sha256,
    generate_experiment_manifest,
)
from .tracking import LocalExperimentTracker
from .datasets import (
    OASISDataset,
    PROACTALSDataset,
    PTBXLECGDataset,
    get_open_dataset,
    get_leakage_safe_cv,
)

__all__ = [
    "ParkinsonsPipelineConfig",
    "ParkinsonsResearchPipeline",
    "compute_file_sha256",
    "generate_experiment_manifest",
    "LocalExperimentTracker",
    "OASISDataset",
    "PROACTALSDataset",
    "PTBXLECGDataset",
    "get_open_dataset",
    "get_leakage_safe_cv",
]
