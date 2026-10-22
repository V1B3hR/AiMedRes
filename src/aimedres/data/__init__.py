"""
AiMedRes Data Management & Open-Access Dataset Connectors
Provides zero-wait, leakage-safe dataset loaders for public clinical benchmarks (OASIS, PRO-ACT, PTB-XL, PPMI).
"""

from .open_datasets import (
    OASISDataset,
    PROACTALSDataset,
    PTBXLECGDataset,
    get_open_dataset,
    get_leakage_safe_cv,
)

__all__ = [
    "OASISDataset",
    "PROACTALSDataset",
    "PTBXLECGDataset",
    "get_open_dataset",
    "get_leakage_safe_cv",
]
